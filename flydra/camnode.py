#emacs, this is -*-Python-*- mode

"""

There are several ways we want to acquire data:

A) From live cameras (for indefinite periods).
B) From full-frame .fmf files (of known length).
C) From small-frame .ufmf files (of unknown length).
D) From a live image generator (for indefinite periods).
E) From a point generator (for indefinite periods).

The processing chain normally consists of:

0) Grab images from ImageSource. (This is not actually part of the chain).
1) Processing the images in ProcessCamClass
2) Save images in SaveCamData.
3) Save small .ufmf images in SaveSmallData.
4) Display images in DisplayCamData.

In cases B-E, some form of image/data control (play, stop, set fps)
must be settable. Ideally, this would be possible from a Python API
(for automated testing) and from a GUI (for visual debugging).

"""

from __future__ import division
from __future__ import with_statement

import os
BENCHMARK = int(os.environ.get('FLYDRA_BENCHMARK',0))
FLYDRA_BT = int(os.environ.get('FLYDRA_BT',0)) # threaded benchmark

NAUGHTY_BUT_FAST = False

#DISABLE_ALL_PROCESSING = True
DISABLE_ALL_PROCESSING = False

near_inf = 9.999999e20

bright_non_gaussian_cutoff = 255
bright_non_gaussian_replacement = 5

import threading, time, socket, sys, struct, warnings, optparse
import traceback
import Queue
import numpy
import numpy as nx
import numpy as np
import errno
import scipy.misc.pilutil
import numpy.dual
import json

import contextlib

import motmot.ufmf.ufmf as ufmf
import motmot.realtime_image_analysis.slow

#import flydra.debuglock
#DebugLock = flydra.debuglock.DebugLock

import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
import motmot.cam_iface.cam_iface_ctypes as cam_iface
import camnode_colors

import roslib;
roslib.load_manifest('sensor_msgs')
roslib.load_manifest('ros_flydra')
import sensor_msgs.msg
import std_msgs.msg

import ros_flydra.cv2_bridge
import ros_flydra.srv
from ros_flydra.srv import MainBrainGetVersion, MainBrainRegisterNewCamera
import rospy

if BENCHMARK:
    class NonExistantError(Exception):
        pass
    ConnectionClosedError = NonExistantError
import flydra.reconstruct_utils as reconstruct_utils
import flydra.reconstruct
import flydra.version
import flydra.rosutils
from flydra.reconstruct import do_3d_operations_on_2d_point

import camnode_utils
import motmot.FastImage.FastImage as FastImage
#FastImage.set_debug(3)
if os.name == 'posix' and sys.platform != 'darwin':
    import posix_sched

import flydra.debuglock
DebugLock = flydra.debuglock.DebugLock

LOG = flydra.rosutils.Log(to_ros=True)

def ros_ensure_valid_name(name):
    return name.replace('-','_')

class SharedValue:
    # in fview
    def __init__(self):
        self.evt = threading.Event()
        self._val = None
    def set(self,value):
        # called from producer thread
        self._val = value
        self.evt.set()
    def is_new_value_waiting(self):
        return self.evt.isSet()
    def get(self,*args,**kwargs):
        # called from consumer thread
        self.evt.wait(*args,**kwargs)
        val = self._val
        self.evt.clear()
        return val
    def get_nowait(self):
        # race condition here -- see comments in fview.py
        val = self._val
        self.evt.clear()
        return val

class SharedValue1(object):
    # in trackem
    def __init__(self,initial_value):
        self._val = initial_value
        #self.lock = DebugLock('SharedValue1')
        self.lock = threading.Lock()
    def get(self):
        self.lock.acquire()
        try:
            val = self._val
        finally:
            self.lock.release()
        return val
    def set(self,new_value):
        self.lock.acquire()
        try:
            self._val = new_value
        finally:
            self.lock.release()

class DummyMainBrain:
    def __init__(self,*args,**kw):
        self.set_image = self.noop
        self.log_message = self.noop
        self.close = self.noop
        self.camno = 0
    def noop(self,*args,**kw):
        return
    def register_new_camera(self,*args,**kw):
        self.camno += 1
        return 12345
    def get_and_clear_commands(self,*args,**kw):
        return {}

class ROSMainBrain:
    def __init__(self,*args,**kw):
        rospy.wait_for_service('/flydra_mainbrain/get_version')
        self._get_version = rospy.ServiceProxy('/flydra_mainbrain/get_version',
                                               MainBrainGetVersion)
        rospy.wait_for_service('/flydra_mainbrain/register_new_camera')
        self._register_new_camera = rospy.ServiceProxy('/flydra_mainbrain/register_new_camera',
                                                       MainBrainRegisterNewCamera)
        rospy.wait_for_service('/flydra_mainbrain/get_and_clear_commands')
        self._get_and_clear_commands = rospy.ServiceProxy('/flydra_mainbrain/get_and_clear_commands',
                                                          ros_flydra.srv.MainBrainGetAndClearCommands)
        rospy.wait_for_service('/flydra_mainbrain/set_image')
        self._set_image = rospy.ServiceProxy('/flydra_mainbrain/set_image',
                                             ros_flydra.srv.MainBrainSetImage)
        rospy.wait_for_service('/flydra_mainbrain/receive_missing_data')
        self._receive_missing_data = rospy.ServiceProxy('/flydra_mainbrain/receive_missing_data',
                                             ros_flydra.srv.MainBrainReceiveMissingData)
        rospy.wait_for_service('/flydra_mainbrain/close_camera')
        self._close_camera = rospy.ServiceProxy('/flydra_mainbrain/close_camera',
                                             ros_flydra.srv.MainBrainCloseCamera)

    def get_version(self):
        return self._get_version().version.data

    def register_new_camera(self, cam_guid, scalar_control_info, camnode_ros_name):
        hostname = socket.gethostname()
        my_ip = socket.gethostbyname(hostname)

        req = ros_flydra.srv.MainBrainRegisterNewCameraRequest()

        req.cam_guid = std_msgs.msg.String(cam_guid)
        req.scalar_control_info_json = std_msgs.msg.String(json.dumps(scalar_control_info))
        req.camnode_ros_name = std_msgs.msg.String(camnode_ros_name)
        req.cam_hostname = std_msgs.msg.String(hostname)
        req.cam_ip = std_msgs.msg.String(my_ip)

        response = self._register_new_camera(req)
        return response.port.data

    def get_and_clear_commands(self, cam_id):
        req = ros_flydra.srv.MainBrainGetAndClearCommandsRequest()
        req.cam_id = std_msgs.msg.String(cam_id)

        response = self._get_and_clear_commands(req)
        cmds_json = response.cmds_json.data
        cmds = json.loads(cmds_json)
        return cmds

    def set_image(self, cam_id, lb, arr):
        assert len(lb)==2

        arr = np.array(arr,copy=False)
        assert arr.ndim==2
        assert arr.dtype==np.uint8

        req = ros_flydra.srv.MainBrainSetImageRequest()
        req.cam_id = std_msgs.msg.String(cam_id)
        req.left = std_msgs.msg.Int32(lb[0])
        req.bottom = std_msgs.msg.Int32(lb[1])
        req.image = ros_flydra.cv2_bridge.numpy_to_imgmsg(arr)

        self._set_image(req)

    def receive_missing_data(self, cam_id, framenumber_offset, missing_data):
        req = ros_flydra.srv.MainBrainReceiveMissingDataRequest(
                    std_msgs.msg.String(cam_id),
                    std_msgs.msg.Int64(framenumber_offset),
                    std_msgs.msg.String(json.dumps( missing_data )))
        self._receive_missing_data(req)

    def close(self, cam_id):
        req = ros_flydra.srv.MainBrainCloseCameraRequest()
        req.cam_id = std_msgs.msg.String(cam_id)
        self._close_camera(req)

class DummySocket:
    def __init__(self,*args,**kw):
        self.connect = self.noop
        self.send = self.noop
        self.sendto = self.noop
    def noop(self,*args,**kw):
        return

import flydra.common_variables

import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time

def TimestampEcho():
    # create listening socket
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = ''
    port = flydra.common_variables.timestamp_echo_listener_port
    try:
        sockobj.bind(( hostname, port))
    except socket.error, err:
        if err.args[0]==98:
            warnings.warn('TimestampEcho not available because port in use')
            return
    sendto_port = flydra.common_variables.timestamp_echo_gatherer_port
    fmt = flydra.common_variables.timestamp_echo_fmt_diff
    while 1:
        try:
            buf, (orig_host,orig_port) = sockobj.recvfrom(4096)
        except socket.error, err:
            if err.args[0] == errno.EINTR: # interrupted system call
                continue
            raise

        if struct is None: # this line prevents bizarre interpreter shutdown errors
            return

        newbuf = buf + struct.pack( fmt, time.time() )
        sender.sendto(newbuf,(orig_host,sendto_port))

def stdout_write(x):
    while 1:
        try:
            sys.stdout.write(x)
            break
        except IOError, err:
            if err.args[0] == errno.EINTR: # interrupted system call
                continue
    while 1:
        try:
            sys.stdout.flush()
            break
        except IOError, err:
            if err.args[0] == errno.EINTR: # interrupted system call
                continue

L_i = nx.array([0,0,0,1,3,2])
L_j = nx.array([1,2,3,2,1,3])

def Lmatrix2Lcoords(Lmatrix):
    return Lmatrix[L_i,L_j]

def pluecker_from_verts(A,B):
    """
    See Hartley & Zisserman (2003) p. 70
    """
    if len(A)==3:
        A = A[0], A[1], A[2], 1.0
    if len(B)==3:
        B = B[0], B[1], B[2], 1.0
    A=nx.reshape(A,(4,1))
    B=nx.reshape(B,(4,1))
    L = nx.dot(A,nx.transpose(B)) - nx.dot(B,nx.transpose(A))
    return Lmatrix2Lcoords(L)

class PreallocatedBuffer(object):
    def __init__(self,size,pool):
        self._size = size
        self._buf = FastImage.FastImage8u(size)
        self._pool = pool
    def get_size(self):
        return self._size
    def get_buf(self):
        return self._buf
    def get_pool(self):
        return self._pool

class PreallocatedBufferPool(object):
    """One instance of this class for each camera. Threadsafe."""
    def __init__(self,size):
        self._lock = threading.Lock()
        # start: vars access controlled by self._lock
        self._allocated_pool = []
        #   end: vars access controlled by self._lock

        self.set_size(size)
        self._buffers_handed_out = 0 # self._zero_buffer_lock is set when this is 0
        self._zero_buffer_lock = threading.Event()
        self._zero_buffer_lock.set()

    def set_size(self,size):
        """size is FastImage.Size() instance"""
        assert isinstance(size,FastImage.Size)
        with self._lock:
            self._size = size
            del self._allocated_pool[:]

    def get_free_buffer(self):
        with self._lock:
            if len(self._allocated_pool):
                buffer = self._allocated_pool.pop()
            else:
                buffer = PreallocatedBuffer(self._size,self)
            self._buffers_handed_out += 1
            self._zero_buffer_lock.clear()
            return buffer

    def return_buffer(self,buffer):
        assert isinstance(buffer, PreallocatedBuffer)
        with self._lock:
            self._buffers_handed_out -= 1
            if buffer.get_size() == self._size:
                self._allocated_pool.append( buffer )

            if self._buffers_handed_out == 0:
                self._zero_buffer_lock.set()

    def get_num_outstanding_buffers(self):
        return self._buffers_handed_out

    def wait_for_0_outstanding_buffers(self,*args):
        self._zero_buffer_lock.wait(*args)

@contextlib.contextmanager
def get_free_buffer_from_pool(pool):
    """manage access to buffers from the pool"""
    buf = pool.get_free_buffer()
    buf._i_promise_to_return_buffer_to_the_pool = False
    try:
        yield buf
    finally:
        if not buf._i_promise_to_return_buffer_to_the_pool:
            pool.return_buffer(buf)

class ProcessCamClass(rospy.SubscribeListener):
    def __init__(self,
                 cam2mainbrain_port=None,
                 cam_id=None,
                 log_message_queue=None,
                 max_num_points=None,
                 roi2_radius=None,
                 bg_frame_interval=None,
                 bg_frame_alpha=None,
                 cam_no=-1,
                 main_brain_ipaddr=None,
                 mask_image=None,
                 diff_threshold_shared=None,
                 clear_threshold_shared=None,
                 n_sigma_shared=None,
                 n_erode_absdiff_shared=None,
                 color_range_1_shared=None,
                 color_range_2_shared=None,
                 color_range_3_shared=None,
                 sat_thresh_shared=None,
                 red_only_shared=None,
                 framerate = None,
                 lbrt=None,
                 max_height=None,
                 max_width=None,
                 globals = None,
                 options = None,
                 initial_image_dict = None,
                 benchmark = False,
                 ):

        self.ros_namespace = cam_id

        self.pub_img_n_subscribed = 0
        self.pub_img = rospy.Publisher('%s/image_raw'%self.ros_namespace,
                                         sensor_msgs.msg.Image,
                                         subscriber_listener=self,
                                         tcp_nodelay=True)
        self.pub_img_rate = float(options.rosrate)
        self.pub_img_lasttime = time.time()

        self.pub_rate = rospy.Publisher('%s/framerate'%self.ros_namespace,
                                         std_msgs.msg.Float32,
                                         tcp_nodelay=True)
        self.pub_rate_lasttime = time.time()
        self.pub_rate_lastframe = 0

        self.benchmark = benchmark
        self.options = options
        self.globals = globals
        self.main_brain_ipaddr = main_brain_ipaddr
        if framerate is not None:
            self.shortest_IFI = 1.0/framerate
        else:
            self.shortest_IFI = numpy.inf
        self.cam2mainbrain_port = cam2mainbrain_port
        self.cam_id = cam_id
        self.log_message_queue = log_message_queue

        self.bg_frame_alpha = bg_frame_alpha
        self.bg_frame_interval = bg_frame_interval

        self.diff_threshold_shared = diff_threshold_shared
        self.clear_threshold_shared = clear_threshold_shared
        self.n_sigma_shared = n_sigma_shared
        self.n_erode_absdiff_shared = n_erode_absdiff_shared
        self.red_only_shared = red_only_shared

        self.color_range_1_shared = color_range_1_shared
        self.color_range_2_shared = color_range_2_shared
        self.color_range_3_shared = color_range_3_shared
        self.sat_thresh_shared = sat_thresh_shared

        self.new_roi = threading.Event()
        self.new_roi_data = None
        self.new_roi_data_lock = threading.Lock()

        self.max_height = max_height
        self.max_width = max_width

        if mask_image is None:
            mask_image = numpy.zeros( (self.max_height,
                                       self.max_width),
                                      dtype=numpy.bool )
            # mask is currently an array of bool
            mask_image = mask_image.astype(numpy.uint8)*255
        self.mask_image = mask_image
        self.max_num_points=max_num_points

        self.realtime_analyzer = realtime_image_analysis.RealtimeAnalyzer(lbrt,
                                                                          self.max_width,
                                                                          self.max_height,
                                                                          self.max_num_points,
                                                                          roi2_radius,
                                                                          )
        self.realtime_analyzer.diff_threshold = self.diff_threshold_shared.get_nowait()
        self.realtime_analyzer.clear_threshold = self.clear_threshold_shared.get_nowait()

        self._hlper = None
        self._pmat = None

        self._chain = camnode_utils.ChainLink()
        self._initial_image_dict = initial_image_dict

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        self.pub_img_n_subscribed += 1

    def peer_unsubscribe(self, topic_name, num_peers):
        self.pub_img_n_subscribed -= 1

    def get_chain(self):
        return self._chain

    def get_roi(self):
        return self.realtime_analyzer.roi
    def set_roi(self, lbrt):
        with self.new_roi_data_lock:
            self.new_roi_data = lbrt
            self.new_roi.set()
    roi = property( get_roi, set_roi )

    def get_pmat(self):
        return self._pmat
    def set_pmat(self,value):
        if value is None:
            self._pmat = None
            self._camera_center = None
            self._pmat_inv = None
            return

        self._pmat = numpy.asarray(value)

        P = self._pmat
        determinant = numpy.dual.det
        r_ = numpy.r_

        # find camera center in 3D world coordinates
        col0_asrow = P[nx.newaxis,:,0]
        col1_asrow = P[nx.newaxis,:,1]
        col2_asrow = P[nx.newaxis,:,2]
        col3_asrow = P[nx.newaxis,:,3]
        X = determinant(  r_[ col1_asrow, col2_asrow, col3_asrow ] )
        Y = -determinant( r_[ col0_asrow, col2_asrow, col3_asrow ] )
        Z = determinant(  r_[ col0_asrow, col1_asrow, col3_asrow ] )
        T = -determinant( r_[ col0_asrow, col1_asrow, col2_asrow ] )

        self._camera_center = nx.array( [ X/T, Y/T, Z/T, 1.0 ] )
        self._pmat_inv = numpy.dual.pinv(self._pmat)

    def make_reconstruct_helper(self, intlin, intnonlin):
        if intlin is None and intnonlin is None:
            self._hlper = None
            return

        fc1 = intlin[0,0]
        fc2 = intlin[1,1]
        cc1 = intlin[0,2]
        cc2 = intlin[1,2]
        k1, k2, p1, p2 = intnonlin

        self._hlper = reconstruct_utils.ReconstructHelper(
            fc1, fc2, cc1, cc2, k1, k2, p1, p2 )

    def _convert_to_wire_order(self, xpoints, hw_roi_frame, running_mean_im, sumsqf ):
        """the images passed in are already in roi coords, as are index_x and index_y.
        convert to values for sending.
        """
        points = []
        hw_roi_frame = numpy.asarray( hw_roi_frame )
        for xpt in xpoints:
            try:
                (x0_abs, y0_abs, area, slope, eccentricity, index_x, index_y) = xpt
            except:
                LOG.warn('converting xpt %s' % xpt)
                raise

            # Find values at location in image that triggered
            # point. Cast to Python int and floats.
            cur_val = int(hw_roi_frame[index_y,index_x])
            mean_val = float(running_mean_im[index_y, index_x])
            sumsqf_val = float(sumsqf[index_y, index_x])

            if numpy.isnan(slope):
                run = numpy.nan
                line_found = False
            else:
                line_found = True
                if numpy.isinf(slope):
                    run = 0
                    rise = 1
                else:
                    run = 1
                    #slope = rise/run
                    rise = slope

            ray_valid = False
            if self._hlper is not None:
                x0u, y0u = self._hlper.undistort( x0_abs, y0_abs )
                if line_found:

                    # (If we have self._hlper _pmat_inv, we can assume we have
                    # self._pmat_inv and sef._camera_center.)
                    (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4,
                     ray5) = do_3d_operations_on_2d_point(self._hlper,x0u,y0u,
                                                          self._pmat_inv,
                                                          self._camera_center,
                                                          x0_abs, y0_abs,
                                                          rise, run)
                    ray_valid = True
            else:
                x0u = x0_abs # fake undistorted data
                y0u = y0_abs

            if not ray_valid:
                p1,p2,p3,p4 = -1, -1, -1, -1 # sentinel value (will be converted to nan)
                (ray0, ray1, ray2, ray3, ray4, ray5) = (0,0,0, 0,0,0)

            slope_found = True
            if numpy.isnan(slope):
                # prevent nan going across network
                slope_found = False
                slope = 0.0

            if numpy.isinf(eccentricity):
                eccentricity = near_inf

            if numpy.isinf(slope):
                slope = near_inf

            # see flydra.common_variables.recv_pt_fmt struct definition:
            pt = (x0_abs, y0_abs, area, slope, eccentricity,
                  p1, p2, p3, p4, line_found, slope_found,
                  x0u, y0u,
                  ray_valid,
                  ray0, ray1, ray2, ray3, ray4, ray5,
                  cur_val, mean_val, sumsqf_val)
            points.append( pt )
        return points

    def _service_ros(self, framenumber, hw_roi_frame, chainbuf):
        now = time.time()

        if (now - self.pub_rate_lasttime) > 2.0:
            self.pub_rate_lasttime = now
            fps = (framenumber - self.pub_rate_lastframe) / 2.0
            self.pub_rate_lastframe = framenumber
            self.pub_rate.publish(fps)

        #maybe this is racy, but its only for debugging. Don't serialize images
        #if noone is subscribed
        if self.pub_img_n_subscribed <= 0:
            return

        if self.pub_img_rate <= 0:
            return

        if now-self.pub_img_lasttime+0.005 > 1./(self.pub_img_rate):

            msg = sensor_msgs.msg.Image()
            msg.header.seq=framenumber
            msg.header.stamp=rospy.Time.from_sec(now) # XXX TODO: once camera trigger is ROS node, get accurate timestamp
            msg.header.frame_id = "0"

            npbuf = np.array(hw_roi_frame)
            (height,width) = npbuf.shape

            msg.height = height
            msg.width = width
            msg.encoding = chainbuf.image_coding
            pixel_format = chainbuf.image_coding
            if pixel_format == 'MONO8':
                msg.encoding = 'mono8'
            elif pixel_format in ('RAW8:RGGB','MONO8:RGGB'):
                msg.encoding = 'bayer_rggb8'
            elif pixel_format in ('RAW8:BGGR','MONO8:BGGR'):
                msg.encoding = 'bayer_bggr8'
            elif pixel_format in ('RAW8:GBRG','MONO8:GBRG'):
                msg.encoding = 'bayer_gbrg8'
            elif pixel_format in ('RAW8:GRBG','MONO8:GRBG'):
                msg.encoding = 'bayer_grbg8'
            else:
                raise ValueError('unknown pixel format "%s"'%pixel_format)

            msg.step = width
            msg.data = npbuf.tostring() # let numpy convert to string

            self.pub_img.publish(msg)
            self.pub_img_lasttime = now

    def mainloop(self):

        disable_ifi_warning = self.options.disable_ifi_warning
        globals = self.globals

        self._globals = globals

        # questionable optimization: speed up by eliminating namespace lookups
        bg_frame_number = -1
        clear_background_isSet = globals['clear_background'].isSet
        clear_background_clear = globals['clear_background'].clear
        take_background_isSet = globals['take_background'].isSet
        take_background_clear = globals['take_background'].clear
        collecting_background_isSet = globals['collecting_background'].isSet


        max_frame_size = FastImage.Size(self.max_width, self.max_height)

        lbrt = self.realtime_analyzer.roi
        l,b,r,t=lbrt
        hw_roi_w = r-l+1
        hw_roi_h = t-b+1
        cur_roi_l = l
        cur_roi_b = b
        #cur_roi_l, cur_roi_b,hw_roi_w, hw_roi_h  = self.cam.get_frame_roi()
        cur_fisize = FastImage.Size(hw_roi_w, hw_roi_h)

        bg_changed = True
        use_roi2 = True
        fi8ufactory = FastImage.FastImage8u
        use_cmp_isSet = globals['use_cmp'].isSet

#        hw_roi_frame = fi8ufactory( cur_fisize )
#        self._hw_roi_frame = hw_roi_frame # make accessible to other code

        if self.benchmark:
            coord_socket = DummySocket()
        else:
            coord_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        old_ts = time.time()
        old_fn = None
        points = []

        if os.name == 'posix' and not BENCHMARK:
            try:
                max_priority = posix_sched.get_priority_max( posix_sched.FIFO )
                sched_params = posix_sched.SchedParam(max_priority)
                posix_sched.setscheduler(0, posix_sched.FIFO, sched_params)
                msg = 'excellent, grab thread running in maximum prioity mode'
            except Exception, x:
                msg = 'WARNING: could not run in maximum priority mode:', str(x)
            self.log_message_queue.put((self.cam_id,time.time(),msg))
            LOG.info(msg)

        #FastImage.set_debug(3) # let us see any images malloced, should only happen on hardware ROI size change


        #################### initialize images ############

        running_mean8u_im_full = self.realtime_analyzer.get_image_view('mean') # this is a view we write into
        absdiff8u_im_full = self.realtime_analyzer.get_image_view('absdiff') # this is a view we write into

        mask_im = self.realtime_analyzer.get_image_view('mask') # this is a view we write into
        newmask_fi = FastImage.asfastimage( self.mask_image )
        newmask_fi.get_8u_copy_put(mask_im, max_frame_size)

        # allocate images and initialize if necessary

        running_mean_im_full = FastImage.FastImage32f(max_frame_size)
        self._running_mean_im_full = running_mean_im_full # make accessible to other code

        fastframef32_tmp_full = FastImage.FastImage32f(max_frame_size)

        mean2_full = FastImage.FastImage32f(max_frame_size)
        self._mean2_full = mean2_full # make accessible to other code
        std2_full = FastImage.FastImage32f(max_frame_size)
        self._std2_full = std2_full # make accessible to other code
        running_stdframe_full = FastImage.FastImage32f(max_frame_size)
        self._running_stdframe_full = running_stdframe_full # make accessible to other code
        compareframe_full = FastImage.FastImage32f(max_frame_size)
        compareframe8u_full = self.realtime_analyzer.get_image_view('cmp') # this is a view we write into
        self._compareframe8u_full = compareframe8u_full

        running_sumsqf_full = FastImage.FastImage32f(max_frame_size)
        running_sumsqf_full.set_val(0,max_frame_size)
        self._running_sumsqf_full = running_sumsqf_full # make accessible to other code

        noisy_pixels_mask_full = FastImage.FastImage8u(max_frame_size)

        # set ROI views of full-frame images
        running_mean8u_im = running_mean8u_im_full.roi(cur_roi_l, cur_roi_b, cur_fisize) # set ROI view
        running_mean_im = running_mean_im_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        fastframef32_tmp = fastframef32_tmp_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        mean2 = mean2_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        std2 = std2_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        running_stdframe = running_stdframe_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        compareframe = compareframe_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        compareframe8u = compareframe8u_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        running_sumsqf = running_sumsqf_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view
        noisy_pixels_mask = noisy_pixels_mask_full.roi(cur_roi_l, cur_roi_b, cur_fisize)  # set ROI view

        if self._initial_image_dict is not None:
            # If we have initial values, load them.

            # implicit conversion to float32
            numpy.asarray(running_mean_im_full)[:,:] = self._initial_image_dict['mean']
            numpy.asarray(running_sumsqf)[:,:] = self._initial_image_dict['sumsqf']

            LOG.warn('WARNING: ignoring initial images and taking new background')
            globals['take_background'].set()

        else:
            globals['take_background'].set()

        running_mean_im.get_8u_copy_put( running_mean8u_im, cur_fisize )

        #################### done initializing images ############

        incoming_raw_frames_queue = globals['incoming_raw_frames']
        incoming_raw_frames_queue_put = incoming_raw_frames_queue.put

        initial_take_bg_state = None

        while 1:

            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf:
                if chainbuf.quit_now:
                    break
                chainbuf.updated_running_mean_image = None
                chainbuf.updated_running_sumsqf_image = None

                hw_roi_frame = chainbuf.get_buf()
                cam_received_time = chainbuf.cam_received_time

                if self.red_only_shared.get_nowait():
                    color_range_1 = self.color_range_1_shared.get_nowait()
                    color_range_2 = self.color_range_2_shared.get_nowait()
                    color_range_3 = self.color_range_3_shared.get_nowait()

                    if color_range_1 < color_range_2:

                        camnode_colors.replace_with_red_image( hw_roi_frame,
                                                               chainbuf.image_coding,
                                                               #camnode_colors.RED_CHANNEL)
                                                               camnode_colors.RED_COLOR,
                                                               color_range_1,
                                                               color_range_2,
                                                               color_range_3,
                                                               self.sat_thresh_shared.get_nowait())
                    else:
                        LOG.error('ERROR: color_range_2 >= color_range_1 -- skipping')

                # get best guess as to when image was taken
                timestamp=chainbuf.timestamp
                framenumber=chainbuf.framenumber

                # publish on ROS network
                if not self.benchmark:
                    self._service_ros(framenumber, hw_roi_frame, chainbuf)

                if 1:
                    if old_fn is None:
                        # no old frame
                        old_fn = framenumber-1
                    if framenumber-old_fn > 1:
                        n_frames_skipped = framenumber-old_fn-1
                        msg = '  %s frames apparently skipped: %d (%d vs %d)'%(self.cam_id, n_frames_skipped, framenumber, old_fn)
                        self.log_message_queue.put((self.cam_id,time.time(),msg))
                        LOG.warn(msg)
                    else:
                        n_frames_skipped = 0

                    diff = timestamp-old_ts
                    time_per_frame = diff/(n_frames_skipped+1)
                    if not disable_ifi_warning:
                        if time_per_frame > 2*self.shortest_IFI:
                            msg = 'Warning: IFI is %f on %s at %s (frame skipped?)'%(time_per_frame,self.cam_id,time.asctime())
                            self.log_message_queue.put((self.cam_id,time.time(),msg))
                            LOG.warn(msg)

                old_ts = timestamp
                old_fn = framenumber

                #print 'erode value', self.n_erode_absdiff_shared.get_nowait()
                xpoints = self.realtime_analyzer.do_work(hw_roi_frame,
                                                         timestamp, framenumber, use_roi2,
                                                         use_cmp_isSet(),
                                                         #max_duration_sec=0.010, # maximum 10 msec in here
                                                         max_duration_sec=self.shortest_IFI-0.0005, # give .5 msec for other processing
                                                         return_debug_values=1,
                                                         n_erode_absdiff=self.n_erode_absdiff_shared.get_nowait(),
                                                         )
                ## if len(xpoints)>=self.max_num_points:
                ##     msg = 'Warning: cannot save acquire points this frame because maximum number already acheived'
                ##     LOG.warn(msg)
                chainbuf.processed_points = xpoints
                if NAUGHTY_BUT_FAST:
                    chainbuf.absdiff8u_im_full = absdiff8u_im_full
                    chainbuf.mean8u_im_full = running_mean8u_im_full
                    chainbuf.compareframe8u_full = compareframe8u_full
                else:
                    chainbuf.absdiff8u_im_full = numpy.array(absdiff8u_im_full,copy=True)
                    chainbuf.mean8u_im_full = numpy.array(running_mean8u_im_full,copy=True)
                    chainbuf.compareframe8u_full = numpy.array(compareframe8u_full,copy=True)
                points = self._convert_to_wire_order( xpoints, hw_roi_frame, running_mean_im, running_sumsqf)

                # allow other thread to see images
                imname = globals['export_image_name'] # figure out what is wanted # XXX theoretically could have threading issue
                if imname == 'raw':
                    export_image = hw_roi_frame
                else:
                    export_image = self.realtime_analyzer.get_image_view(imname) # get image
                globals['most_recent_frame_potentially_corrupt'] = (0,0), export_image # give view of image, receiver must be careful

                if 1:
                    # allow other thread to see raw image always (for saving)
                    if incoming_raw_frames_queue.qsize() >1000:
                        # chop off some old frames to prevent memory explosion
                        LOG.warn('ERROR: deleting old frames to make room for new ones! (and sleeping)')
                        for i in range(100):
                            incoming_raw_frames_queue.get_nowait()
                    incoming_raw_frames_queue_put(
                        (hw_roi_frame.get_8u_copy(hw_roi_frame.size), # save a copy
                         timestamp,
                         framenumber,
                         points,
                         self.realtime_analyzer.roi,
                         cam_received_time,
                         ) )

                do_bg_maint = False

                if initial_take_bg_state is not None:
                    assert initial_take_bg_state == 'gather'
                    n_initial_take = 20
                    if 1:
                        initial_take_frames.append( numpy.array(hw_roi_frame,copy=True) )
                        if len( initial_take_frames ) >= n_initial_take:

                            initial_take_frames = numpy.array( initial_take_frames, dtype=numpy.float32 )
                            mean_frame = numpy.mean( initial_take_frames, axis=0)
                            sumsqf_frame = numpy.sum(initial_take_frames**2, axis=0)/len( initial_take_frames )

                            numpy.asarray(running_mean_im)[:,:] = mean_frame
                            numpy.asarray(running_sumsqf)[:,:] = sumsqf_frame
                            LOG.info('using slow method, calculated mean and sumsqf frames from first %d frames' % n_initial_take)

                            # we're done with initial transient, set stuff
                            do_bg_maint = True
                            initial_take_bg_state = None
                            del initial_take_frames
                    elif 0:
                        # faster approach (currently seems broken)

                        # accummulate sum

                        # I could re-write this to use IPP instead of
                        # numpy, but would that really matter much?
                        npy_view =  numpy.asarray(hw_roi_frame)
                        numpy.asarray(running_mean_im)[:,:] = numpy.asarray(running_mean_im) +  npy_view
                        numpy.asarray(running_sumsqf)[:,:]  = numpy.asarray(running_sumsqf)  +  npy_view.astype(numpy.float32)**2
                        initial_take_frames_done += 1
                        del npy_view

                        if initial_take_frames_done >= n_initial_take:

                            # now divide to take average
                            numpy.asarray(running_mean_im)[:,:] = numpy.asarray(running_mean_im) / initial_take_frames_done
                            numpy.asarray(running_sumsqf)[:,:]  = numpy.asarray(running_sumsqf) / initial_take_frames_done

                            # we're done with initial transient, set stuff
                            do_bg_maint = True
                            initial_take_bg_state = None
                            del initial_take_frames_done

                if take_background_isSet():
                    LOG.info('taking new bg')
                    # reset background image with current frame as mean and 0 STD
                    if cur_fisize != max_frame_size:
                        LOG.warn('ERROR: can only take background image if not using ROI')
                    else:
                        if 0:
                            # old way
                            hw_roi_frame.get_32f_copy_put(running_sumsqf,max_frame_size)
                            running_sumsqf.toself_square(max_frame_size)

                            hw_roi_frame.get_32f_copy_put(running_mean_im,cur_fisize)
                            running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )
                            do_bg_maint = True
                        else:
                            initial_take_bg_state = 'gather'
                            if 1:
                                initial_take_frames = [ numpy.array(hw_roi_frame,copy=True) ] # for slow approach
                            elif 0:

                                initial_take_frames_done = 1 # for faster approach

                                # set running_mean_im
                                hw_roi_frame.get_32f_copy_put(running_mean_im,cur_fisize)
                                running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )

                                # set running_sumsqf
                                hw_roi_frame.get_32f_copy_put(running_sumsqf,max_frame_size)
                                running_sumsqf.toself_square(max_frame_size)

                    take_background_clear()

                if collecting_background_isSet():
                    bg_frame_number += 1
                    if (bg_frame_number % self.bg_frame_interval == 0):
                        do_bg_maint = True

                if do_bg_maint:
                    realtime_image_analysis.do_bg_maint(
                    #print 'doing slow bg maint, frame', chainbuf.framenumber
                    #tmpresult = motmot.realtime_image_analysis.slow.do_bg_maint(
                        running_mean_im,#in
                        hw_roi_frame,#in
                        cur_fisize,#in
                        self.bg_frame_alpha, #in
                        running_mean8u_im,
                        fastframef32_tmp,
                        running_sumsqf, #in
                        mean2,
                        std2,
                        running_stdframe,
                        self.n_sigma_shared.get_nowait(),#in
                        compareframe8u,
                        bright_non_gaussian_cutoff,#in
                        noisy_pixels_mask,#in
                        bright_non_gaussian_replacement,#in
                        bench=0 )
                        #debug=0)
                    #chainbuf.real_std_est= tmpresult
                    bg_changed = True
                    bg_frame_number = 0

                if self.options.debug_std:
                    if framenumber % 200 == 0:
                        mean_std = numpy.mean( numpy.mean( numpy.array(running_stdframe,dtype=numpy.float32 )))
                        LOG.info('%s mean STD %.2f'%(self.cam_id,mean_std))

                if clear_background_isSet():
                    # reset background image with 0 mean and 0 STD
                    running_mean_im.set_val( 0, max_frame_size )
                    running_mean8u_im.set_val(0, max_frame_size )
                    running_sumsqf.set_val( 0, max_frame_size )
                    compareframe8u.set_val(0, max_frame_size )
                    bg_changed = True
                    clear_background_clear()

                if bg_changed:
                    chainbuf.updated_running_mean_image = numpy.array( running_mean_im, copy=True )
                    chainbuf.updated_running_sumsqf_image = numpy.array( running_sumsqf, copy=True )
                    bg_changed = False

                if self.diff_threshold_shared.is_new_value_waiting():
                    self.realtime_analyzer.diff_threshold = (
                        self.diff_threshold_shared.get_nowait() )

                if self.clear_threshold_shared.is_new_value_waiting():
                    self.realtime_analyzer.clear_threshold = (
                        self.clear_threshold_shared.get_nowait() )

                # XXX could speed this with a join operation I think
                data = struct.pack(flydra.common_variables.recv_pt_header_fmt,
                                   timestamp,cam_received_time,
                                   framenumber,len(points),n_frames_skipped)
                for point_tuple in points:
                    try:
                        data = data + struct.pack(flydra.common_variables.recv_pt_fmt,*point_tuple)
                    except:
                        LOG.warn('error-causing data: %s' % (point_tuple,))
                        raise
                if 0:
                    local_processing_time = (time.time()-cam_received_time)*1e3
                    LOG.debug('local_processing_time % 3.1f'%local_processing_time)

                try:
                    coord_socket.sendto(data,(self.main_brain_ipaddr,self.cam2mainbrain_port))
                except socket.error:
                    LOG.warn('WARNING: ignoring error: %s' % traceback.format_exc())

                if 0 and self.new_roi.isSet():
                    with self.new_roi_data_lock:
                        lbrt = self.new_roi_data
                        self.new_roi_data = None
                        self.new_roi.clear()
                    l,b,r,t=lbrt
                    w = r-l+1
                    h = t-b+1
                    self.realtime_analyzer.roi = lbrt
                    LOG.info('desired l,b,w,h %s,%s,%s,%s' % (l,b,w,h))

                    l2,b2,w2,h2 = self.cam.get_frame_roi()
                    if ((l==l2) and (b==b2) and (w==w2) and (h==h2)):
                        LOG.info('current ROI matches desired ROI - not changing')
                    else:
                        self.cam.set_frame_roi(l,b,w,h)
                        l,b,w,h = self.cam.get_frame_roi()
                        LOG.info('actual l,b,w,h %s,%s,%s,%s' % (l,b,w,h))
                    r = l+w-1
                    t = b+h-1
                    cur_fisize = FastImage.Size(w, h)
                    hw_roi_frame = fi8ufactory( cur_fisize )
                    self.realtime_analyzer.roi = (l,b,r,t)

                    # set ROI views of full-frame images
                    running_mean8u_im = running_mean8u_im_full.roi(l, b, cur_fisize) # set ROI view
                    running_mean_im = running_mean_im_full.roi(l, b, cur_fisize)  # set ROI view
                    fastframef32_tmp = fastframef32_tmp_full.roi(l, b, cur_fisize)  # set ROI view
                    mean2 = mean2_full.roi(l, b, cur_fisize)  # set ROI view
                    std2 = std2_full.roi(l, b, cur_fisize)  # set ROI view
                    running_stdframe = running_stdframe_full.roi(l, b, cur_fisize)  # set ROI view
                    compareframe8u = compareframe8u_full.roi(l, b, cur_fisize)  # set ROI view
                    running_sumsqf = running_sumsqf_full.roi(l, b, cur_fisize)  # set ROI view
                    noisy_pixels_mask = noisy_pixels_mask_full.roi(l, b, cur_fisize)  # set ROI view

class FakeProcessCamData(object):
    def __init__(self,cam_id=None):
        self._chain = camnode_utils.ChainLink()
        self._cam_id = cam_id
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            with camnode_utils.use_buffer_from_chain(self._chain) as buf:
                #stdout_write('P')
                buf.processed_points = [ (10,20) ]

class SaveCamData(object):
    def __init__(self,cam_id=None,quit_event=None):
        self._chain = camnode_utils.ChainLink()
        self._cam_id = cam_id
        self.cmd = Queue.Queue()
    def get_chain(self):
        return self._chain
    def start_recording(self,
                        raw_file_basename = None):
        """threadsafe"""
        self.cmd.put( ('save',raw_file_basename) )

    def stop_recording(self,*args,**kw):
        """threadsafe"""
        self.cmd.put( ('stop',) )

    def mainloop(self):
        # Note: need to accummulate frames into queue and add with .add_frames() for speed
        # Also: old version uses fmf version 1. Not sure why.

        raw = []
        meancmp = []

        state = 'pass'

        last_bgcmp_image_timestamp = None
        last_running_mean_image = None
        last_running_sumsqf_image = None

        image_coding = None

        while 1:

            # 1: process commands
            while 1:
                if self.cmd.empty():
                    break
                cmd = self.cmd.get()
                if cmd[0] == 'save':
                    raw_file_basename = cmd[1]
                    full_raw = raw_file_basename + '.fmf'
                    full_bg = raw_file_basename + '_mean.fmf'
                    full_std = raw_file_basename + '_sumsqf.fmf'
                    LOG.info('saving movies to %s' % raw_file_basename)
                    raw_movie = FlyMovieFormat.FlyMovieSaver(full_raw,
                                                             format=image_coding,
                                                             bits_per_pixel=8,
                                                             version=3)
                    if image_coding.startswith('MONO8:'):
                        tmp_coding = 'MONO32f:' + image_coding[6:]
                    else:
                        if image_coding != 'MONO8':
                            LOG.warn('WARNING: unknown image coding %s for .fmf files' % image_coding)
                        tmp_coding = 'MONO32f'
                    bg_movie = FlyMovieFormat.FlyMovieSaver(full_bg,
                                                            format=tmp_coding,
                                                            bits_per_pixel=32,
                                                            version=3)
                    std_movie = FlyMovieFormat.FlyMovieSaver(full_std,
                                                             format='MONO32f', # std is monochrome
                                                             bits_per_pixel=32,
                                                             version=3)
                    del tmp_coding
                    state = 'saving'

                    if last_bgcmp_image_timestamp is not None:
                        bg_movie.add_frame(FastImage.asfastimage(last_running_mean_image),
                                           last_bgcmp_image_timestamp,
                                           error_if_not_fast=True)
                        std_movie.add_frame(FastImage.asfastimage(last_running_sumsqf_image),
                                            last_bgcmp_image_timestamp,
                                            error_if_not_fast=True)
                    else:
                        LOG.warn('WARNING: could not save initial bg and std frames')

                elif cmd[0] == 'stop':
                    LOG.info('done saving movies')
                    if state=='saving':
                        raw_movie.close()
                        bg_movie.close()
                        std_movie.close()
                        state = 'pass'
                    else:
                        LOG.warn("Hmm, you want me to stop saving movies, but I'm not")

            # 2: block for image data
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf: # must do on every frame
                if chainbuf.quit_now:
                    break

                if image_coding is None:
                    image_coding = chainbuf.image_coding

                if chainbuf.updated_running_mean_image is not None:
                    # Always keep the current bg and std images so
                    # that we can save them when starting a new .fmf
                    # movie save sequence.
                    last_bgcmp_image_timestamp = chainbuf.cam_received_time
                    # Keeping references to these images should be OK,
                    # not need to copy - the Process thread already
                    # made a copy of the realtime analyzer's internal
                    # copy.
                    last_running_mean_image = chainbuf.updated_running_mean_image
                    last_running_sumsqf_image = chainbuf.updated_running_sumsqf_image

                if state == 'saving':
                    raw.append( (numpy.array(chainbuf.get_buf(), copy=True),
                                 chainbuf.cam_received_time) )
                    if chainbuf.updated_running_mean_image is not None:
                        meancmp.append( (chainbuf.updated_running_mean_image,
                                         chainbuf.updated_running_sumsqf_image,
                                         chainbuf.cam_received_time)) # these were copied in process thread

            # 3: grab any more that are here
            try:
                with camnode_utils.use_buffer_from_chain(self._chain,blocking=False) as chainbuf:
                    if chainbuf.quit_now:
                        break

                    if chainbuf.updated_running_mean_image is not None:
                        # Always keep the current bg and std images so
                        # that we can save them when starting a new .fmf
                        # movie save sequence.
                        last_bgcmp_image_timestamp = chainbuf.cam_received_time
                        # Keeping references to these images should be OK,
                        # not need to copy - the Process thread already
                        # made a copy of the realtime analyzer's internal
                        # copy.
                        last_running_mean_image = chainbuf.updated_running_mean_image
                        last_running_sumsqf_image = chainbuf.updated_running_sumsqf_image

                    if state == 'saving':
                        raw.append( (numpy.array(chainbuf.get_buf(), copy=True),
                                     chainbuf.cam_received_time) )
                        if chainbuf.updated_running_mean_image is not None:
                            meancmp.append( (chainbuf.updated_running_mean_image,
                                             chainbuf.updated_running_sumsqf_image,
                                             chainbuf.cam_received_time)) # these were copied in process thread
            except Queue.Empty:
                pass

            # 4: actually save the data
            #   TODO: switch to add_frames() method which doesn't acquire GIL after each frame.
            if state == 'saving':
                for frame,timestamp in raw:
                    raw_movie.add_frame(FastImage.asfastimage(frame),timestamp,error_if_not_fast=True)
                for running_mean,running_sumsqf,timestamp in meancmp:
                    bg_movie.add_frame(FastImage.asfastimage(running_mean),timestamp,error_if_not_fast=True)
                    std_movie.add_frame(FastImage.asfastimage(running_sumsqf),timestamp,error_if_not_fast=True)
            del raw[:]
            del meancmp[:]

class SaveSmallData(object):
    def __init__(self,cam_id=None,
                 options = None,
                 mkdir_lock = None,
                 ):
        self.options = options
        self._chain = camnode_utils.ChainLink()
        self._cam_id = cam_id
        self.cmd = Queue.Queue()
        self._ufmf = None
        if mkdir_lock is not None:
            self._mkdir_lock = mkdir_lock
        else:
            self._mkdir_lock = threading.Lock()

    def get_chain(self):
        return self._chain
    def start_recording(self,
                        small_filebasename=None,
                        ):
        """threadsafe"""
        fname = small_filebasename
        self.cmd.put( ('save',fname))

    def stop_recording(self,*args,**kw):
        """threadsafe"""
        self.cmd.put( ('stop',) )

    def mainloop(self):
        # Note: need to accummulate frames into queue and add with .add_frames() for speed
        # Also: old version uses fmf version 1. Not sure why.

        meancmp = []

        state = 'pass'

        last_bgcmp_image_timestamp = None
        last_running_mean_image = None
        last_running_sumsqf_image = None

        while 1:

            while 1:
                if self.cmd.empty():
                    break
                cmd = self.cmd.get()
                if cmd[0] == 'save':
                    filename_base = cmd[1]
                    raw_file_basename = os.path.expanduser(filename_base)
                    state = 'saving'
                elif cmd[0] == 'stop':
                    if self._ufmf is not None:
                        self._ufmf.close()
                        self._ufmf = None
                    state = 'pass'

            # block for images
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf:
                if chainbuf.quit_now:
                    break

                if chainbuf.updated_running_mean_image is not None:
                    # Always keep the current bg and std images so
                    # that we can save them when starting a new .fmf
                    # movie save sequence.
                    last_bgcmp_image_timestamp = chainbuf.cam_received_time
                    # Keeping references to these images should be OK,
                    # not need to copy - the Process thread already
                    # made a copy of the realtime analyzer's internal
                    # copy.
                    last_running_mean_image = chainbuf.updated_running_mean_image
                    last_running_sumsqf_image = chainbuf.updated_running_sumsqf_image

                if state == 'saving':
                    if chainbuf.updated_running_mean_image is not None:
                        meancmp.append( (chainbuf.updated_running_mean_image,
                                         chainbuf.updated_running_sumsqf_image,
                                         chainbuf.cam_received_time)) # these were copied in process thread
                    if self._ufmf is None:
                        filename_base = os.path.abspath(os.path.expanduser(filename_base))
                        dirname = os.path.split(filename_base)[0]

                        with self._mkdir_lock:
                            # Because this is a multi-threaded
                            # program, sometimes another thread will
                            # try to create this directory.
                            if not os.path.exists(dirname):
                                os.makedirs(dirname)
                        filename = filename_base + '.ufmf'
                        LOG.info('saving to %s' % filename)
                        if chainbuf.image_coding.startswith('MONO8'):
                            h,w=numpy.array(chainbuf.get_buf(), copy=False).shape
                        else:
                            raise NotImplementedError(
                                'unable to determine shape from image with '
                                'coding %s'%(chainbuf.image_coding,))
                        self._ufmf = ufmf.AutoShrinkUfmfSaverV3( filename,
                                                                 coding = chainbuf.image_coding,
                                                                 max_width=w,
                                                                 max_height=h,
                                                                 )
                        del h,w


                        if last_running_mean_image is not None:
                            self._ufmf.add_keyframe('mean',
                                                    last_running_mean_image,
                                                    last_bgcmp_image_timestamp)
                            self._ufmf.add_keyframe('sumsq',
                                                    last_running_sumsqf_image,
                                                    last_bgcmp_image_timestamp)

                    self._tobuf( chainbuf )

            # grab any more that are here
            try:
                with camnode_utils.use_buffer_from_chain(self._chain,blocking=False) as chainbuf:
                    if chainbuf.quit_now:
                        break

                    if chainbuf.updated_running_mean_image is not None:
                        # Always keep the current bg and std images so
                        # that we can save them when starting a new .fmf
                        # movie save sequence.
                        last_bgcmp_image_timestamp = chainbuf.cam_received_time
                        # Keeping references to these images should be OK,
                        # not need to copy - the Process thread already
                        # made a copy of the realtime analyzer's internal
                        # copy.
                        last_running_mean_image = chainbuf.updated_running_mean_image
                        last_running_sumsqf_image = chainbuf.updated_running_sumsqf_image

                    if state == 'saving':
                        self._tobuf( chainbuf ) # actually save the .ufmf data
                        if chainbuf.updated_running_mean_image is not None:
                            meancmp.append( (chainbuf.updated_running_mean_image,
                                             chainbuf.updated_running_sumsqf_image,
                                             chainbuf.cam_received_time)) # these were copied in process thread
            except Queue.Empty:
                pass

            # actually save the data
            #   TODO: switch to add_frames() method which doesn't acquire GIL after each frame.
            if state == 'saving':
                for running_mean,running_sumsqf,timestamp in meancmp:
                    self._ufmf.add_keyframe('mean',running_mean,timestamp)
                    self._ufmf.add_keyframe('sumsq',running_sumsqf,timestamp)
            del meancmp[:]

    def _tobuf( self, chainbuf ):
        frame = chainbuf.get_buf()
        if 0:
            LOG.info('saving %d points' % len(chainbuf.processed_points))
        pts = []
        wh = self.options.small_save_radius*2
        for pt in chainbuf.processed_points:
            pts.append( (pt[0],pt[1],wh,wh) )
        self._ufmf.add_frame( frame, chainbuf.cam_received_time, pts )

class ImageSource(threading.Thread):
    """One instance of this class for each camera. Do nothing but get
    new frames, copy them, and pass to listener chain."""

    def __init__(self,
                 chain=None,
                 cam=None,
                 buffer_pool=None,
                 debug_acquire = False,
                 cam_id = '<unassigned>',
                 quit_event = None,
                 ):

        threading.Thread.__init__(self,name='ImageSource')
        self._chain = chain
        self.cam = cam
        with self.cam.lock:
            self.image_coding = self.cam.get_pixel_coding()
        self.buffer_pool = buffer_pool
        self.debug_acquire = debug_acquire
        self.quit_event = quit_event
        self.cam_id = cam_id

    def set_chain(self,new_chain):
        # XXX TODO FIXME: put self._chain behind lock
        if self._chain is not None:
            raise NotImplementedError('replacing a processing chain not implemented')
        self._chain = new_chain
    def get_buffer_pool(self):
        return self.buffer_pool
    def run(self):
        LOG.info('ImageSource running in process %s' % os.getpid())
        buffer_pool = self.buffer_pool
        process_quit_event_isSet = self.quit_event.isSet
        while not process_quit_event_isSet():
            self._block_until_ready() # no-op for realtime camera processing
            if buffer_pool.get_num_outstanding_buffers() > 100:
                # Grab some frames (wait) until the number of
                # outstanding buffers decreases -- give processing
                # threads time to catch up.
                LOG.warn('ERROR: We seem to be leaking buffers - will not acquire more images for a while!')
                while 1:
                    self._grab_buffer_quick()
                    if buffer_pool.get_num_outstanding_buffers() < 10:
                        LOG.info('Resuming normal image acquisition')
                        break

            # this gets a new (unused) buffer from the preallocated pool
            with get_free_buffer_from_pool( buffer_pool ) as chainbuf:
                chainbuf.quit_now = False

                _bufim = chainbuf.get_buf()

                try_again_condition, timestamp, framenumber = self._grab_into_buffer( _bufim )
                if try_again_condition:
                    continue

                if self.debug_acquire:
                    stdout_write(self.cam_id)

                cam_received_time = time.time()

                chainbuf.cam_received_time = cam_received_time
                chainbuf.timestamp = timestamp
                chainbuf.framenumber = framenumber
                chainbuf.image_coding = self.image_coding

                # Now we get rid of the frame from this thread by passing
                # it to processing threads. The last one of these will
                # return the buffer to buffer_pool when done.
                if self._chain is not None:

                    # Setting this gives responsibility to the last
                    # chain to call
                    # "buffer_pool.return_buffer(chainbuf)" when
                    # done. This is acheived automatically by the
                    # context manager in use_buffer_from_chain() and
                    # the ChainLink.end_buf() method which returns the
                    # buffer when the last link in the chain is done.
                    chainbuf._i_promise_to_return_buffer_to_the_pool = True

                    self._chain.fire( chainbuf ) # the end of the chain will call return_buffer()
        # now, we are quitting, so fire one last event through the chain to signal quit
        with get_free_buffer_from_pool( buffer_pool ) as chainbuf:
            chainbuf.quit_now = True

            # see above for this stuff
            if self._chain is not None:
                chainbuf._i_promise_to_return_buffer_to_the_pool = True
                self._chain.fire( chainbuf )

class ImageSourceBaseController(object):
    pass

class ImageSourceFromCamera(ImageSource):
    def __init__(self,*args,**kwargs):
        ImageSource.__init__(self,*args,**kwargs)

    def _block_until_ready(self):
        # no-op for realtime camera processing
        pass

    def spawn_controller(self):
        controller = ImageSourceBaseController()
        return controller

    def _grab_buffer_quick(self):
        try:
            with self.cam.lock:
                trash = self.cam.grab_next_frame_blocking()
        except cam_iface.BuffersOverflowed:
            LOG.warn('ERROR: buffers overflowed on %s'%(self.cam_id,))
        except cam_iface.FrameDataMissing:
            pass
        except cam_iface.FrameDataCorrupt:
            pass
        except cam_iface.FrameSystemCallInterruption:
            pass

    def _grab_into_buffer(self, _bufim ):
        try_again_condition= False

        with self.cam.lock:
            # transfer thread ownership into this thread. (This is a
            # semi-evil hack into camera class... Should call a method
            # like self.cam.acquire_thread())
            # self.cam.mythread=threading.currentThread()

            try:
                self.cam.grab_next_frame_into_buf_blocking(_bufim)
            except cam_iface.BuffersOverflowed:
                if self.debug_acquire:
                    stdout_write('(O%s)'%self.cam_id)
                now = time.time()
                msg = 'ERROR: buffers overflowed on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                self.log_message_queue.put((self.cam_id,now,msg))
                LOG.warn(msg)
                try_again_condition = True
            except cam_iface.FrameDataMissing:
                if self.debug_acquire:
                    stdout_write('(M%s)'%self.cam_id)
                now = time.time()
                msg = 'Warning: frame data missing on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                #self.log_message_queue.put((self.cam_id,now,msg))
                LOG.warn(msg)
                try_again_condition = True
            except cam_iface.FrameDataCorrupt:
                if self.debug_acquire:
                    stdout_write('(C%s)'%self.cam_id)
                now = time.time()
                msg = 'Warning: frame data corrupt on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                #self.log_message_queue.put((self.cam_id,now,msg))
                LOG.warn(msg)
                try_again_condition = True
            except (cam_iface.FrameSystemCallInterruption, cam_iface.NoFrameReturned):
                if self.debug_acquire:
                    stdout_write('(S%s)'%self.cam_id)
                try_again_condition = True

            if not try_again_condition:
                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()
                framenumber=self.cam.get_last_framenumber()
            else:
                timestamp = framenumber = None
        return try_again_condition, timestamp, framenumber

class ImageSourceFakeCamera(ImageSource):

    # XXX TODO: I should actually just incorporate all the fake cam
    # stuff in this class. There doesn't seem to be much point in
    # having a separate fake cam class. On the other hand, the fake
    # cam gets called by another thread, so the separation would be
    # less clear about what is running in which thread.

    def __init__(self,*args,**kw):
        self._do_step = threading.Event()
        self._fake_cam = kw['cam']
        self._buffer_pool = None
        self._count = 0
        super( ImageSourceFakeCamera, self).__init__(*args,**kw)

    def _block_until_ready(self):
        while 1:
            if self.quit_event.isSet():
                return

            if self._count==0:
                self._tstart = time.time()
            elif self._count>=1000:
                tstop = time.time()
                dur = tstop-self._tstart
                fps = self._count/dur
                LOG.debug('fps: %.1f' % fps)

                # prepare for next
                self._tstart = tstop
                self._count = 0
            self._count += 1

            # This lock ping-pongs execution back and forth between
            # "acquire" and process.

            self._do_step.wait(0.01) # timeout
            if self._do_step.isSet():
                self._do_step.clear()
                return
            if self._buffer_pool is not None:
                r=self._buffer_pool.get_num_outstanding_buffers()
                self._do_step.set()

    def register_buffer_pool( self, buffer_pool ):
        assert self._buffer_pool is None,'buffer pool may only be set once'
        self._buffer_pool = buffer_pool

    def spawn_controller(self):
        class ImageSourceFakeCameraController(ImageSourceBaseController):
            def __init__(self, do_step=None, fake_cam=None, quit_event=None):
                self._do_step = do_step
                self._fake_cam = fake_cam
                self._quit_event = quit_event
            def trigger_single_frame_start(self):
                self._do_step.set()
            def set_to_frame_0(self):
                self._fake_cam.set_to_frame_0()
            def is_finished(self):
                return self._fake_cam.is_finished()
            def quit_now(self):
                self._quit_event.set()
            def get_n_frames(self):
                return self._fake_cam.get_n_frames()
        controller = ImageSourceFakeCameraController(self._do_step,
                                                     self._fake_cam,
                                                     self.quit_event)
        return controller

    def _grab_buffer_quick(self):
        time.sleep(0.05)

    def _grab_into_buffer(self, _bufim ):
        with self.cam.lock:
            self.cam.grab_next_frame_into_buf_blocking(_bufim, self.quit_event)

            try_again_condition = False
            timestamp=self.cam.get_last_timestamp()
            framenumber=self.cam.get_last_framenumber()
        return try_again_condition, timestamp, framenumber

class _Camera(object):

    ROS_PROPERTIES = {}

    def __init__(self, guid):
        self.guid = guid
        self.lock = threading.Lock()

    def start_camera(self):
        pass

    def get_framerate(self):
        return 100

    def get_num_camera_properties(self):
        return 0

    def get_trigger_mode_number(self):
        return 0

    def get_max_height(self):
        l,b,w,h = self.get_frame_roi()
        return h

    def get_max_width(self):
        l,b,w,h = self.get_frame_roi()
        return w

    def close(self):
        return

    def set_trigger_mode_number(self, v):
        pass

    def get_trigger_mode_number(self):
        return 0

    def get_num_trigger_modes(self):
        return 1

    def get_trigger_mode_string(self,i):
        return 'fake camera trigger'

    def get_frame_roi(self):
        raise NotImplementedError

    def set_camera_property(self, prop_num, prop_value, auto):
        pass

    def load_configuration(self):
        pass

class CamifaceCamera(cam_iface.Camera, _Camera):

    ROS_PROPERTIES = dict(
        shutter=9000,
        gain=300,
        trigger='',
        trigger_mode=-1,
    )

    def __init__(self, guid, cam_no, show_cam_details, num_buffers=50):
        self._show_cam_details = show_cam_details

        # auto select format7_0 mode
        N_modes = cam_iface.get_num_modes(cam_no)
        use_mode = None
        if self._show_cam_details:
            LOG.info('camera info: %s' % (cam_iface.get_camera_info(cam_no),))
        for i in range(N_modes):
            mode_string = cam_iface.get_mode_string(cam_no,i)
            if self._show_cam_details:
                LOG.info('  mode %d: %s'%(i,mode_string))
            if 'format7_0' in mode_string.lower():
                use_mode = i
        if use_mode is None:
            use_mode = 0

        cam_iface.Camera.__init__(self,cam_no,num_buffers,use_mode)

        # cache trigger mode names
        self._trigger_mode_numbers_from_name = {}
        N_trigger_modes = self.get_num_trigger_modes()
        if self._show_cam_details:
            LOG.info('  %d available trigger modes:'%N_trigger_modes)
            LOG.info('  current trigger mode: %d' % self.get_trigger_mode_number())
        for i in range(N_trigger_modes):
            mode_string = self.get_trigger_mode_string(i)
            self._trigger_mode_numbers_from_name[mode_string] = i
            if self._show_cam_details:
                LOG.info('  mode %d: %s'%(i,mode_string))

        #cache the properties
        num_props = self.get_num_camera_properties()
        self._prop_numbers_from_name = {}
        self._prop_names_from_numbers = {}
        for i in range(num_props):
            info = self.get_camera_property_info(i)
            name = info['name']
            self._prop_numbers_from_name[name] = i
            self._prop_names_from_numbers[i] = name

        _Camera.__init__(self, guid)

    def _get_rosparam_path(self, paramname):
        return "%s/%s" % (self.guid, paramname)

    def load_configuration(self):

        def _get_param_with_fallback(_k, _v):
            _parampath = self._get_rosparam_path(_k)
            _paramval = rospy.get_param(
                        "flydra/%s" % _k,                #try in the flydra namespace first
                        rospy.get_param(_parampath, _v)   #try in the private namespace (with default)
            )
            return _parampath, _paramval

        for k,v in self.ROS_PROPERTIES.iteritems():
            parampath, paramval = _get_param_with_fallback(k,v)
            if k in self._prop_numbers_from_name:
                prop_num = self._prop_numbers_from_name[k]
                #call directly to camiface, these parameters are already from the parameter server
                cam_iface.Camera.set_camera_property(self,prop_num,paramval,0)

        #prefer trigger (which is a string) to trigger_mode (which is a number)
        _, trigger = _get_param_with_fallback("trigger", self.ROS_PROPERTIES["trigger"])
        _, trigger_mode = _get_param_with_fallback("trigger_mode", self.ROS_PROPERTIES["trigger_mode"])
        trigger_mode_number = self._trigger_mode_numbers_from_name.get(trigger, trigger_mode)

        if trigger_mode_number < 0:
            LOG.info("trigger_mode number not set or correct (%s), setting camera to max framerate" % trigger_mode_number)
            cam_iface.Camera.set_framerate(self, 999)
        else:
            LOG.info("setting trigger_mode number = %s" % trigger_mode_number)
            cam_iface.Camera.set_trigger_mode_number(self, trigger_mode_number)

    def set_camera_property(self, prop_num, prop_value, auto):
        if prop_num in self._prop_names_from_numbers:
            paramname = self._prop_names_from_numbers[prop_num]
            if paramname in self.ROS_PROPERTIES:
                parampath = self._get_rosparam_path(paramname)
                rospy.set_param(parampath, prop_value)
        cam_iface.Camera.set_camera_property(self,prop_num,prop_value,auto)

    def set_trigger_mode_number(self, v):
        parampath = self._get_rosparam_path("trigger_mode")
        rospy.set_param(parampath, v)

    def set_framerate(self, *args):
        raise NotImplementedError

class FakeCameraFromRNG(_Camera):
    def __init__(self,guid,frame_size):
        print guid*10
        _Camera.__init__(self, guid)
        self.frame_size = frame_size
        self.last_timestamp = 0.0
        self.last_count = -1

    def get_pixel_coding(self):
        return 'MONO8'

    def get_frame_roi(self):
        w,h=self.frame_size
        return 0,0,w,h

    def grab_next_frame_into_buf_blocking(self,buf, quit_event):
        # XXX TODO: implement quit_event checking
        w,h = self.frame_size
        new_raw = np.asarray( buf )
        assert new_raw.shape == (h,w)
        self.last_timestamp = time.time()
        self.last_count += 1
        for pt_num in range( np.random.randint(5) ):
            x,y = np.random.uniform(0.0,1.0,size=(2,))
            xi = int(round(x*(w-1)))
            yi = int(round(y*(h-1)))
            new_raw[yi,xi] = 10
        return new_raw

    def get_last_timestamp(self):
        return self.last_timestamp

    def get_last_framenumber(self):
        return self.last_count

class FakeCameraFromFMF(_Camera):
    def __init__(self,filename):
        _Camera.__init__(self, filename)
        self.fmf_recarray = FlyMovieFormat.mmap_flymovie( filename )

        self._n_frames = len(self.fmf_recarray)
        self._curframe = SharedValue1(0)
        self._frame_offset = 0

    def get_n_frames(self):
        return self._n_frames

    def get_frame_roi(self):
        h,w = self.fmf_recarray['frame'][0].shape
        return 0,0,w,h

    def grab_next_frame_into_buf_blocking(self, buf, quit_event):
        buf = numpy.asarray( buf )
        curframe = self._curframe.get()
        while self.is_finished():
            if quit_event.isSet():
                return
            # We're being asked to go off the end here...
            # wait until we get told to return to beginning.
            time.sleep(0.05)
            curframe = self._curframe.get()
        buf[:,:] = self.fmf_recarray['frame'][ curframe ]
        self._last_timestamp = self.fmf_recarray['timestamp'][ curframe ]
        self._last_framenumber = curframe + self._frame_offset
        self._curframe.set( curframe + 1 )

    def get_last_timestamp(self):
        return self._last_timestamp

    def get_last_framenumber(self):
        return self._last_framenumber

    def set_to_frame_0(self):
        self._frame_offset += self._curframe.get()
        self._curframe.set( 0 )

    def is_finished(self):
        # this can is called by any thread
        result = self._curframe.get() >= len( self.fmf_recarray['frame'] )
        return result

def create_cam_for_emulation_image_source( filename_or_pseudofilename, cam_no ):
    """factory function to create fake camera and ImageSourceModel"""
    fname = filename_or_pseudofilename
    if fname.endswith('.fmf'):
        cam = FakeCameraFromFMF(fname)
        ImageSourceModel = ImageSourceFakeCamera

        mean_filename = os.path.splitext(fname)[0] + '_mean' + '.fmf'
        sumsqf_filename = os.path.splitext(fname)[0] + '_sumsqf' + '.fmf'

        fmf_ra = FlyMovieFormat.mmap_flymovie( fname )
        mean_ra =  FlyMovieFormat.mmap_flymovie( mean_filename )
        sumsqf_ra = FlyMovieFormat.mmap_flymovie( sumsqf_filename ) # not really mean2 (actually running_sumsqf)

        t0 = fmf_ra['timestamp'][0]
        mean_t0 = mean_ra['timestamp'][0]
        sumsqf_t0 = sumsqf_ra['timestamp'][0]

        if not ((t0 >= mean_t0) and (t0 >= sumsqf_t0)):
            LOG.warn('WARNING timestamps of first image frame is not before mean image timestamps. they are'
                     ' raw .fmf: %s\n'%
                     ' mean .fmf:  %s\n'%
                     ' sumsqf .fmf: %s'%(repr(t0),repr(mean_t0),repr(sumsqf_t0)))

        initial_image_dict = {'mean':mean_ra['frame'][0],
                              'sumsqf':sumsqf_ra['frame'][0],  # not really mean2 (actually running_sumsqf)
                              'raw':fmf_ra['frame'][0]}
        if 0 and len( mean_ra['frame'] ) > 1:
            LOG.info("no current support for reading back multi-frame "
                     "background/cmp. (But this should not be necessary, "
                     "as you can reconstruct them, anyway.)")

    elif fname.endswith('.ufmf'):
        raise NotImplementedError('patience, young grasshopper')
    elif fname == '<rng>':
        width, height = 640, 480
        cam = FakeCameraFromRNG('fakecam%s' % cam_no,(width,height))
        ImageSourceModel = ImageSourceFakeCamera
        with cam.lock:
            l,b,w,h = cam.get_frame_roi()

        mean = np.ones( (h,w), dtype=np.uint8 )
        sumsqf = np.ones( (h,w), dtype=np.uint8 )
        raw = np.ones( (h,w), dtype=np.uint8 )

        initial_image_dict = {'mean':mean,
                              'sumsqf':sumsqf,
                              'raw':raw}
    else:
        raise ValueError('could not create emulation image source')
    return cam, ImageSourceModel, initial_image_dict

class ConsoleApp(object):
    def __init__(self, call_often=None):
        self.call_often = call_often
        self.exit_value = 0
        self.quit_now = False
    def MainLoop(self):
        while not self.quit_now:
            time.sleep(0.05)
            self.call_often()
        if self.exit_value != 0:
            sys.exit(self.exit_value)
    def OnQuit(self, exit_value=0):
        self.quit_now = True
        self.exit_value = exit_value

    def generate_view(self, model, controller ):
        if hasattr(controller, 'trigger_single_frame_start' ):
            warnings.warn('no control in ConsoleApp for %s'%controller)
            controller.trigger_single_frame_start()

class AppState(object):
    """This class handles all camera states, properties, etc."""
    def __init__(self,
                 benchmark = False,
                 options = None,
                 ):
        self.options = options
        self._real_quit_function = None

        #Performance of the coordinate processor thread is critical. The time
        #taken to perform DNS lookups is non-trivial (i.e. 2ms x ncameras x 100Hz)
        #so can cause the camnode to fall behind mainbrain and never catch up. Ensure
        #that communication is done using ip addresses, and check that these are valid
        try:
            self.main_brain_ipaddr = socket.gethostbyname(options.main_brain)
        except socket.gaierror:
            raise RuntimeError('Mainbrain host %s not found' % options.main_brain)
        try:
            socket.inet_aton(self.main_brain_ipaddr)
        except socket.error:
            raise RuntimeError('Mainbrain ip address %s not valid' % self.main_brain_ipaddr)

        self.log_message_queue = Queue.Queue()

        force_cam_ids = options.force_cam_ids
        if force_cam_ids is not None:
            force_cam_ids = force_cam_ids.split(',')

        emulation_image_sources = options.emulation_image_sources
        if emulation_image_sources is not None:
            emulation_image_sources = emulation_image_sources.split( os.pathsep )
            num_cams = len( emulation_image_sources )
        elif options.simulate_point_extraction is not None:
            image_sources = options.simulate_point_extraction.split( os.pathsep )
            num_cams = len( image_sources )
        elif benchmark:
            num_cams = 1
        else:
            ##################################################################
            #
            # Setup cameras
            #
            ##################################################################

            guid = None
            all_cam_info_list = []
            for i in range(cam_iface.get_num_cameras()):
                try:
                    this_info1 =  cam_iface.get_camera_info(i)
                    mfg,model,guid = this_info1
                    if options.cams_only and guid not in set(options.cams_only.split(',')):
                        LOG.info('skipping camera guid: %s (cam_id: %d)' % (guid,i))
                        continue
                except cam_iface.CameraNotAvailable:
                    this_info2 =  ('(not available)',i)
                else:
                    LOG.info('choosing camera guid: %s (cam_id: %d)' % (guid,i))
                    this_info2 =  ('\0'.join(this_info1),i)
                all_cam_info_list.append(this_info2)

            all_cam_info_list.sort() # make sure list is always in same order for given cameras
            all_cam_info_list.reverse() # any ordering will do, but reverse for historical reasons
            cam_order = [ x[1] for x in all_cam_info_list]
            del all_cam_info_list, guid
            num_cams = len(cam_order)

        if num_cams == 0:
            raise RuntimeError('No cameras detected')

        self.all_cams = [None]*num_cams
        self.cam_status = [None]*num_cams
        self.all_cam_chains = [None]*num_cams
        self.all_cam_processors = [None]*num_cams
        self.all_savers = [None]*num_cams
        self.all_small_savers = [None]*num_cams
        self.globals = [None]*num_cams
        self.all_cam_ids = [None]*num_cams

        self._image_sources = [None]*num_cams
        self._image_controllers = [None]*num_cams
        initial_images = [None]*num_cams
        self.critical_threads = []

        for cam_no in range(num_cams):
            ##################################################################
            #
            # Initialize "global" variables
            #
            ##################################################################

            self.globals[cam_no] = {} # intialize
            globals = self.globals[cam_no] # shorthand

            globals['debug_acquire']=options.debug_acquire
            globals['incoming_raw_frames']=Queue.Queue()
            globals['raw_fmf_and_bg_fmf']=None
            globals['most_recent_frame_potentially_corrupt']=None
            globals['saved_bg_frame']=False

            # control flow events for threading model
            globals['process_quit_event'] = threading.Event()
            globals['listen_thread_done'] = threading.Event()
            globals['take_background'] = threading.Event()
            globals['clear_background'] = threading.Event()
            globals['collecting_background'] = threading.Event()
            globals['collecting_background'].set()
            globals['export_image_name'] = 'raw'
            globals['use_cmp'] = threading.Event()

            #print 'not using ongoing variance estimate'
            globals['use_cmp'].set()

            if benchmark: # emulate full images with random number generator
                # call factory function
                (cam, ImageSourceModel, initial_image_dict) = \
                      create_cam_for_emulation_image_source( '<rng>', cam_no )
                self.all_cam_ids[cam_no] = cam.guid
            elif options.simulate_point_extraction: # emulate points
                # call factory function
                (cam, ImageSourceModel,
                 initial_image_dict)  = create_cam_for_emulation_image_source( image_sources[cam_no], cam_no )
                self.all_cam_ids[cam_no] = cam.guid
            elif emulation_image_sources: # emulate full images
                # call factory function
                (cam, ImageSourceModel,
                 initial_image_dict)  = create_cam_for_emulation_image_source( emulation_image_sources[cam_no], cam_no )
                self.all_cam_ids[cam_no] = cam.guid
            else:
                #cam_id is the libcamiface number of the camera
                cam_id = cam_order[cam_no]
                try:
                    mfg,model,guid = cam_iface.get_camera_info(cam_id)
                    if not guid:
                        raise RuntimeError('libcamiface camera %d has invalid guid' % (cam_id, guid))
                    self.all_cam_ids[cam_no] = ros_ensure_valid_name(guid)
                except cam_iface.CameraNotAvailable:
                    raise RuntimeError('camera %d not available' % cam_no)

                LOG.info('constructing camera guid: %s (cam_id: %d cam_no: %d)' % (guid,cam_id,cam_no))
                del guid
                cam = CamifaceCamera(self.all_cam_ids[cam_no],cam_id,options.show_cam_details)
                ImageSourceModel = ImageSourceFromCamera
                initial_image_dict = None


            #load backend specific configuration (i.e. from ROS at a backend specific prefix)
            cam.load_configuration()

            if initial_image_dict is None:
                globals['take_background'].set()
            else:
                globals['take_background'].clear()

            initial_images[cam_no] = initial_image_dict

            self.all_cams[cam_no] = cam
            if cam is not None:
                with cam.lock:
                    try:
                        cam.start_camera()  # start camera
                    except:
                        print 'FAILED to open camera %s'%cam.guid
                        raise
            self.cam_status[cam_no]= 'started'
            if ImageSourceModel is not None:
                with cam.lock:
                    l,b,w,h = cam.get_frame_roi()
                buffer_pool = PreallocatedBufferPool(FastImage.Size(w,h))
                del l,b,w,h
                image_source = ImageSourceModel(chain = None,
                                                cam = cam,
                                                buffer_pool = buffer_pool,
                                                debug_acquire = options.debug_acquire,
                                                cam_id = self.all_cam_ids[cam_no],
                                                quit_event = globals['process_quit_event'],
                                                )
                if benchmark: # should maybe be for any simulated camera in non-GUI mode?
                    image_source.register_buffer_pool( buffer_pool )

                controller = image_source.spawn_controller()

                image_source.setDaemon(True)
                self._image_sources[cam_no] = image_source
                self._image_controllers[cam_no]= controller
            else:
                self._image_sources[cam_no] = None
                self._image_controllers[cam_no]= None

        ##################################################################
        #
        # Initialize network connections
        #
        ##################################################################

        if benchmark:
            self.main_brain = DummyMainBrain()
        else:
            self.main_brain = ROSMainBrain()
            main_brain_version = self.main_brain.get_version()

            if not options.ignore_version:
                assert main_brain_version == flydra.version.__version__, "version mismatch: %s vs %s"%(main_brain_version,flydra.version.__version__)

        ##################################################################
        #
        # Initialize more stuff
        #
        ##################################################################

        if (not benchmark) or (not FLYDRA_BT):
            # run in single-thread for benchmark
            timestamp_echo_thread=threading.Thread(target=TimestampEcho,
                                                   name='TimestampEcho')
            timestamp_echo_thread.setDaemon(True) # quit that thread if it's the only one left...
            timestamp_echo_thread.start()

        mask_images = options.mask_images
        if mask_images is not None:
            mask_images = mask_images.split( os.pathsep )

        save_small_data_mkdir_lock = threading.Lock()

        for cam_no in range(num_cams):
            cam = self.all_cams[cam_no]
            with cam.lock:

                left,top,width,height = cam.get_frame_roi()
                del left,top
                globals = self.globals[cam_no] # shorthand

                if mask_images is not None:
                    mask_image_fname = mask_images[cam_no]
                    print("----------------------------------------------------------------------------------------------------")
                    print("Camera guid ='%s' \n has mask image: '%s'" % (self.all_cam_ids[cam_no], mask_image_fname))
                    im = scipy.misc.pilutil.imread( mask_image_fname )
                    if len(im.shape) != 3:
                        raise ValueError('mask image must have color channels')
                    if im.shape[2] != 4:
                        raise ValueError('mask image must have an alpha (4th) channel')
                    alpha = im[:,:,3]
                    if numpy.any((alpha > 0) & (alpha < 255)):
                        LOG.warning('WARNING: some alpha values between 0 and '
                                    '255 detected. Only zero and non-zero values are '
                                    'considered.')
                    mask = alpha.astype(numpy.bool)
                else:
                    mask = numpy.zeros( (height,width), dtype=numpy.bool )
                # mask is currently an array of bool
                mask = mask.astype(numpy.uint8)*255


                # get settings
                scalar_control_info = {}
                globals['cam_controls'] = {}
                CAM_CONTROLS = globals['cam_controls']

                num_props = cam.get_num_camera_properties()
                prop_names = []
                for prop_num in range(num_props):
                    props = cam.get_camera_property_info(prop_num)
                    current_value,auto = cam.get_camera_property( prop_num )
                    new_value = current_value
                    min_value = props['min_value']
                    max_value = props['max_value']
                    force_manual = True
                    if props['has_manual_mode']:
                        if force_manual or min_value <= new_value <= max_value:
                            try:
                                if options.show_cam_details:
                                    LOG.info('setting camera property "%s" to manual mode' % props['name'])
                                cam.set_camera_property( prop_num, new_value, 0 )
                            except:
                                LOG.warn('error while setting property %s to %d (from %d)' % (props['name'],new_value,current_value))
                                raise
                        else:
                            if options.show_cam_details:
                                LOG.info('not setting property %s to %d (from %d) because out of range (%d<=value<=%d)'%(props['name'],new_value,current_value,min_value,max_value))

                        CAM_CONTROLS[props['name']]=prop_num
                    current_value,auto = cam.get_camera_property( prop_num )
                    current_value = new_value
                    scalar_control_info[props['name']] = (current_value,
                                                          min_value, max_value)
                    prop_names.append( props['name'] )

                scalar_control_info['camprops'] = prop_names

                diff_threshold_shared = SharedValue()
                diff_threshold_shared.set(options.diff_threshold)
                scalar_control_info['diff_threshold'] = diff_threshold_shared.get_nowait()

                clear_threshold_shared = SharedValue()
                clear_threshold_shared.set(options.clear_threshold)
                scalar_control_info['clear_threshold'] = clear_threshold_shared.get_nowait()
                scalar_control_info['visible_image_view'] = 'raw'

                try:
                    scalar_control_info['trigger_mode'] = cam.get_trigger_mode_number()
                except cam_iface.CamIFaceError:
                    scalar_control_info['trigger_mode'] = 0
                scalar_control_info['cmp'] = globals['use_cmp'].isSet()

                n_sigma_shared = SharedValue()
                n_sigma_shared.set(options.n_sigma)
                scalar_control_info['n_sigma'] = n_sigma_shared.get_nowait()

                n_erode_absdiff_shared = SharedValue()
                n_erode_absdiff_shared.set(options.n_erode_absdiff)
                scalar_control_info['n_erode_absdiff'] = n_erode_absdiff_shared.get_nowait()

                color_range_1_shared = SharedValue()
                color_range_1_shared.set(options.color_range_1)
                scalar_control_info['color_range_1'] = color_range_1_shared.get_nowait()

                color_range_2_shared = SharedValue()
                color_range_2_shared.set(options.color_range_2)
                scalar_control_info['color_range_2'] = color_range_2_shared.get_nowait()

                color_range_3_shared = SharedValue()
                color_range_3_shared.set(options.color_range_3)
                scalar_control_info['color_range_3'] = color_range_3_shared.get_nowait()

                sat_thresh_shared = SharedValue()
                sat_thresh_shared.set(options.sat_thresh)
                scalar_control_info['sat_thresh'] = sat_thresh_shared.get_nowait()

                red_only_shared = SharedValue()
                red_only_shared.set(int(options.red_only))
                scalar_control_info['red_only'] = red_only_shared.get_nowait()

                scalar_control_info['width'] = width
                scalar_control_info['height'] = height
                scalar_control_info['roi'] = 0,0,width-1,height-1
                scalar_control_info['max_framerate'] = cam.get_framerate()
                scalar_control_info['collecting_background']=globals['collecting_background'].isSet()
                scalar_control_info['expected_trigger_framerate'] = 0.0

                # register self with remote server
                cam_guid = self.all_cam_ids[cam_no]
                cam2mainbrain_port = self.main_brain.register_new_camera(cam_guid,
                                                                         scalar_control_info,
                                                                         camnode_ros_name = rospy.get_name())

                ##################################################################
                #
                # Processing chains
                #
                ##################################################################

                # setup chain for this camera:
                if not DISABLE_ALL_PROCESSING:
                    if 0:
                        cam_processor = FakeProcessCamData()
                    else:

                        initial_image_dict = initial_images[cam_no]

                        cam.get_max_height()
                        l,b,w,h = cam.get_frame_roi()
                        r = l+w-1
                        t = b+h-1
                        lbrt = l,b,r,t
                        cam_processor = ProcessCamClass(
                            cam2mainbrain_port=cam2mainbrain_port,
                            cam_id=cam_guid,
                            log_message_queue=self.log_message_queue,
                            max_num_points=options.num_points,
                            roi2_radius=options.software_roi_radius,
                            bg_frame_interval=options.background_frame_interval,
                            bg_frame_alpha=options.background_frame_alpha,
                            cam_no=cam_no,
                            main_brain_ipaddr=self.main_brain_ipaddr,
                            mask_image=mask,
                            diff_threshold_shared=diff_threshold_shared,
                            clear_threshold_shared=clear_threshold_shared,
                            n_sigma_shared=n_sigma_shared,
                            n_erode_absdiff_shared=n_erode_absdiff_shared,
                            color_range_1_shared=color_range_1_shared,
                            color_range_2_shared=color_range_2_shared,
                            color_range_3_shared=color_range_3_shared,
                            sat_thresh_shared=sat_thresh_shared,
                            red_only_shared=red_only_shared,
                            framerate=None,
                            lbrt=lbrt,
                            max_height=cam.get_max_height(),
                            max_width=cam.get_max_width(),
                            globals=globals,
                            options=options,
                            initial_image_dict = initial_image_dict,
                            benchmark=benchmark,
                            )
                    self.all_cam_processors[cam_no]= cam_processor

                    cam_processor_chain = cam_processor.get_chain()
                    self.all_cam_chains[cam_no]=cam_processor_chain
                    thread = threading.Thread( target = cam_processor.mainloop,
                                               name = 'cam_processor.mainloop')
                    thread.setDaemon(True)
                    thread.start()
                    self.critical_threads.append( thread )

                    if 1:
                        save_cam = SaveCamData()
                        self.all_savers[cam_no]= save_cam
                        cam_processor_chain.append_link( save_cam.get_chain() )
                        thread = threading.Thread( target = save_cam.mainloop,
                                                   name = 'save_cam.mainloop')
                        thread.setDaemon(True)
                        thread.start()
                        self.critical_threads.append( thread )
                    else:
                        LOG.info('not starting full .fmf thread')

                    if 1:
                        save_small = SaveSmallData(options=self.options,
                                                   mkdir_lock = save_small_data_mkdir_lock)
                        self.all_small_savers[cam_no]= save_small
                        cam_processor_chain.append_link( save_small.get_chain() )
                        thread = threading.Thread( target = save_small.mainloop,
                                                   name = 'save_small.mainloop')
                        thread.setDaemon(True)
                        thread.start()
                        self.critical_threads.append( thread )
                    else:
                        LOG.info('not starting small .fmf thread')

                else:
                    cam_processor_chain = None

                self._image_sources[cam_no].set_chain( cam_processor_chain )

                ##################################################################
                #
                # Misc
                #
                ##################################################################

                if cam_iface is not None:
                    driver_string = 'using cam_iface driver: %s (wrapper: %s)'%(
                        cam_iface.get_driver_name(),
                        cam_iface.get_wrapper_name())
                    self.log_message_queue.put((cam_guid,time.time(),driver_string))

        self.last_frames_by_cam = [ [] for c in range(num_cams) ]
        self.last_points_by_cam = [ [] for c in range(num_cams) ]
        self.last_points_framenumbers_by_cam = [ [] for c in range(num_cams) ]
        self.last_return_info_check = [ 0.0 for i in range(num_cams)]

        for cam_no in range(num_cams):
            cam = self.all_cams[cam_no]
            with cam.lock:
                image_source = self._image_sources[cam_no]
                if image_source is not None:
                    image_source.start()

    def get_image_sources(self):
        return self._image_sources

    def get_image_controllers(self):
        return self._image_controllers

    def quit_function(self,exit_value):
        for globals in self.globals:
            globals['process_quit_event'].set()

        for thread in self.critical_threads:
            if thread.isAlive():
                thread.join(0.01)

        if self._real_quit_function is not None:
            self._real_quit_function(exit_value)

    def set_quit_function(self, quit_function=None):
        self._real_quit_function = quit_function

    def append_chain(self,
                     klass=None,
                     args=None,
                     basename=None,
                     kwargs=None,
                     kwargs_per_instance=None,
                     ):
        if basename is None:
            basename = 'appended thread'
        targets = {}
        for cam_no, (cam_id, chain) in enumerate(zip(self.all_cam_ids,
                                                     self.all_cam_chains)):
            base_kwargs = dict(cam_id=cam_id)

            if kwargs is not None:
                base_kwargs.update( kwargs )

            if kwargs_per_instance is not None:
                base_kwargs.update( kwargs_per_instance[ cam_no ] )

            if args is None:
                thread_instance = klass(**base_kwargs)
            else:
                thread_instance = klass(*args,**base_kwargs)

            chain.append_link( thread_instance.get_chain() )
            name = basename + ' ' + cam_id
            thread = threading.Thread( target = thread_instance.mainloop,
                                       name = name )
            thread.setDaemon(True)
            thread.start()
            self.critical_threads.append( thread )
            targets[cam_id] = thread_instance
        return targets

    def main_thread_task(self):
        """gets called often in mainloop of app"""
        try:
            for cam_no, cam_id in enumerate(self.all_cam_ids):
                if self.cam_status[cam_no] == 'destroyed':
                    # ignore commands for closed cameras
                    continue
                try:
                    cmds=self.main_brain.get_and_clear_commands(cam_id)
                except KeyError:
                    LOG.warn('main brain appears to have lost cam_id %s' % cam_id)
                except rospy.ServiceException:
                    #mainbrain shutting down
                    pass
                else:
                    self.handle_commands(cam_no,cmds)

            # test if all closed
            all_closed = True
            for cam_no, cam_id in enumerate(self.all_cam_ids):
                if self.cam_status[cam_no] != 'destroyed':
                    all_closed = False
                    break

            # quit if no more cameras
            if all_closed:
                if self.quit_function is None:
                    raise RuntimeError('all cameras closed, but no quit_function set')
                self.quit_function(0)

            # if any threads have died, quit
            for thread in self.critical_threads:
                if not thread.isAlive():
                    LOG.fatal('ERROR: thread %s died unexpectedly. Quitting' % thread.getName())
                    self.quit_function(1)

            if not DISABLE_ALL_PROCESSING:
                for cam_no, cam_id in enumerate(self.all_cam_ids):
                    globals = self.globals[cam_no] # shorthand
                    last_frames = self.last_frames_by_cam[cam_no]
                    last_points = self.last_points_by_cam[cam_no]
                    last_points_framenumbers = self.last_points_framenumbers_by_cam[cam_no]

                    # Get new raw frames from grab thread.
                    get_raw_frame = globals['incoming_raw_frames'].get_nowait
                    try:
                        while 1:
                            (frame,timestamp,framenumber,points,lbrt,
                             cam_received_time) = get_raw_frame() # this may raise Queue.Empty
                            last_frames.append( (frame,timestamp,framenumber,points) ) # save for post-triggering
                            while len(last_frames)>200:
                                del last_frames[0]

                            last_points_framenumbers.append( framenumber ) # save for dropped packet recovery
                            last_points.append( (timestamp,points,cam_received_time) ) # save for dropped packet recovery
                            while len(last_points)>10000:
                                del last_points[:100]
                                del last_points_framenumbers[:100]

                    except Queue.Empty:
                        pass

        except:
            LOG.fatal(traceback.format_exc())
            self.quit_function(1)

    def handle_commands(self, cam_no, cmds_orig):
        cmds = cmds_orig.copy() # copy dict to prevent potential threading issues
        if cmds:
            cam_processor = self.all_cam_processors[cam_no]
            saver = self.all_savers[cam_no]
            small_saver = self.all_small_savers[cam_no]
            cam_id = self.all_cam_ids[cam_no]
            cam = self.all_cams[cam_no]
            #print 'handle_commands:',cam_id, cmds
            globals = self.globals[cam_no]

            CAM_CONTROLS = globals['cam_controls']

        for key in cmds.keys():
            LOG.info('  handle_commands: key %s' % key)
            if key == 'set':
                with cam.lock:

                    for property_name,value in cmds['set'].iteritems():
                        LOG.info('setting camera property %s=%s' % (property_name, value))
                        sys.stdout.flush()
                        if property_name in CAM_CONTROLS:
                            enum = CAM_CONTROLS[property_name]
                            if type(value) == tuple: # setting whole thing
                                props = cam.get_camera_property_info(enum)
                                if value[1] != props['min_value']:
                                    import warnings
                                    warnings.warn('value[1] != props["min_value"] (%d != %d)'%(value[1], props['min_value']))
                                if value[2] != props['max_value']:
                                    import warnings
                                    warnings.warn('value[2] != props["max_value"] (%d != %d)'%(value[2], props['max_value']))
                                value = value[0]
                            cam.set_camera_property(enum,value,0)
                        elif property_name == 'roi':
                            LOG.info('flydra_camera_node.py: ignoring ROI command for now...')
                        elif property_name == 'n_sigma':
                            cam_processor.n_sigma_shared.set(value)
                        elif property_name == 'n_erode_absdiff':
                            cam_processor.n_erode_absdiff_shared.set(value)
                        elif property_name == 'red_only':
                            cam_processor.red_only.set(value)
                        elif property_name == 'color_range_1':
                            cam_processor.color_range_1_shared.set(value)
                        elif property_name == 'color_range_2':
                            cam_processor.color_range_2_shared.set(value)
                        elif property_name == 'color_range_3':
                            cam_processor.color_range_3_shared.set(value)
                        elif property_name == 'sat_thresh':
                            cam_processor.sat_thresh_shared.set(value)
                        elif property_name == 'diff_threshold':
                            cam_processor.diff_threshold_shared.set(value)
                        elif property_name == 'clear_threshold':
                            cam_processor.clear_threshold_shared.set(value)
                        elif property_name == 'width':
                            assert cam.get_max_width() == value
                        elif property_name == 'height':
                            assert cam.get_max_height() == value
                        elif property_name == 'trigger_mode':
                            try:
                                cam.set_trigger_mode_number( value ) # XXX don't crash on exception here
                            except Exception,err:
                                LOG.warn('ERROR setting trigger mode\n%s' % traceback.format_exc())
                        elif property_name == 'cmp':
                            if value:
                                globals['use_cmp'].set()
                            else: globals['use_cmp'].clear()
                        elif property_name == 'expected_trigger_framerate':
                            if value==0.0:
                                LOG.warn('WARNING: expected_trigger_framerate is set '
                                         'to 0, but setting shortest IFI to 10 msec '
                                         'anyway')
                                cam_processor.shortest_IFI = 0.01 # XXX TODO: FIXME: thread crossing bug
                            else:
                                cam_processor.shortest_IFI = 1.0/value # XXX TODO: FIXME: thread crossing bug
                        elif property_name == 'max_framerate':
                            if 0:
                                pass
                            else:
                                try:
                                    cam.set_framerate(value)
                                except Exception,err:
                                    LOG.warn('ERROR: failed setting framerate: %s' % err)
                        elif property_name == 'collecting_background':
                            if value: globals['collecting_background'].set()
                            else: globals['collecting_background'].clear()
                        elif property_name == 'color_filter':
                            cam_processor.red_only_shared.set(value)
                        elif property_name == 'visible_image_view':
                            globals['export_image_name'] = value
                        else:
                            LOG.warn('IGNORING property %s' % property_name)

            elif key == 'get_im':
                val = globals['most_recent_frame_potentially_corrupt']
                if val is not None: # prevent race condition
                    lb, im = val
                    #nxim = nx.array(im) # copy to native nx form, not view of __array_struct__ form
                    nxim = nx.asarray(im) # view of __array_struct__ form
                    self.main_brain.set_image(cam_id, lb, nxim)

            elif key == 'request_missing':
                camn_and_list = map(int,cmds[key].split())
                camn, framenumber_offset = camn_and_list[:2]
                missing_framenumbers = camn_and_list[2:]
                msg = 'I know main brain wants %d frames (camn %d) at %s:'%(
                        len(missing_framenumbers),
                        camn,time.asctime())
                if len(missing_framenumbers) > 200:
                    LOG.info("%s %s + ... + %s" % (msg,missing_framenumbers[:25],missing_framenumbers[-25:]))
                else:
                    LOG.info("%s %s" % (msg,missing_framenumbers))

                last_points_framenumbers = self.last_points_framenumbers_by_cam[cam_no]
                last_points = self.last_points_by_cam[cam_no]

                # convert to numpy arrays for quick indexing
                last_points_framenumbers = numpy.array( last_points_framenumbers, dtype=numpy.int64 )
                missing_framenumbers = numpy.array( missing_framenumbers, dtype=numpy.int64 )

                # now find missing_framenumbers in last_points_framenumbers
                idxs = last_points_framenumbers.searchsorted( missing_framenumbers )

                missing_data = []
                still_missing = []
                for ii,(idx,missing_framenumber) in enumerate(zip(idxs,missing_framenumbers)):
                    if idx == 0:
                        # search sorted will sometimes return 0 when value not in range
                        found_framenumber = last_points_framenumbers[idx]
                        if found_framenumber != missing_framenumber:
                            still_missing.append( missing_framenumber )
                            continue
                    elif idx == len(last_points_framenumbers):
                        still_missing.append( missing_framenumber )
                        continue

                    timestamp, points, camn_received_time = last_points[idx]
                    # make sure data is pure python, (not numpy)
                    missing_data.append( (int(camn), int(missing_framenumber), float(timestamp),
                                          float(camn_received_time), points) )
                if len(missing_data):
                    self.main_brain.receive_missing_data(cam_id, framenumber_offset, missing_data)

                if len(still_missing):
                    msg = '  Unable to find %d frames (camn %d):'%(
                            len(still_missing),
                            camn),
                    if len(still_missing) > 200:
                        LOG.info("%s %s + ... + %s" % (msg,still_missing[:25],still_missing[-25:]))
                    else:
                        LOG.info("%s %s" % (msg,still_missing))

            elif key == 'quit':
                self._image_sources[cam_no].join(0.1)
                # XXX TODO: quit and join chain threads
                with cam.lock:
                    cam.close()
                self.cam_status[cam_no] = 'destroyed'
                cmds=self.main_brain.close(cam_id)
            elif key == 'take_bg':
                globals['take_background'].set()
            elif key == 'clear_bg':
                globals['clear_background'].set()

            elif key == 'start_recording':
                if saver is None:
                    LOG.warn('no save thread -- cannot save movies')
                    continue
                raw_file_basename = cmds[key]

                raw_file_basename = os.path.expanduser(raw_file_basename)

                save_dir = os.path.split(raw_file_basename)[0]
                if not os.path.exists(save_dir):
                    LOG.info('making %s'%save_dir)
                    os.makedirs(save_dir)

                saver.start_recording(
                    raw_file_basename = raw_file_basename)

            elif key == 'stop_recording':
                saver.stop_recording()

            elif key == 'start_small_recording':
                if small_saver is None:
                    LOG.warn('no small save thread -- cannot save small movies')
                    continue

                small_filebasename = cmds[key]
                small_saver.start_recording(small_filebasename=small_filebasename)
            elif key == 'stop_small_recording':
                small_saver.stop_recording()
            elif key == 'cal':
                LOG.info('setting calibration')
                pmat, intlin, intnonlin = cmds[key]
                pmat = np.array(pmat)
                intlin = np.array(intlin)
                intnonlin = np.array(intnonlin)

                # XXX TODO: FIXME: thread crossing bug
                # these three should always be done together in this order:
                cam_processor.set_pmat( pmat )
                cam_processor.make_reconstruct_helper(intlin, intnonlin) # let grab thread make one
            else:
                raise ValueError('unknown key "%s"'%key)

def get_app_defaults():
    #some defaults are per camera node, other per flydra instance
    flydra_defaults = dict(
                       main_brain = socket.gethostname())

    for k,v in flydra_defaults.items():
        flydra_defaults[k] = rospy.get_param('/flydra/%s' % k, v)

    camnode_defaults = dict(
                    cams_only="",

                    # these are the most important 2D tracking parameters:
                    diff_threshold = 5,
                    n_sigma=7.0,
                    n_erode_absdiff=0,
                    color_range_1 = 0,
                    color_range_2 = 150,
                    color_range_3 = 255,
                    sat_thresh = 100,
                    red_only=0,
                    clear_threshold = 0.3,

                    debug_acquire=False,
                    disable_ifi_warning=False,
                    num_points=20,
                    software_roi_radius=10,
                    num_buffers=50,
                    small_save_radius=10,
                    background_frame_interval=50,
                    background_frame_alpha=1.0/50.0,
                    mask_images = None,
                    )
    for k,v in camnode_defaults.items():
        camnode_defaults[k] = rospy.get_param('~%s' % k, v)

    defaults = flydra_defaults.copy()
    defaults.update(camnode_defaults)

    return defaults

def main(rospy_init_node=True,cmdline_args=None):
    if rospy_init_node:
        if cmdline_args is not None:
            raise Exception("Not supported, makes no sense")
        cmdline_args = rospy.myargv()[1:]
        rospy.init_node('flydra_camera_node',disable_signals=True)
        rosthread = threading.Thread(target=rospy.spin,name='rosthread')
        rosthread.start()

    LOG.info('ROS name: %s' % rospy.get_name())

    if cmdline_args is None:
        cmdline_args = sys.argv[1:]

    parse_args_and_run(False, cmdline_args)
    if rospy_init_node:
        rospy.signal_shutdown("quit")

def benchmark():
    parse_args_and_run(True, sys.argv[1:])

def parse_args_and_run(benchmark, cmdline_args):
    parser = optparse.OptionParser(usage="%prog [options]",
                          version="%prog "+flydra.version.__version__)

    defaults = get_app_defaults()
    parser.set_defaults(**defaults)

    parser.add_option("--main-brain", type='string',
                      help="hostname of mainbrain")
    parser.add_option("--n-sigma", type='float',
                      help=("criterion used to determine if a pixel is significantly "
                            "different than the mean [default: %default]"))

    parser.add_option("--red-only", action='store_true', default=False,
                      help=("if set, detect points only in red channel (requires color cameras)"))

    parser.add_option("--debug-drop", action='store_true',
                      help="save debugging information regarding dropped network packets")

    parser.add_option("--debug-std", action='store_true',
                      help="show mean pixel STD every 200 frames")

    parser.add_option("--debug-acquire", action='store_true',
                      help="print to the console information on each frame")

    parser.add_option("--disable-ifi-warning", action='store_true',
                      help=("do not print a warning if the inter-frame-interval "
                            "(IFI) is longer than expected"))

    parser.add_option("--ignore-version", action='store_true',
                      help=("do not care if version is mismatched with mainbrain"))

    parser.add_option("--num-points", type="int",
                      help="number of points to track per cameras [default: %default]")

    parser.add_option("--software-roi-radius", type="int",
                      help="radius of software region of interest [default: %default]")

    parser.add_option("--background-frame-interval", type="int",
                      help="every N frames, add a new BG image to the accumulator [default: %default]")

    parser.add_option("--background-frame-alpha", type="float",
                      help="weight for each BG frame added to accumulator [default: %default]")

    parser.add_option("--mode-num", type="int", default=None,
                      help="force a camera mode")

    parser.add_option("--num-buffers", type="int",
                      help="force number of buffers [default: %default]")

    parser.add_option("--mask-images", type="string",
                      help="list of masks for each camera (uses OS-specific path separator, ':' for POSIX, ';' for Windows)")

    parser.add_option("--emulation-image-sources", type="string",
                      help=("list of image sources for each camera (uses OS-specific "
                            "path separator, ':' for POSIX, ';' for Windows) ends with '.fmf', "
                            "'.ufmf', or is '<random:params=x>'"))

    parser.add_option("--simulate-point-extraction", type="string",
                      help="list of image sources for each camera")

    parser.add_option("--force-cam-ids", type="string",
                      help="list of names for each camera (comma separated)")

    parser.add_option("--cams-only", type="string",
                      help="list cameras to use (comma separated)")

    parser.add_option("--show-cam-details", action='store_true', default=False)

    parser.add_option("--small-save-radius", type="int",
                      help='half the edge length of .ufmf movies [default: %default]')

    parser.add_option("--rosrate", type="float", dest='rosrate', default=1.,
                      help='desired framerate for the ROS raw image emitter (if ROS enabled)')

    parser.add_option("--sleep-first", type="int",
                      help='time to sleep before initilizing anything (to stop camera discovery races)')

    (options, args) = parser.parse_args(cmdline_args)

    if options.sleep_first is not None:
        time.sleep(options.sleep_first)

    app_state=AppState(options = options,
                       benchmark=benchmark,
                       )

    app=ConsoleApp(call_often = app_state.main_thread_task)
    app_state.set_quit_function( app.OnQuit )

    for (model, controller) in zip(app_state.get_image_sources(),
                                   app_state.get_image_controllers()):
        app.generate_view( model, controller )
    app.MainLoop()

if __name__=='__main__':
    main()

