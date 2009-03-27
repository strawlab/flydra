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

import pkg_resources
import os
BENCHMARK = int(os.environ.get('FLYDRA_BENCHMARK',0))
FLYDRA_BT = int(os.environ.get('FLYDRA_BT',0)) # threaded benchmark

#NAUGHTY_BUT_FAST = False
NAUGHTY_BUT_FAST = True

#DISABLE_ALL_PROCESSING = True
DISABLE_ALL_PROCESSING = False

near_inf = 9.999999e20

bright_non_gaussian_cutoff = 255
bright_non_gaussian_replacement = 5

import threading, time, socket, sys, struct, select, math, warnings
import Queue
import numpy
import numpy as nx
import numpy as np
import errno
import scipy.misc.pilutil
import numpy.dual

import contextlib

import motmot.ufmf.ufmf as ufmf
import motmot.realtime_image_analysis.slow

#import flydra.debuglock
#DebugLock = flydra.debuglock.DebugLock

import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
cam_iface = None # global variable, value set in main()
import motmot.cam_iface.choose as cam_iface_choose
from optparse import OptionParser

def DEBUG(*args):
    if 0:
        sys.stdout.write(' '.join(map(str,args))+'\n')
        sys.stdout.flush()

if not BENCHMARK:
    import Pyro.core, Pyro.errors, Pyro.util
    Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!
    Pyro.config.PYRO_TRACELEVEL = 3
    Pyro.config.PYRO_USER_TRACELEVEL = 3
    Pyro.config.PYRO_DETAILED_TRACEBACK = 1
    Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1
    ConnectionClosedError = Pyro.errors.ConnectionClosedError
else:
    class NonExistantError(Exception):
        pass
    ConnectionClosedError = NonExistantError
import flydra.reconstruct_utils as reconstruct_utils
import flydra.reconstruct
from flydra.reconstruct import do_3d_operations_on_2d_point

import camnode_utils
import motmot.FastImage.FastImage as FastImage
#FastImage.set_debug(3)
if os.name == 'posix' and sys.platform != 'darwin':
    import posix_sched

import flydra.debuglock
DebugLock = flydra.debuglock.DebugLock

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
        self.set_fps = self.noop
        self.log_message = self.noop
        self.close = self.noop
        self.camno = 0
    def noop(self,*args,**kw):
        return
    def get_cam2mainbrain_port(self,*args,**kw):
        return 12345
    def register_new_camera(self,*args,**kw):
        result = 'camdummy_%d'%self.camno
        self.camno += 1
        return result
    def get_and_clear_commands(self,*args,**kw):
        return {}

class DummySocket:
    def __init__(self,*args,**kw):
        self.connect = self.noop
        self.send = self.noop
        #sself.sendto = self.noop
    def noop(self,*args,**kw):
        return

import flydra.common_variables
NETWORK_PROTOCOL = flydra.common_variables.NETWORK_PROTOCOL

import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time

pt_fmt = '<dddddddddBBddBddddddBdd' # keep in sync with MainBrain.py
small_datafile_fmt = '<dII'

def TimestampEcho():
    # create listening socket
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = ''
    port = flydra.common_variables.timestamp_echo_listener_port
    sockobj.bind(( hostname, port))
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



class ProcessCamClass(object):
    def __init__(self,
                 cam2mainbrain_port=None,
                 cam_id=None,
                 log_message_queue=None,
                 max_num_points=None,
                 roi2_radius=None,
                 bg_frame_interval=None,
                 bg_frame_alpha=None,
                 cam_no=-1,
                 main_brain_hostname=None,
                 mask_image=None,
                 diff_threshold_shared=None,
                 clear_threshold_shared=None,
                 n_sigma_shared=None,
                 framerate = None,
                 lbrt=None,
                 max_height=None,
                 max_width=None,
                 globals = None,
                 options = None,
                 initial_image_dict = None,
                 benchmark = False,
                 ):
        self.benchmark = benchmark
        self.options = options
        self.globals = globals
        self.main_brain_hostname = main_brain_hostname
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
        self._scale_factor = None # for 3D calibration stuff
        self.cam_no_str = str(cam_no)

        self._chain = camnode_utils.ChainLink()
        self._initial_image_dict = initial_image_dict

    def get_chain(self):
        return self._chain

    def get_scale_factor(self):
        return self._scale_factor
    def set_scale_factor(self,value):
        self._scale_factor = value

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
            self._scale_factor = None
            self._pmat_meters = None
            self._pmat_meters_inv = None
            self._camera_center_meters = None
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

        scale_array = numpy.ones((3,4))
        scale_array[:,3] = self._scale_factor # mulitply last column by scale_factor
        self._pmat_meters = scale_array*self._pmat # element-wise multiplication
        self._pmat_meters_inv = numpy.dual.pinv(self._pmat_meters)
        P = self._pmat_meters
        # find camera center in 3D world coordinates
        col0_asrow = P[nx.newaxis,:,0]
        col1_asrow = P[nx.newaxis,:,1]
        col2_asrow = P[nx.newaxis,:,2]
        col3_asrow = P[nx.newaxis,:,3]
        X = determinant(  r_[ col1_asrow, col2_asrow, col3_asrow ] )
        Y = -determinant( r_[ col0_asrow, col2_asrow, col3_asrow ] )
        Z = determinant(  r_[ col0_asrow, col1_asrow, col3_asrow ] )
        T = -determinant( r_[ col0_asrow, col1_asrow, col2_asrow ] )
        self._camera_center_meters = nx.array( [ X/T, Y/T, Z/T, 1.0 ] )

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
                print 'xpt',xpt
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
                    # self._pmat_inv and sef._pmat_meters.)
                    (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4,
                     ray5) = do_3d_operations_on_2d_point(self._hlper,x0u,y0u,
                                                          self._pmat_inv, self._pmat_meters_inv,
                                                          self._camera_center, self._camera_center_meters,
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

            # see pt_fmt struct definition:
            pt = (x0_abs, y0_abs, area, slope, eccentricity,
                  p1, p2, p3, p4, line_found, slope_found,
                  x0u, y0u,
                  ray_valid,
                  ray0, ray1, ray2, ray3, ray4, ray5,
                  cur_val, mean_val, sumsqf_val)
            points.append( pt )
        return points

    def mainloop(self):
        disable_ifi_warning = self.options.disable_ifi_warning
        globals = self.globals

        self._globals = globals
        DEBUG_DROP = globals['debug_drop']
        if DEBUG_DROP:
            debug_fd = open('debug_framedrop_cam.txt',mode='w')

        # questionable optimization: speed up by eliminating namespace lookups
        cam_quit_event_isSet = globals['cam_quit_event'].isSet
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
            if NETWORK_PROTOCOL == 'udp':
                coord_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif NETWORK_PROTOCOL == 'tcp':
                coord_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                coord_socket.connect((self.main_brain_hostname,self.cam2mainbrain_port))
            else:
                raise ValueError('unknown NETWORK_PROTOCOL')

        old_ts = time.time()
        old_fn = None
        points = []

        ## if os.name == 'posix' and not BENCHMARK:
        ##     try:
        ##         max_priority = posix_sched.get_priority_max( posix_sched.FIFO )
        ##         sched_params = posix_sched.SchedParam(max_priority)
        ##         posix_sched.setscheduler(0, posix_sched.FIFO, sched_params)
        ##         msg = 'excellent, grab thread running in maximum prioity mode'
        ##     except Exception, x:
        ##         msg = 'WARNING: could not run in maximum priority mode:', str(x)
        ##     self.log_message_queue.put((self.cam_id,time.time(),msg))
        ##     print msg

        #FastImage.set_debug(3) # let us see any images malloced, should only happen on hardware ROI size change


        #################### initialize images ############

        running_mean8u_im_full = self.realtime_analyzer.get_image_view('mean') # this is a view we write into
        absdiff8u_im_full = self.realtime_analyzer.get_image_view('absdiff') # this is a view we write into

        mask_im = self.realtime_analyzer.get_image_view('mask') # this is a view we write into
        newmask_fi = FastImage.asfastimage( self.mask_image )
        newmask_fi.get_8u_copy_put(mask_im, max_frame_size)

        # allocate images and initialize if necessary

        bg_image_full = FastImage.FastImage8u(max_frame_size)
        std_image_full = FastImage.FastImage8u(max_frame_size)

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
        mean_duration_no_bg = 0.0053 # starting value
        mean_duration_bg = 0.020 # starting value

        # set ROI views of full-frame images
        bg_image = bg_image_full.roi(cur_roi_l, cur_roi_b, cur_fisize) # set ROI view
        std_image = std_image_full.roi(cur_roi_l, cur_roi_b, cur_fisize) # set ROI view
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

            if 1:
                print 'WARNING: ignoring initial images and taking new background'
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
                chainbuf.updated_bg_image = None
                chainbuf.updated_cmp_image = None
                chainbuf.updated_running_mean_image = None
                chainbuf.updated_running_sumsqf_image = None

                hw_roi_frame = chainbuf.get_buf()
                cam_received_time = chainbuf.cam_received_time

                # get best guess as to when image was taken
                timestamp=chainbuf.timestamp
                framenumber=chainbuf.framenumber

                if 1:
                    if old_fn is None:
                        # no old frame
                        old_fn = framenumber-1
                    if framenumber-old_fn > 1:
                        n_frames_skipped = framenumber-old_fn-1
                        msg = '  frames apparently skipped: %d'%(n_frames_skipped,)
                        self.log_message_queue.put((self.cam_id,time.time(),msg))
                        print >> sys.stderr, msg
                    else:
                        n_frames_skipped = 0

                    diff = timestamp-old_ts
                    time_per_frame = diff/(n_frames_skipped+1)
                    if not disable_ifi_warning:
                        if time_per_frame > 2*self.shortest_IFI:
                            msg = 'Warning: IFI is %f on %s at %s (frame skipped?)'%(time_per_frame,self.cam_id,time.asctime())
                            self.log_message_queue.put((self.cam_id,time.time(),msg))
                            print >> sys.stderr, msg

                old_ts = timestamp
                old_fn = framenumber

                work_start_time = time.time()
                xpoints = self.realtime_analyzer.do_work(hw_roi_frame,
                                                         timestamp, framenumber, use_roi2,
                                                         use_cmp_isSet(),
                                                         #max_duration_sec=0.010, # maximum 10 msec in here
                                                         max_duration_sec=self.shortest_IFI-0.0005, # give .5 msec for other processing
                                                         return_debug_values=1,
                                                         )
                ## if len(xpoints)>=self.max_num_points:
                ##     msg = 'Warning: cannot save acquire points this frame because maximum number already acheived'
                ##     print >> sys.stderr, msg
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

                work_done_time = time.time()

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
                        print 'ERROR: deleting old frames to make room for new ones! (and sleeping)'
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
                    #print ' '*20,'put frame'

                do_bg_maint = False

                if initial_take_bg_state is not None:
                    assert initial_take_bg_state == 'gather'
                    n_initial_take = 50
                    if 1:
                        initial_take_frames.append( numpy.array(hw_roi_frame,copy=True) )
                        if len( initial_take_frames ) >= n_initial_take:

                            initial_take_frames = numpy.array( initial_take_frames, dtype=numpy.float32 )
                            mean_frame = numpy.mean( initial_take_frames, axis=0)
                            sumsqf_frame = numpy.sum(initial_take_frames**2, axis=0)/len( initial_take_frames )

                            numpy.asarray(running_mean_im)[:,:] = mean_frame
                            numpy.asarray(running_sumsqf)[:,:] = sumsqf_frame
                            print 'using slow method, calculated mean and sumsqf frames from first %d frames'%(n_initial_take,)

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
                    print 'taking new bg'
                    # reset background image with current frame as mean and 0 STD
                    if cur_fisize != max_frame_size:
                        print cur_fisize
                        print max_frame_size
                        print 'ERROR: can only take background image if not using ROI'
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
                        print '%s mean STD %.2f'%(self.cam_id,mean_std)

                if clear_background_isSet():
                    # reset background image with 0 mean and 0 STD
                    running_mean_im.set_val( 0, max_frame_size )
                    running_mean8u_im.set_val(0, max_frame_size )
                    running_sumsqf.set_val( 0, max_frame_size )
                    compareframe8u.set_val(0, max_frame_size )
                    bg_changed = True
                    clear_background_clear()

                if bg_changed:
                    if 1:
##                        bg_image = running_mean8u_im.get_8u_copy(running_mean8u_im.size)
##                        std_image = compareframe8u.get_8u_copy(compareframe8u.size)
                        running_mean8u_im.get_8u_copy_put(bg_image, running_mean8u_im.size)
                        compareframe8u.get_8u_copy_put(std_image, compareframe8u.size)
                    elif 0:
                        bg_image = nx.array(running_mean8u_im) # make copy (we don't want to send live versions of image
                        std_image = nx.array(compareframe8u) # make copy (we don't want to send live versions of image
                    else:
                        bg_image = running_mean8u_im
                        std_image = compareframe8u

                    chainbuf.updated_bg_image = numpy.array( bg_image, copy=True )
                    chainbuf.updated_cmp_image = numpy.array( std_image, copy=True )

                    chainbuf.updated_running_mean_image = numpy.array( running_mean_im, copy=True )
                    chainbuf.updated_running_sumsqf_image = numpy.array( running_sumsqf, copy=True )

#                    globals['current_bg_frame_and_timestamp']=bg_image,std_image,timestamp # only used when starting to save
##                     if not BENCHMARK:
##                         globals['incoming_bg_frames'].put(
##                             (bg_image,std_image,timestamp,framenumber) ) # save it
                    bg_changed = False

                if self.diff_threshold_shared.is_new_value_waiting():
                    self.realtime_analyzer.diff_threshold = (
                        self.diff_threshold_shared.get_nowait() )

                if self.clear_threshold_shared.is_new_value_waiting():
                    self.realtime_analyzer.clear_threshold = (
                        self.clear_threshold_shared.get_nowait() )

                # XXX could speed this with a join operation I think
                data = struct.pack('<ddliI',timestamp,cam_received_time,
                                   framenumber,len(points),n_frames_skipped)
                for point_tuple in points:
                    try:
                        data = data + struct.pack(pt_fmt,*point_tuple)
                    except:
                        print 'error-causing data: ',point_tuple
                        raise
                if 0:
                    local_processing_time = (time.time()-cam_received_time)*1e3
                    print 'local_processing_time % 3.1f'%local_processing_time
                if NETWORK_PROTOCOL == 'udp':
                    try:
                        coord_socket.sendto(data,
                                            (self.main_brain_hostname,self.cam2mainbrain_port))
                    except socket.error, err:
                        import traceback
                        print >> sys.stderr, 'WARNING: ignoring error:'
                        traceback.print_exc()

                elif NETWORK_PROTOCOL == 'tcp':
                    coord_socket.send(data)
                else:
                    raise ValueError('unknown NETWORK_PROTOCOL')
                if DEBUG_DROP:
                    debug_fd.write('%d,%d\n'%(framenumber,len(points)))
                #print 'sent data...'

                if 0 and self.new_roi.isSet():
                    with self.new_roi_data_lock:
                        lbrt = self.new_roi_data
                        self.new_roi_data = None
                        self.new_roi.clear()
                    l,b,r,t=lbrt
                    w = r-l+1
                    h = t-b+1
                    self.realtime_analyzer.roi = lbrt
                    print 'desired l,b,w,h',l,b,w,h

                    l2,b2,w2,h2 = self.cam.get_frame_roi()
                    if ((l==l2) and (b==b2) and (w==w2) and (h==h2)):
                        print 'current ROI matches desired ROI - not changing'
                    else:
                        self.cam.set_frame_roi(l,b,w,h)
                        l,b,w,h = self.cam.get_frame_roi()
                        print 'actual l,b,w,h',l,b,w,h
                    r = l+w-1
                    t = b+h-1
                    cur_fisize = FastImage.Size(w, h)
                    hw_roi_frame = fi8ufactory( cur_fisize )
                    self.realtime_analyzer.roi = (l,b,r,t)

                    # set ROI views of full-frame images
                    bg_image = bg_image_full.roi(l, b, cur_fisize) # set ROI view
                    std_image = std_image_full.roi(l, b, cur_fisize) # set ROI view
                    running_mean8u_im = running_mean8u_im_full.roi(l, b, cur_fisize) # set ROI view
                    running_mean_im = running_mean_im_full.roi(l, b, cur_fisize)  # set ROI view
                    fastframef32_tmp = fastframef32_tmp_full.roi(l, b, cur_fisize)  # set ROI view
                    mean2 = mean2_full.roi(l, b, cur_fisize)  # set ROI view
                    std2 = std2_full.roi(l, b, cur_fisize)  # set ROI view
                    running_stdframe = running_stdframe_full.roi(l, b, cur_fisize)  # set ROI view
                    compareframe = compareframe_full.roi(l, b, cur_fisize)  # set ROI view
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
                    print 'saving movies','-'*50
                    raw_movie = FlyMovieFormat.FlyMovieSaver(full_raw,
                                                             format='MONO8',
                                                             bits_per_pixel=8,
                                                             version=3)
                    bg_movie = FlyMovieFormat.FlyMovieSaver(full_bg,
                                                            format='MONO32f',
                                                            bits_per_pixel=32,
                                                            version=3)
                    std_movie = FlyMovieFormat.FlyMovieSaver(full_std,
                                                             format='MONO32f',
                                                             bits_per_pixel=32,
                                                             version=3)
                    state = 'saving'

                    if last_bgcmp_image_timestamp is not None:
                        bg_movie.add_frame(FastImage.asfastimage(last_running_mean_image),
                                           last_bgcmp_image_timestamp,
                                           error_if_not_fast=True)
                        std_movie.add_frame(FastImage.asfastimage(last_running_sumsqf_image),
                                            last_bgcmp_image_timestamp,
                                            error_if_not_fast=True)
                    else:
                        print 'WARNING: could not save initial bg and std frames'

                elif cmd[0] == 'stop':
                    print 'done saving movies','-'*50
                    raw_movie.close()
                    bg_movie.close()
                    std_movie.close()
                    state = 'pass'

            # 2: block for image data
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf: # must do on every frame
                if chainbuf.quit_now:
                    break

                if chainbuf.updated_bg_image is not None:
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
                    if chainbuf.updated_bg_image is not None:
                        meancmp.append( (chainbuf.updated_bg_image,
                                         chainbuf.updated_cmp_image,
                                         chainbuf.updated_running_mean_image,
                                         chainbuf.updated_running_sumsqf_image,
                                         chainbuf.cam_received_time)) # these were copied in process thread

            # 3: grab any more that are here
            try:
                with camnode_utils.use_buffer_from_chain(self._chain,blocking=False) as chainbuf:
                    if chainbuf.quit_now:
                        break

                    if state == 'saving':
                        raw.append( (numpy.array(chainbuf.get_buf(), copy=True),
                                     chainbuf.cam_received_time) )
                        if chainbuf.updated_bg_image is not None:
                            meancmp.append( (chainbuf.updated_bg_image,
                                             chainbuf.updated_cmp_image,
                                             chainbuf.updated_running_mean_image,
                                             chainbuf.updated_running_sumsqf_image,
                                             chainbuf.cam_received_time)) # these were copied in process thread
            except Queue.Empty:
                pass

            # 4: actually save the data
            #   TODO: switch to add_frames() method which doesn't acquire GIL after each frame.
            if state == 'saving':
                for frame,timestamp in raw:
                    raw_movie.add_frame(FastImage.asfastimage(frame),timestamp,error_if_not_fast=True)
                for bg,cmp,running_mean,running_sumsqf,timestamp in meancmp:
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
                    raw_file_basename = os.path.expanduser(os.path.splitext(filename_base)[0])

                    full_bg = raw_file_basename + '_mean.fmf'
                    full_std = raw_file_basename + '_sumsqf.fmf'
                    bg_movie = FlyMovieFormat.FlyMovieSaver(full_bg,
                                                            format='MONO32f',
                                                            bits_per_pixel=32,
                                                            version=3)
                    std_movie = FlyMovieFormat.FlyMovieSaver(full_std,
                                                             format='MONO32f',
                                                             bits_per_pixel=32,
                                                             version=3)
                    state = 'saving'
                    if last_bgcmp_image_timestamp is not None:
                        bg_movie.add_frame(FastImage.asfastimage(last_running_mean_image),
                                           last_bgcmp_image_timestamp,
                                           error_if_not_fast=True)
                        std_movie.add_frame(FastImage.asfastimage(last_running_sumsqf_image),
                                            last_bgcmp_image_timestamp,
                                            error_if_not_fast=True)
                    else:
                        print 'WARNING: could not save initial bg and std frames'

                elif cmd[0] == 'stop':
                    if self._ufmf is not None:
                        self._ufmf.close()
                        self._ufmf = None
                    bg_movie.close()
                    std_movie.close()
                    state = 'pass'

            # block for images
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf:
                if chainbuf.quit_now:
                    break

                if chainbuf.updated_bg_image is not None:
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
                    if chainbuf.updated_bg_image is not None:
                        meancmp.append( (chainbuf.updated_bg_image,
                                         chainbuf.updated_cmp_image,
                                         chainbuf.updated_running_mean_image,
                                         chainbuf.updated_running_sumsqf_image,
                                         chainbuf.cam_received_time)) # these were copied in process thread
                    if self._ufmf is None:
                        frame1 = numpy.asarray(chainbuf.get_buf())
                        timestamp1 = chainbuf.cam_received_time
                        filename_base = os.path.expanduser(filename_base)
                        dirname = os.path.split(filename_base)[0]

                        with self._mkdir_lock:
                            # Because this is a multi-threaded
                            # program, sometimes another thread will
                            # try to create this directory.
                            if not os.path.exists(dirname):
                                os.makedirs(dirname)
                        filename = filename_base + '.ufmf'
                        if 1:
                            print 'saving to',filename
                        self._ufmf = ufmf.UfmfSaver( filename,
                                                     frame1,
                                                     timestamp1,
                                                     image_radius=self.options.small_save_radius )
                    self._tobuf( chainbuf )

            # grab any more that are here
            try:
                with camnode_utils.use_buffer_from_chain(self._chain,blocking=False) as chainbuf:
                    if chainbuf.quit_now:
                        break

                    if state == 'saving':
                        self._tobuf( chainbuf ) # actually save the .ufmf data
                        if chainbuf.updated_bg_image is not None:
                            meancmp.append( (chainbuf.updated_bg_image,
                                             chainbuf.updated_cmp_image,
                                             chainbuf.updated_running_mean_image,
                                             chainbuf.updated_running_sumsqf_image,
                                             chainbuf.cam_received_time)) # these were copied in process thread
            except Queue.Empty:
                pass

            # actually save the data
            #   TODO: switch to add_frames() method which doesn't acquire GIL after each frame.
            if state == 'saving':
                for bg,cmp,running_mean,running_sumsqf,timestamp in meancmp:
                    bg_movie.add_frame(FastImage.asfastimage(running_mean),timestamp,error_if_not_fast=True)
                    std_movie.add_frame(FastImage.asfastimage(running_sumsqf),timestamp,error_if_not_fast=True)
            del meancmp[:]

    def _tobuf( self, chainbuf ):
        frame = chainbuf.get_buf()
        if 0:
            print 'saving %d points'%(len(chainbuf.processed_points ),)
        self._ufmf.add_frame( frame, chainbuf.cam_received_time, chainbuf.processed_points )

class ImageSource(threading.Thread):
    """One instance of this class for each camera. Do nothing but get
    new frames, copy them, and pass to listener chain."""
    def __init__(self,
                 chain=None,
                 cam=None,
                 buffer_pool=None,
                 debug_acquire = False,
                 cam_no = None,
                 quit_event = None,
                 ):

        threading.Thread.__init__(self,name='ImageSource')
        self._chain = chain
        self.cam = cam
        self.buffer_pool = buffer_pool
        self.debug_acquire = debug_acquire
        self.cam_no_str = str(cam_no)
        self.quit_event = quit_event
    def set_chain(self,new_chain):
        # XXX TODO FIXME: put self._chain behind lock
        if self._chain is not None:
            raise NotImplementedError('replacing a processing chain not implemented')
        self._chain = new_chain
    def get_buffer_pool(self):
        return self.buffer_pool
    def run(self):
        print 'ImageSource running in process',os.getpid()
        buffer_pool = self.buffer_pool
        cam_quit_event_isSet = self.quit_event.isSet
        while not cam_quit_event_isSet():
            self._block_until_ready() # no-op for realtime camera processing
            if buffer_pool.get_num_outstanding_buffers() > 100:
                # Grab some frames (wait) until the number of
                # outstanding buffers decreases -- give processing
                # threads time to catch up.
                print ('*'*80+'\n')*5
                print 'ERROR: We seem to be leaking buffers - will not acquire more images for a while!'
                print ('*'*80+'\n')*5
                while 1:
                    self._grab_buffer_quick()
                    if buffer_pool.get_num_outstanding_buffers() < 10:
                        print 'Resuming normal image acquisition'
                        break

            # this gets a new (unused) buffer from the preallocated pool
            with get_free_buffer_from_pool( buffer_pool ) as chainbuf:
                chainbuf.quit_now = False

                _bufim = chainbuf.get_buf()

                try_again_condition, timestamp, framenumber = self._grab_into_buffer( _bufim )
                if try_again_condition:
                    continue

                if self.debug_acquire:
                    stdout_write(self.cam_no_str)

                cam_received_time = time.time()

                chainbuf.cam_received_time = cam_received_time
                chainbuf.timestamp = timestamp
                chainbuf.framenumber = framenumber

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
        self._prosilica_hack_last_framenumber = None
        self._prosilica_hack_framenumber_offset = 0

    def _block_until_ready(self):
        # no-op for realtime camera processing
        pass

    def spawn_controller(self):
        controller = ImageSourceBaseController()
        return controller

    def _grab_buffer_quick(self):
        try:
            trash = self.cam.grab_next_frame_blocking()
        except cam_iface.BuffersOverflowed:
            msg = 'ERROR: buffers overflowed on %s at %s'%(self.cam_no_str,time.asctime(time.localtime(now)))
            self.log_message_queue.put((self.cam_no_str,now,msg))
            print >> sys.stderr, msg
        except cam_iface.FrameDataMissing:
            pass
        except cam_iface.FrameSystemCallInterruption:
            pass

    def _grab_into_buffer(self, _bufim ):
        try_again_condition= False

        try:
            self.cam.grab_next_frame_into_buf_blocking(_bufim)
        except cam_iface.BuffersOverflowed:
            if self.debug_acquire:
                stdout_write('(O%s)'%self.cam_no_str)
            now = time.time()
            msg = 'ERROR: buffers overflowed on %s at %s'%(self.cam_no_str,time.asctime(time.localtime(now)))
            self.log_message_queue.put((self.cam_no_str,now,msg))
            print >> sys.stderr, msg
            try_again_condition = True
        except cam_iface.FrameDataMissing:
            if self.debug_acquire:
                stdout_write('(M%s)'%self.cam_no_str)
            now = time.time()
            msg = 'Warning: frame data missing on %s at %s'%(self.cam_no_str,time.asctime(time.localtime(now)))
            #self.log_message_queue.put((self.cam_no_str,now,msg))
            print >> sys.stderr, msg
            try_again_condition = True
        except cam_iface.FrameSystemCallInterruption:
            if self.debug_acquire:
                stdout_write('(S%s)'%self.cam_no_str)
            try_again_condition = True

        if not try_again_condition:
            # get best guess as to when image was taken
            timestamp=self.cam.get_last_timestamp()
            framenumber=self.cam.get_last_framenumber()

            # Hack to deal with Prosilica framenumber resetting at
            # 65535 (even though it's an unsigned long).

            _prosilica_hack_max_skipped_frames = 100
            if ((framenumber<=_prosilica_hack_max_skipped_frames) and
                (self._prosilica_hack_last_framenumber >= 65536-_prosilica_hack_max_skipped_frames) and
                (self._prosilica_hack_last_framenumber < 65536)):
                # We're dealing with a Prosilica camera which just
                # rolled over.
                self._prosilica_hack_framenumber_offset += 65636
            self._prosilica_hack_last_framenumber = framenumber
            framenumber += self._prosilica_hack_framenumber_offset
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
        super( ImageSourceFakeCamera, self).__init__(*args,**kw)

    def _block_until_ready(self):
        if isinstance(self._fake_cam,FakeCameraFromRNG):
            return

        while 1:
            if self.quit_event.isSet():
                return

            # This lock ping-pongs execution back and forth between
            # "acquire" and process.

            self._do_step.wait(0.01) # timeout
            if self._do_step.isSet():
                self._do_step.clear()
                return

    def spawn_controller(self):
        if isinstance(self._fake_cam,FakeCameraFromRNG):
            # no control necessary for random number generator
            controller = ImageSourceBaseController()
            return controller

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
                #print 'self._fake_cam.is_finished()',self._fake_cam.is_finished()
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
        self.cam.grab_next_frame_into_buf_blocking(_bufim, self.quit_event)

        try_again_condition = False
        timestamp=self.cam.get_last_timestamp()
        framenumber=self.cam.get_last_framenumber()
        return try_again_condition, timestamp, framenumber

class FakeCamera(object):
    def start_camera(self):
        # no-op
        pass

    def get_framerate(self):
        return 123456

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

    def get_num_trigger_modes(self):
        return 1

    def get_trigger_mode_string(self,i):
        return 'fake camera trigger'

class FakeCameraFromNetwork(FakeCamera):
    def __init__(self,id,frame_size):
        self.id = id
        self.frame_size = frame_size
        self.remote = None

    def get_frame_roi(self):
        w,h = self.frame_size
        return 0,0,w,h

    def _ensure_remote(self):
        if self.remote is None:
            hostname = 'localhost'
            port = flydra.common_variables.emulated_camera_control
            name = 'remote_camera_source'
            remote_URI = "PYROLOC://%s:%d/%s" % (hostname,port,name)
            self.remote = Pyro.core.getProxyForURI(remote_URI)

    def grab_next_frame_into_buf_blocking(self,buf, quit_event):
        # XXX TODO: implement quit_event checking
        self._ensure_remote()

        pt_list = self.remote.get_point_list(self.id) # this will block...
        w,h = self.frame_size
        new_raw = np.asarray( buf )
        assert new_raw.shape == (h,w)
        for pt in pt_list:
            x,y = pt
            xi = int(round(x))
            yi = int(round(y))
            new_raw[yi,xi] = 10
        return new_raw

    def get_last_timestamp(self):
        self._ensure_remote()
        return self.remote.get_last_timestamp(self.id) # this will block...

    def get_last_framenumber(self):
        self._ensure_remote()
        return self.remote.get_last_framenumber(self.id) # this will block...

class FakeCameraFromRNG(FakeCamera):
    def __init__(self,id,frame_size):
        self.id = id
        self.frame_size = frame_size
        self.remote = None
        self.last_timestamp = 0.0
        self.last_count = -1

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

class FakeCameraFromFMF(FakeCamera):
    def __init__(self,filename):
        self.fmf_recarray = FlyMovieFormat.mmap_flymovie( filename )
        if 0:
            print 'short!'
            self.fmf_recarray = self.fmf_recarray[:600]

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
        #print "len( self.fmf_recarray['frame'] )",len( self.fmf_recarray['frame'] )
        #print "self._curframe.get()",self._curframe.get()
        result = self._curframe.get() >= len( self.fmf_recarray['frame'] )
        #print result
        #print
        return result

def create_cam_for_emulation_image_source( filename_or_pseudofilename ):
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
            print '*'*80
            print 'WARNING timestamps of first image frame is not before mean image timestamps. they are'
            print ' raw .fmf: %s'%repr(t0)
            print ' mean .fmf:  %s'%repr(mean_t0)
            print ' sumsqf .fmf: %s'%repr(sumsqf_t0)
            print '*'*80

        initial_image_dict = {'mean':mean_ra['frame'][0],
                              'sumsqf':sumsqf_ra['frame'][0],  # not really mean2 (actually running_sumsqf)
                              'raw':fmf_ra['frame'][0]}
        if 0 and len( mean_ra['frame'] ) > 1:
            print ("no current support for reading back multi-frame "
                   "background/cmp. (But this should not be necessary, "
                   "as you can reconstruct them, anyway.)")

    elif fname.endswith('.ufmf'):
        raise NotImplementedError('patience, young grasshopper')
    elif fname.startswith('<net') and fname.endswith('>'):
        args = fname[4:-1].strip()
        args = args.split()
        port, width, height = map(int, args)
        cam = FakeCameraFromNetwork(port,(width,height))
        ImageSourceModel = ImageSourceFakeCamera
        l,b,w,h = cam.get_frame_roi()
        del l,b

        mean = np.ones( (h,w), dtype=np.uint8 )
        sumsqf = np.ones( (h,w), dtype=np.uint8 )
        raw = np.ones( (h,w), dtype=np.uint8 )

        initial_image_dict = {'mean':mean,
                              'sumsqf':sumsqf,
                              'raw':raw}
    elif fname == '<rng>':
        width, height = 640, 480
        cam = FakeCameraFromRNG('fakecam1',(width,height))
        ImageSourceModel = ImageSourceFakeCamera
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
            raise NotImplementedError('no control in ConsoleApp for %s'%controller)

class AppState(object):
    """This class handles all camera states, properties, etc."""
    def __init__(self,
                 benchmark = False,
                 options = None,
                 ):
        global cam_iface

        self.options = options
        self._real_quit_function = None

        if options.server is None:
            self.main_brain_hostname = default_main_brain_hostname
        else:
            self.main_brain_hostname = options.server

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

            cam_iface = cam_iface_choose.import_backend( options.backend, options.wrapper )

            all_cam_info_list = [
                (cam_iface.get_camera_info(i),i) for i in range(cam_iface.get_num_cameras()) ]
            all_cam_info_list.sort() # make sure list is always in same order for given cameras
            all_cam_info_list.reverse() # any ordering will do, but reverse for historical reasons
            cam_order = [ x[1] for x in all_cam_info_list]
            print 'camera order',cam_order
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

        self.reconstruct_helper = []
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

            globals['debug_drop']=options.debug_drop
            globals['debug_acquire']=options.debug_acquire
            globals['incoming_raw_frames']=Queue.Queue()
            globals['raw_fmf_and_bg_fmf']=None
            globals['most_recent_frame_potentially_corrupt']=None
            globals['saved_bg_frame']=False

            # control flow events for threading model
            globals['cam_quit_event'] = threading.Event()
            globals['listen_thread_done'] = threading.Event()
            globals['take_background'] = threading.Event()
            globals['clear_background'] = threading.Event()
            globals['collecting_background'] = threading.Event()
            globals['collecting_background'].set()
            globals['export_image_name'] = 'raw'
            globals['use_cmp'] = threading.Event()

            #print 'not using ongoing variance estimate'
            globals['use_cmp'].set()

            if cam_iface is not None:
                backend = cam_iface.get_driver_name()
                N_modes = cam_iface.get_num_modes(cam_order[cam_no])
                use_mode = options.mode_num
                print 'camera info:',cam_iface.get_camera_info(cam_order[cam_no])
                print '%d available modes:'%N_modes
                for i in range(N_modes):
                    mode_string = cam_iface.get_mode_string(cam_order[cam_no],i)
                    print '  mode %d: %s'%(i,mode_string)
                    if 'format7_0' in mode_string.lower():
                        # prefer format7_0
                        if use_mode is None:
                            use_mode = i
                if use_mode is None:
                    use_mode = 0
                cam = cam_iface.Camera(cam_order[cam_no],options.num_buffers,use_mode)
                print 'using mode %d: %s'%(use_mode, cam_iface.get_mode_string(cam_order[cam_no],use_mode))
                ImageSourceModel = ImageSourceFromCamera

                initial_image_dict = None
            elif options.simulate_point_extraction: # emulate points
                # call factory function
                (cam, ImageSourceModel,
                 initial_image_dict)  = create_cam_for_emulation_image_source( image_sources[cam_no] )
            elif benchmark: # emulate full images with random number generator
                # call factory function
                (cam, ImageSourceModel, initial_image_dict) = \
                      create_cam_for_emulation_image_source( '<rng>' )
            else: # emulate full images
                # call factory function
                (cam, ImageSourceModel,
                 initial_image_dict)  = create_cam_for_emulation_image_source( emulation_image_sources[cam_no] )

            if initial_image_dict is None:
                globals['take_background'].set()
            else:
                globals['take_background'].clear()

            initial_images[cam_no] = initial_image_dict

            self.all_cams[cam_no] = cam
            if cam is not None:
                cam.start_camera()  # start camera
            self.cam_status[cam_no]= 'started'
            if ImageSourceModel is not None:
                l,b,w,h = cam.get_frame_roi()
                buffer_pool = PreallocatedBufferPool(FastImage.Size(w,h))
                del l,b,w,h
                image_source = ImageSourceModel(chain = None,
                                                cam = cam,
                                                buffer_pool = buffer_pool,
                                                debug_acquire = options.debug_acquire,
                                                cam_no = cam_no,
                                                quit_event = globals['cam_quit_event'],
                                                )

                controller = image_source.spawn_controller()

                image_source.setDaemon(True)
                image_source.start()
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
            Pyro.core.initClient(banner=0)
            port = flydra.common_variables.mainbrain_port
            name = 'main_brain'
            main_brain_URI = "PYROLOC://%s:%d/%s" % (self.main_brain_hostname,port,name)
            try:
                self.main_brain = Pyro.core.getProxyForURI(main_brain_URI)
            except:
                print 'ERROR while connecting to',main_brain_URI
                raise
            self.main_brain._setOneway(['set_image','set_fps','close','log_message','receive_missing_data'])

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
            left,top,width,height = cam.get_frame_roi()
            del left,top
            globals = self.globals[cam_no] # shorthand

            if mask_images is not None:
                mask_image_fname = mask_images[cam_no]
                im = scipy.misc.pilutil.imread( mask_image_fname )
                if len(im.shape) != 3:
                    raise ValueError('mask image must have color channels')
                if im.shape[2] != 4:
                    raise ValueError('mask image must have an alpha (4th) channel')
                alpha = im[:,:,3]
                if numpy.any((alpha > 0) & (alpha < 255)):
                    print ('WARNING: some alpha values between 0 and '
                           '255 detected. Only zero and non-zero values are '
                           'considered.')
                mask = alpha.astype(numpy.bool)
            else:
                mask = numpy.zeros( (height,width), dtype=numpy.bool )
            # mask is currently an array of bool
            mask = mask.astype(numpy.uint8)*255


            # get settings
            scalar_control_info = {}

            if 1:
                # trigger modes
                N_trigger_modes = cam.get_num_trigger_modes()
                print '  %d available trigger modes:'%N_trigger_modes
                for i in range(N_trigger_modes):
                    mode_string = cam.get_trigger_mode_string(i)
                    print '  mode %d: %s'%(i,mode_string)
                scalar_control_info['N_trigger_modes'] = N_trigger_modes
                # XXX TODO: scalar_control_info['trigger_mode'] # current value

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
                            print 'setting camera property "%s" to manual mode'%(props['name'],)
                            cam.set_camera_property( prop_num, new_value, 0 )
                        except:
                            print 'error while setting property %s to %d (from %d)'%(props['name'],new_value,current_value)
                            raise
                    else:
                        print 'not setting property %s to %d (from %d) because out of range (%d<=value<=%d)'%(props['name'],new_value,current_value,min_value,max_value)

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

            scalar_control_info['width'] = width
            scalar_control_info['height'] = height
            scalar_control_info['roi'] = 0,0,width-1,height-1
            scalar_control_info['max_framerate'] = cam.get_framerate()
            scalar_control_info['collecting_background']=globals['collecting_background'].isSet()
            scalar_control_info['debug_drop']=globals['debug_drop']
            scalar_control_info['expected_trigger_framerate'] = 0.0

            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            if force_cam_ids is None:
                force_cam_id = None
            else:
                force_cam_id = force_cam_ids[cam_no]
            cam_id = self.main_brain.register_new_camera(cam_no,
                                                         scalar_control_info,
                                                         port,
                                                         force_cam_id=force_cam_id,
                                                         )

            self.all_cam_ids[cam_no]=cam_id
            cam2mainbrain_port = self.main_brain.get_cam2mainbrain_port(self.all_cam_ids[cam_no])

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
                        cam_id=cam_id,
                        log_message_queue=self.log_message_queue,
                        max_num_points=options.num_points,
                        roi2_radius=options.software_roi_radius,
                        bg_frame_interval=options.background_frame_interval,
                        bg_frame_alpha=options.background_frame_alpha,
                        cam_no=cam_no,
                        main_brain_hostname=self.main_brain_hostname,
                        mask_image=mask,
                        diff_threshold_shared=diff_threshold_shared,
                        clear_threshold_shared=clear_threshold_shared,
                        n_sigma_shared=n_sigma_shared,
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
                    print 'not starting full .fmf thread'

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
                    print 'not starting small .fmf thread'

            else:
                cam_processor_chain = None

            self._image_sources[cam_no].set_chain( cam_processor_chain )

            ##################################################################
            #
            # Misc
            #
            ##################################################################

            self.reconstruct_helper.append( None )

            if cam_iface is not None:
                driver_string = 'using cam_iface driver: %s (wrapper: %s)'%(
                    cam_iface.get_driver_name(),
                    cam_iface.get_wrapper_name())
                self.log_message_queue.put((cam_id,time.time(),driver_string))

        self.last_frames_by_cam = [ [] for c in range(num_cams) ]
        self.last_points_by_cam = [ [] for c in range(num_cams) ]
        self.last_points_framenumbers_by_cam = [ [] for c in range(num_cams) ]
        self.n_raw_frames = [0 for i in range(num_cams)]
        self.last_measurement_time = [time_func() for i in range(num_cams)]
        self.last_return_info_check = [ 0.0 for i in range(num_cams)]

    def get_image_sources(self):
        return self._image_sources

    def get_image_controllers(self):
        return self._image_controllers

    def quit_function(self,exit_value):
        for globals in self.globals:
            globals['cam_quit_event'].set()

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
            globals = self.globals[cam_no]
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
            # handle pyro function calls
            for cam_no, cam_id in enumerate(self.all_cam_ids):
                if self.cam_status[cam_no] == 'destroyed':
                    # ignore commands for closed cameras
                    continue
                try:
                    cmds=self.main_brain.get_and_clear_commands(cam_id)
                except KeyError:
                    print 'main brain appears to have lost cam_id',cam_id
                except Exception, x:
                    print >> sys.stderr,'Remote traceback:','*'*30
                    print >> sys.stderr,''.join(Pyro.util.getPyroTraceback(x))
                    raise
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
                    print 'ERROR: thread %s died unexpectedly. Quitting'%(thread.getName())
                    self.quit_function(1)

            if not DISABLE_ALL_PROCESSING:
                for cam_no, cam_id in enumerate(self.all_cam_ids):
                    globals = self.globals[cam_no] # shorthand
                    last_frames = self.last_frames_by_cam[cam_no]
                    last_points = self.last_points_by_cam[cam_no]
                    last_points_framenumbers = self.last_points_framenumbers_by_cam[cam_no]

                    now = time_func() # roughly flydra_camera_node.py line 1504

                    # calculate and send FPS every 5 sec
                    elapsed = now-self.last_measurement_time[cam_no]
                    if elapsed > 5.0:
                        fps = self.n_raw_frames[cam_no]/elapsed
                        self.main_brain.set_fps(cam_id,fps)
                        self.last_measurement_time[cam_no] = now
                        self.n_raw_frames[cam_no] = 0

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

                            self.n_raw_frames[cam_no] += 1
                    except Queue.Empty:
                        pass

        except:
            import traceback
            traceback.print_exc()
            self.quit_function(1)

    def handle_commands(self, cam_no, cmds):
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
            DEBUG('  handle_commands: key',key)
            if key == 'set':
                for property_name,value in cmds['set'].iteritems():
                    print 'setting camera property', property_name, value,'...',
                    sys.stdout.flush()
                    if property_name in CAM_CONTROLS:
                        enum = CAM_CONTROLS[property_name]
                        if type(value) == tuple: # setting whole thing
                            props = cam.get_camera_property_info(enum)
                            #print 'props:',props
                            if value[1] != props['min_value']:
                                import warnings
                                warnings.warn('value[1] != props["min_value"] (%d != %d)'%(value[1], props['min_value']))
                            if value[2] != props['max_value']:
                                import warnings
                                warnings.warn('value[2] != props["max_value"] (%d != %d)'%(value[2], props['max_value']))
                            value = value[0]
                        cam.set_camera_property(enum,value,0)
                    elif property_name == 'roi':
                        print 'flydra_camera_node.py: ignoring ROI command for now...'
                        #cam_processor.roi = value
                    elif property_name == 'n_sigma':
                        print 'setting n_sigma',value
                        cam_processor.n_sigma_shared.set(value)
                    elif property_name == 'diff_threshold':
                        cam_processor.diff_threshold_shared.set(value)
                    elif property_name == 'clear_threshold':
                        cam_processor.clear_threshold_shared.set(value)
                    elif property_name == 'width':
                        assert cam.get_max_width() == value
                    elif property_name == 'height':
                        assert cam.get_max_height() == value
                    elif property_name == 'trigger_mode':
                        print 'cam.set_trigger_mode_number( value )',value
                        cam.set_trigger_mode_number( value )
                    elif property_name == 'cmp':
                        if value:
                            # print 'ignoring request to use_cmp'
                            globals['use_cmp'].set()
                        else: globals['use_cmp'].clear()
                    elif property_name == 'expected_trigger_framerate':
                        #print 'expecting trigger fps',value
                        cam_processor.shortest_IFI = 1.0/value # XXX TODO: FIXME: thread crossing bug
                    elif property_name == 'max_framerate':
                        if 0:
                            #print 'ignoring request to set max_framerate'
                            pass
                        else:
                            try:
                                cam.set_framerate(value)
                            except Exception,err:
                                print 'ERROR: failed setting framerate:',err
                    elif property_name == 'collecting_background':
                        if value: globals['collecting_background'].set()
                        else: globals['collecting_background'].clear()
                    elif property_name == 'visible_image_view':
                        globals['export_image_name'] = value
                        #print 'displaying',value,'image'
                    else:
                        print 'IGNORING property',property_name

                    print 'OK'


            elif key == 'get_im':
                val = globals['most_recent_frame_potentially_corrupt']
                if val is not None: # prevent race condition
                    lb, im = val
                    #nxim = nx.array(im) # copy to native nx form, not view of __array_struct__ form
                    nxim = nx.asarray(im) # view of __array_struct__ form
                    self.main_brain.set_image(cam_id, (lb, nxim))


            elif key == 'request_missing':
                camn_and_list = map(int,cmds[key].split())
                camn, framenumber_offset = camn_and_list[:2]
                missing_framenumbers = camn_and_list[2:]
                print 'I know main brain wants %d frames (camn %d) at %s:'%(
                    len(missing_framenumbers),
                    camn,time.asctime()),
                if len(missing_framenumbers) > 200:
                    print str(missing_framenumbers[:25]) + ' + ... + ' + str(missing_framenumbers[-25:])
                else:
                    print str(missing_framenumbers)

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
                    print '  Unable to find %d frames (camn %d):'%(
                        len(still_missing),
                        camn),
                    if len(still_missing) > 200:
                        print str(still_missing[:25]) + ' + ... + ' + str(still_missing[-25:])
                    else:
                        print str(still_missing)

            elif key == 'quit':
                self._image_sources[cam_no].join(0.1)
                # XXX TODO: quit and join chain threads
                cam.close()
                self.cam_status[cam_no] = 'destroyed'
                cmds=self.main_brain.close(cam_id)
            elif key == 'take_bg':
                globals['take_background'].set()
            elif key == 'clear_bg':
                globals['clear_background'].set()

            elif key == 'start_recording':
                if saver is None:
                    print 'no save thread -- cannot save movies'
                    continue
                raw_file_basename = cmds[key]

                raw_file_basename = os.path.expanduser(raw_file_basename)

                save_dir = os.path.split(raw_file_basename)[0]
                if not os.path.exists(save_dir):
                    print 'making %s'%save_dir
                    os.makedirs(save_dir)

                saver.start_recording(
                    raw_file_basename = raw_file_basename)

            elif key == 'stop_recording':
                saver.stop_recording()

            elif key == 'start_small_recording':
                if small_saver is None:
                    print 'no small save thread -- cannot save small movies'
                    continue

                small_filebasename = cmds[key]
                small_saver.start_recording(small_filebasename=small_filebasename)
            elif key == 'stop_small_recording':
                small_saver.stop_recording()
            elif key == 'cal':
                print 'setting calibration'
                pmat, intlin, intnonlin, scale_factor = cmds[key]

                # XXX TODO: FIXME: thread crossing bug
                # these three should always be done together in this order:
                cam_processor.set_scale_factor( scale_factor )
                cam_processor.set_pmat( pmat )
                cam_processor.make_reconstruct_helper(intlin, intnonlin) # let grab thread make one

                ######
                fc1 = intlin[0,0]
                fc2 = intlin[1,1]
                cc1 = intlin[0,2]
                cc2 = intlin[1,2]
                k1, k2, p1, p2 = intnonlin

                # we make one, too
                self.reconstruct_helper[cam_no] = reconstruct_utils.ReconstructHelper(
                    fc1, fc2, cc1, cc2, k1, k2, p1, p2 )
            else:
                raise ValueError('unknown key "%s"'%key)

def get_app_defaults():

    # where is the "main brain" server?
    try:
        default_main_brain_hostname = socket.gethostbyname('brain1')
    except:
        # try localhost
        try:
            default_main_brain_hostname = socket.gethostbyname(socket.gethostname())
        except: #socket.gaierror?
            default_main_brain_hostname = ''

    defaults = dict(wrapper='ctypes',
                    backend='unity',

                    # these are the most important 2D tracking parameters:
                    diff_threshold = 5,
                    n_sigma=7.0,

                    clear_threshold = 0.3,

                    debug_drop=False,
                    wx=False,
                    sdl=False,
                    debug_acquire=False,
                    disable_ifi_warning=False,
                    num_points=20,
                    software_roi_radius=10,
                    num_buffers=50,
                    small_save_radius=10,
                    background_frame_interval=50,
                    background_frame_alpha=1.0/50.0,
                    server = default_main_brain_hostname,
                    mask_images = None,
                    )
    return defaults

def main():
    parse_args_and_run()

def benchmark():
    parse_args_and_run(benchmark=True)

def parse_args_and_run(benchmark=False):
    usage_lines = ['%prog [options]',
                   '',
                   '  available wrappers and backends:']

    for wrapper,backends in cam_iface_choose.wrappers_and_backends.iteritems():
        for backend in backends:
            usage_lines.append('    --wrapper %s --backend %s'%(wrapper,backend))
    del wrapper, backend # delete temporary variables
    usage = '\n'.join(usage_lines)

    parser = OptionParser(usage=usage,
                          version="%prog 0.1")

    defaults = get_app_defaults()
    parser.set_defaults(**defaults)

    parser.add_option("--server", dest="server", type='string',
                      help="hostname of mainbrain SERVER",
                      metavar="SERVER [default: %default]")

    parser.add_option("--wrapper", type='string',
                      help="cam_iface WRAPPER to use [default: %default]",
                      metavar="WRAPPER")

    parser.add_option("--backend", type='string',
                      help="cam_iface BACKEND to use [default: %default]",
                      metavar="BACKEND")

    parser.add_option("--n-sigma", type='float',
                      help=("criterion used to determine if a pixel is significantly "
                            "different than the mean [default: %default]"))

    parser.add_option("--debug-drop", action='store_true',
                      help="save debugging information regarding dropped network packets")

    parser.add_option("--debug-std", action='store_true',
                      help="show mean pixel STD every 200 frames")

    parser.add_option("--sdl", action='store_true',
                      help="SDL-based display of raw images")

    parser.add_option("--wx", action='store_true',
                      help="wx-based GUI to display raw images")

    parser.add_option("--wx-full", action='store_true',
                      help="wx-based GUI to display raw and processed images")

    parser.add_option("--debug-acquire", action='store_true',
                      help="print to the console information on each frame")

    parser.add_option("--disable-ifi-warning", action='store_true',
                      help=("do not print a warning if the inter-frame-interval "
                            "(IFI) is longer than expected"))

    parser.add_option("--num-points", type="int",
                      help="number of points to track per cameras [default: %default]")

    parser.add_option("--software-roi-radius", type="int",
                      help="radius of software region of interest [default: %default]")

    parser.add_option("--background-frame-interval", type="int",
                      help="every N frames, add a new BG image to the accumulator [default: %default]")

    parser.add_option("--background-frame-alpha", type="float",
                      help="weight for each BG frame added to accumulator [default: %default]")

    parser.add_option("--mode-num", type="int",
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

    parser.add_option("--small-save-radius", type="int",
                      help='half the edge length of .ufmf movies [default: %default]')

    (options, args) = parser.parse_args()
    #print dir(options)

    if not options.wrapper:
        print 'WRAPPER must be set'
        parser.print_help()
        return

    if not options.backend:
        print 'BACKEND must be set'
        parser.print_help()
        return

    app_state=AppState(options = options,
                       benchmark=benchmark,
                       )

    if options.wx or options.wx_full:
        assert options.sdl == False, 'cannot have wx and sdl simultaneously enabled!'
        full = bool(options.wx_full)
        import camnodewx
        app=camnodewx.WxApp()
        if not DISABLE_ALL_PROCESSING:
            app_state.append_chain( klass = camnodewx.DisplayCamData, args=(app,),
                                    kwargs = dict(full=full),
                                    basename = 'camnodewx.DisplayCamData' )
        app.post_init(call_often = app_state.main_thread_task,full=full)
        app_state.set_quit_function( app.OnQuit )
    elif options.sdl:
        import camnodesdl
        app=camnodesdl.SdlApp(
                              call_often = app_state.main_thread_task)
        if not DISABLE_ALL_PROCESSING:
            app_state.append_chain( klass = camnodesdl.DisplayCamData, args=(app,),
                                    basename = 'camnodesdl.DisplayCamData' )
        app_state.set_quit_function( app.OnQuit )
    else:
        app=ConsoleApp(call_often = app_state.main_thread_task)
        app_state.set_quit_function( app.OnQuit )

    for (model, controller) in zip(app_state.get_image_sources(),
                                   app_state.get_image_controllers()):
        app.generate_view( model, controller )
    app.MainLoop()

if __name__=='__main__':
    main()

