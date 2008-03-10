#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import pkg_resources
import os
BENCHMARK = int(os.environ.get('FLYDRA_BENCHMARK',0))
FLYDRA_BT = int(os.environ.get('FLYDRA_BT',0)) # threaded benchmark
near_inf = 9.999999e20

bright_non_gaussian_cutoff = 255
bright_non_gaussian_replacement = 25

import threading, time, socket, sys, struct, select, math, warnings
import Queue
import numpy
import numpy as nx
import errno
import scipy.misc.pilutil
import numpy.dual

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
    import Pyro.core, Pyro.errors
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
import camnode_utils
import motmot.FastImage.FastImage as FastImage
#FastImage.set_debug(3)
if os.name == 'posix' and sys.platform != 'darwin':
    import posix_sched

class DummyMainBrain:
    def __init__(self,*args,**kw):
        self.set_image = self.noop
        self.set_fps = self.noop
        self.log_message = self.noop
        self.close = self.noop
    def noop(self,*args,**kw):
        return
    def get_cam2mainbrain_port(self,*args,**kw):
        return 12345
    def register_new_camera(self,*args,**kw):
        return 'camdummy_0'
    def get_and_clear_commands(self,*args,**kw):
        return {}

class DummySocket:
    def __init__(self,*args,**kw):
        self.connect = self.noop
        self.send = self.noop
        self.sendto = self.noop
    def noop(self,*args,**kw):
        return

import flydra.common_variables
NETWORK_PROTOCOL = flydra.common_variables.NETWORK_PROTOCOL

import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time

pt_fmt = '<dddddddddBBddBdddddd'
small_datafile_fmt = '<dII'

# where is the "main brain" server?
try:
    default_main_brain_hostname = socket.gethostbyname('brain1')
except:
    # try localhost
    try:
        default_main_brain_hostname = socket.gethostbyname(socket.gethostname())
    except: #socket.gaierror?
        default_main_brain_hostname = ''

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
        self._buffers_handed_out = 0
        #   end: vars access controlled by self._lock

        self.set_size(size)

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
            return buffer

    def return_buffer(self,buffer):
        assert isinstance(buffer, PreallocatedBuffer)
        with self._lock:
            self._buffers_handed_out -= 1
            if buffer.get_size() == self._size:
                self._allocated_pool.append( buffer )

class ProcessCamData(object):
    def __init__(self):
        self._chain = camnode_utils.ChainLink()
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            buf= self._chain.get_buf()
            stdout_write('P')
            buf.processed_points = [ (10,20) ]
            self._chain.end_buf(buf)

class SaveCamData(object):
    def __init__(self):
        self._chain = camnode_utils.ChainLink()
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            buf= self._chain.get_buf()
            stdout_write('S')
            self._chain.end_buf(buf)

class IsoThread(threading.Thread):
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

        threading.Thread.__init__(self)
        self.chain = chain
        self.cam = cam
        self.buffer_pool = buffer_pool
        self.debug_acquire = debug_acquire
        self.cam_no_str = str(cam_no)
        self.quit_event = quit_event

    def run(self):
        buffer_pool = self.buffer_pool
        chain = self.chain
        cam_quit_event_isSet = self.quit_event.isSet
        cam = self.cam
        DEBUG_ACQUIRE = self.debug_acquire
        while not cam_quit_event_isSet():
            buf = buffer_pool.get_free_buffer()
            _bufim = buf.get_buf()
            try:
                cam.grab_next_frame_into_buf_blocking(_bufim)
            except cam_iface.BuffersOverflowed:
                if DEBUG_ACQUIRE:
                    stdout_write('(O%s)'%self.cam_no_str)
                now = time.time()
                msg = 'ERROR: buffers overflowed on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                self.log_message_queue.put((self.cam_id,now,msg))
                print >> sys.stderr, msg
                continue
            except cam_iface.FrameDataMissing:
                if DEBUG_ACQUIRE:
                    stdout_write('(M%s)'%self.cam_no_str)
                now = time.time()
                msg = 'Warning: frame data missing on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                #self.log_message_queue.put((self.cam_id,now,msg))
                print >> sys.stderr, msg
                continue
            except cam_iface.FrameSystemCallInterruption:
                if DEBUG_ACQUIRE:
                    stdout_write('(S%s)'%self.cam_no_str)
                continue

            if DEBUG_ACQUIRE:
                stdout_write(self.cam_no_str)

            # Now we get rid of the frame from this thread by passing
            # it to processing threads. The last one of these will
            # return the buffer to buffer_pool when done.

            chain.fire( buf )
        print 'exiting camera IsoThread for camera',self.cam_no_str

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

class AppState(object):
    """This class handles all camera states, properties, etc."""
    def __init__(self,
                 max_num_points_per_camera=2,
                 roi2_radius=10,
                 bg_frame_interval=50,
                 bg_frame_alpha=0.001,
                 main_brain_hostname = None,
                 emulation_reconstructor = None,
                 use_mode=None,
                 debug_drop = False, # debug dropped network packets
                 debug_acquire = False,
                 num_buffers=None,
                 mask_images = None,
                 n_sigma = None
                 ):

        self.quit_function = None

        if main_brain_hostname is None:
            self.main_brain_hostname = default_main_brain_hostname
        else:
            self.main_brain_hostname = main_brain_hostname


        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------


        num_cams = cam_iface.get_num_cameras()
        if num_cams == 0:
            return

        self.all_cams = []
        self.cam_status = []
        self.all_cam_chains = []
        self.all_grabbers = []
        self.globals = []
        self.all_cam_ids = []

        self.reconstruct_helper = []
        self.iso_threads = []

        for cam_no in range(num_cams):

            # ----------------------------------------------------------------
            #
            # Initialize "global" variables
            #
            # ----------------------------------------------------------------

            self.globals.append({})
            globals = self.globals[cam_no] # shorthand

            globals['debug_drop']=debug_drop
            globals['debug_acquire']=debug_acquire
            globals['incoming_raw_frames']=Queue.Queue()
            globals['incoming_bg_frames']=Queue.Queue()
            globals['raw_fmf_and_bg_fmf']=None
            globals['small_fmf']=None
            globals['most_recent_frame_potentially_corrupt']=None
            globals['saved_bg_frame']=False
            globals['current_bg_frame_and_timestamp']=None

            # control flow events for threading model
            globals['cam_quit_event'] = threading.Event()
            globals['listen_thread_done'] = threading.Event()
            globals['take_background'] = threading.Event()
            globals['clear_background'] = threading.Event()
            globals['collecting_background'] = threading.Event()
            globals['collecting_background'].set()
            globals['export_image_name'] = 'raw'
            globals['use_roi2'] = threading.Event()
            globals['use_cmp'] = threading.Event()
            #globals['use_cmp'].clear()
            #print 'not using ongoing variance estimate'
            globals['use_cmp'].set()


            backend = cam_iface.get_driver_name()
            if num_buffers is None:
                if backend.startswith('prosilica_gige'):
                    num_buffers = 50
                else:
                    num_buffers = 205
            N_modes = cam_iface.get_num_modes(cam_no)
            for i in range(N_modes):
                mode_string = cam_iface.get_mode_string(cam_no,i)
                print '  mode %d: %s'%(i,mode_string)
                if 'format7_0' in mode_string.lower():
                    # prefer format7_0
                    if use_mode is None:
                        use_mode = i
            if use_mode is None:
                use_mode = 0
            print 'attempting to initialize camera with %d buffers, mode "%s"'%(
                num_buffers,cam_iface.get_mode_string(cam_no,use_mode))
            cam = cam_iface.Camera(cam_no,num_buffers,use_mode)
            print 'allocated %d buffers'%num_buffers
            self.all_cams.append( cam )

            cam.start_camera()  # start camera
            self.cam_status.append( 'started' )

            # setup chain for this camera:
            process_cam = ProcessCamData()
            self.all_grabbers.append( process_cam )

            process_cam_chain = process_cam.get_chain()
            self.all_cam_chains.append(process_cam_chain)
            thread = threading.Thread( target = process_cam.mainloop )
            thread.setDaemon(True)
            thread.start()

            save_cam = SaveCamData()
            process_cam_chain.append_link( save_cam.get_chain() )
            thread = threading.Thread( target = save_cam.mainloop )
            thread.setDaemon(True)
            thread.start()

            buffer_pool = PreallocatedBufferPool(FastImage.Size(*cam.get_frame_size()))
            iso_thread = IsoThread(chain = process_cam_chain,
                                   cam = cam,
                                   buffer_pool = buffer_pool,
                                   debug_acquire = debug_acquire,
                                   cam_no = cam_no,
                                   quit_event = globals['cam_quit_event'],
                                   )
            iso_thread.setDaemon(True)
            iso_thread.start()
            self.iso_threads.append( iso_thread )

        # ----------------------------------------------------------------
        #
        # Initialize network connections
        #
        # ----------------------------------------------------------------
        if BENCHMARK:
            self.main_brain = DummyMainBrain()
        else:
            Pyro.core.initClient(banner=0)
            port = 9833
            name = 'main_brain'
            main_brain_URI = "PYROLOC://%s:%d/%s" % (self.main_brain_hostname,port,name)
            print 'connecting to',main_brain_URI
            self.main_brain = Pyro.core.getProxyForURI(main_brain_URI)
            self.main_brain._setOneway(['set_image','set_fps','close','log_message','receive_missing_data'])
        self.main_brain_lock = threading.Lock()
        #self.main_brain_lock = DebugLock('main_brain_lock',verbose=True)

        # ----------------------------------------------------------------
        #
        # Initialize more stuff
        #
        # ----------------------------------------------------------------

        if not BENCHMARK or not FLYDRA_BT:
            print 'starting TimestampEcho thread'
            # run in single-thread for benchmark
            timestamp_echo_thread=threading.Thread(target=TimestampEcho,
                                                   name='TimestampEcho')
            timestamp_echo_thread.setDaemon(True) # quit that thread if it's the only one left...
            timestamp_echo_thread.start()

        if mask_images is not None:
            mask_images = mask_images.split( os.pathsep )

        for cam_no in range(num_cams):
            cam = self.all_cams[cam_no]
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
                width,height = cam.get_frame_size()
                mask = numpy.zeros( (height,width), dtype=numpy.bool )
            # mask is currently an array of bool
            mask = mask.astype(numpy.uint8)*255


            # get settings
            scalar_control_info = {}

            globals['cam_controls'] = {}
            CAM_CONTROLS = globals['cam_controls']

            num_props = cam.get_num_camera_properties()
            for prop_num in range(num_props):
                props = cam.get_camera_property_info(prop_num)
                current_value,auto = cam.get_camera_property( prop_num )
                # set defaults
                if props['name'] == 'shutter':
                    new_value = 300
                elif props['name'] == 'gain':
                    new_value = 72
                elif props['name'] == 'brightness':
                    new_value = 783
                else:
                    print "WARNING: don't know default value for property %s, "\
                          "leaving as default"%(props['name'],)
                    new_value = current_value
                min_value = props['min_value']
                max_value = props['max_value']
                if props['has_manual_mode']:
                    if min_value <= new_value <= max_value:
                        try:
                            print 'setting camera property', props['name'], new_value
                            cam.set_camera_property( prop_num, new_value, 0 )
                        except:
                            print 'error while setting property %s to %d (from %d)'%(props['name'],new_value,current_value)
                            raise
                    else:
                        print 'not setting property %s to %d (from %d) because out of range (%d<=value<=%d)'%(props['name'],new_value,current_value,min_value,max_value)
                    current_value = new_value
                    CAM_CONTROLS[props['name']]=prop_num
                scalar_control_info[props['name']] = (current_value,
                                                      min_value, max_value)

            diff_threshold = 11
            scalar_control_info['diff_threshold'] = diff_threshold
            clear_threshold = 0.2
            scalar_control_info['clear_threshold'] = clear_threshold
            scalar_control_info['visible_image_view'] = 'raw'

            try:
                scalar_control_info['trigger_mode'] = cam.get_trigger_mode_number()
            except cam_iface.CamIFaceError:
                scalar_control_info['trigger_mode'] = 0
            scalar_control_info['roi2'] = globals['use_roi2'].isSet()
            scalar_control_info['cmp'] = globals['use_cmp'].isSet()

            scalar_control_info['width'] = width
            scalar_control_info['height'] = height
            scalar_control_info['roi'] = 0,0,width-1,height-1
            scalar_control_info['max_framerate'] = cam.get_framerate()
            scalar_control_info['collecting_background']=globals['collecting_background'].isSet()
            scalar_control_info['debug_drop']=globals['debug_drop']
            scalar_control_info['expected_trigger_framerate'] = 0.0

            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            with self.main_brain_lock:
                cam_id = self.main_brain.register_new_camera(cam_no,
                                                             scalar_control_info,
                                                             port)

            self.all_cam_ids.append(cam_id)
            cam2mainbrain_port = self.main_brain.get_cam2mainbrain_port(self.all_cam_ids[cam_no])

            # ----------------------------------------------------------------
            #
            # Misc
            #
            # ----------------------------------------------------------------

            self.reconstruct_helper.append( None )

            # ----------------------------------------------------------------
            #
            # start camera thread
            #
            # ----------------------------------------------------------------

            self.log_message_queue = Queue.Queue()
            driver_string = 'using cam_iface driver: %s (wrapper: %s)'%(
                cam_iface.get_driver_name(),
                cam_iface.get_wrapper_name())
            print >> sys.stderr, driver_string
            self.log_message_queue.put((cam_id,time.time(),driver_string))
            print 'max_num_points_per_camera',max_num_points_per_camera

    def set_quit_function(self, quit_function=None):
        self.quit_function = quit_function

    def append_chain(self, klass=None):
        for first_chain in self.all_cam_chains:
            thread_instance = klass()
            print 'thread_instance',thread_instance
            first_chain.append_link( thread_instance.get_chain() )
            thread = threading.Thread( target = thread_instance.mainloop )
            thread.setDaemon(True)
            thread.start()

    def update(self):
        try:
            for cam_no, cam_id in enumerate(self.all_cam_ids):
                if self.cam_status[cam_no] == 'destroyed':
                    # ignore commands for closed cameras
                    continue
                with self.main_brain_lock:
                    cmds=self.main_brain.get_and_clear_commands(cam_id)
                self.handle_commands(cam_no,cmds)

            # test if all closed
            all_closed = True
            for cam_no, cam_id in enumerate(self.all_cam_ids):
                if self.cam_status[cam_no] != 'destroyed':
                    all_closed = False
                    break
            if all_closed:
                if self.quit_function is None:
                    raise RuntimeError('all cameras closed, but no quit_function set')
                self.quit_function(0)
        except:
            import traceback
            traceback.print_exc()
            self.quit_function(1)

    def handle_commands(self, cam_no, cmds):
        if cmds:
            grabber = self.all_grabbers[cam_no]
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
                            assert value[1] == props['min_value']
                            assert value[2] == props['max_value']
                            value = value[0]
                        cam.set_camera_property(enum,value,0)
                    elif property_name == 'roi':
                        #print 'flydra_camera_node.py: ignoring ROI command for now...'
                        grabber.roi = value
                    elif property_name == 'diff_threshold':
                        #print 'setting diff_threshold',value
                        grabber.diff_threshold = value
                    elif property_name == 'clear_threshold':
                        grabber.clear_threshold = value
                    elif property_name == 'width':
                        assert cam.get_max_width() == value
                    elif property_name == 'height':
                        assert cam.get_max_height() == value
                    elif property_name == 'trigger_mode':
                        #print 'cam.set_trigger_mode_number( value )',value
                        cam.set_trigger_mode_number( value )
                    elif property_name == 'roi2':
                        if value: globals['use_roi2'].set()
                        else: globals['use_roi2'].clear()
                    elif property_name == 'cmp':
                        if value:
                            # print 'ignoring request to use_cmp'
                            globals['use_cmp'].set()
                        else: globals['use_cmp'].clear()
                    elif property_name == 'expected_trigger_framerate':
                        #print 'expecting trigger fps',value
                        self.shortest_IFI = 1.0/value
                    elif property_name == 'max_framerate':
                        if 1:
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

            elif key == 'quit':
                print 'quitting cam',cam_no
                globals['cam_quit_event'].set()
                print "globals['cam_quit_event'].isSet()",globals['cam_quit_event'].isSet()
                self.iso_threads[cam_no].join()
                print 'done with IsoThread %d - joined'%cam_no
                cam.close()
                print 'camera %d closed'%cam_no
                self.cam_status[cam_no] = 'destroyed'
                with self.main_brain_lock:
                    cmds=self.main_brain.close(cam_id)
            else:
                raise NotImplementedError('no support yet for key "%s"'%key)

def main():
    global cam_iface

    if BENCHMARK:
        cam_iface = cam_iface_choose.import_backend('dummy','dummy')
        print 'benchmark imported backend',cam_iface
        print '(from file %s)'%cam_iface.__file__
        max_num_points_per_camera=2

        app=App(max_num_points_per_camera,
                roi2_radius=10,
                bg_frame_interval=50,
                bg_frame_alpha=0.001,
                )
        app.mainloop()
        return

    usage_lines = ['%prog [options]',
                   '',
                   '  available wrappers and backends:']

    for wrapper,backends in cam_iface_choose.wrappers_and_backends.iteritems():
        for backend in backends:
            usage_lines.append('    --wrapper %s --backend %s'%(wrapper,backend))
    del wrapper, backend # delete temporary variables
    usage = '\n'.join(usage_lines)

    parser = OptionParser(usage)

    parser.add_option("--server", dest="server", type='string',
                      help="hostname of mainbrain SERVER",
                      metavar="SERVER")

    parser.add_option("--wrapper", type='string',
                      help="cam_iface WRAPPER to use",
                      default='ctypes',
                      metavar="WRAPPER")

    parser.add_option("--backend", type='string',
                      help="cam_iface BACKEND to use",
                      default='unity',
                      metavar="BACKEND")

    parser.add_option("--n-sigma", type='float',
                      default=2.0)

    parser.add_option("--debug-drop", action='store_true',
                      help="save debugging information regarding dropped network packets",
                      default=False)

    parser.add_option("--wx", action='store_true',
                      default=False)

    parser.add_option("--debug-acquire", action='store_true',
                      help="print to the console information on each frame",
                      default=False)

    parser.add_option("--num-points", type="int",
                      help="number of points to track per camera")

    parser.add_option("--emulation-cal", type="string",
                      help="name of calibration (directory or .h5 file); Run in emulation mode.")

    parser.add_option("--software-roi-radius", type="int",
                      help="radius of software region of interest")

    parser.add_option("--background-frame-interval", type="int",
                      help="every N frames, add a new BG image to the accumulator")

    parser.add_option("--background-frame-alpha", type="float",
                      help="weight for each BG frame added to accumulator")
    parser.add_option("--mode-num", type="int", default=None,
                      help="force a camera mode")
    parser.add_option("--num-buffers", type="int", default=None,
                      help="force number of buffers")

    parser.add_option("--mask-images", type="string",
                      help="list of masks for each camera (uses OS-specific path separator, ':' for POSIX, ';' for Windows)")

    (options, args) = parser.parse_args()

    emulation_cal=options.emulation_cal
    print 'emulation_cal',repr(emulation_cal)
    if emulation_cal is not None:
        emulation_cal = os.path.expanduser(emulation_cal)
        print 'emulation_cal',repr(emulation_cal)
        emulation_reconstructor = flydra.reconstruct.Reconstructor(
            emulation_cal)
    else:
        emulation_reconstructor = None

    if not emulation_reconstructor:
        if not options.wrapper:
            print 'WRAPPER must be set (except in benchmark or emulation mode)'
            parser.print_help()
            return

        if not options.backend:
            print 'BACKEND must be set (except in benchmark or emulation mode)'
            parser.print_help()
            return
        cam_iface = cam_iface_choose.import_backend( options.backend, options.wrapper )
    else:
        cam_iface = cam_iface_choose.import_backend('dummy','dummy')
        #cam_iface = cam_iface_choose.import_backend('blank','ctypes')
        cam_iface.set_num_cameras(len(emulation_reconstructor.get_cam_ids()))

    if options.num_points is not None:
        max_num_points_per_camera = options.num_points
    else:
        max_num_points_per_camera = 2

    if options.software_roi_radius is not None:
        roi2_radius = options.software_roi_radius
    else:
        roi2_radius = 10

    if options.background_frame_interval is not None:
        bg_frame_interval = options.background_frame_interval
    else:
        bg_frame_interval = 50

    if options.background_frame_alpha is not None:
        bg_frame_alpha = options.background_frame_alpha
    else:
        bg_frame_alpha = 0.001

    app_state=AppState(max_num_points_per_camera=max_num_points_per_camera,
                       roi2_radius=roi2_radius,
                       bg_frame_interval=bg_frame_interval,
                       bg_frame_alpha=bg_frame_alpha,
                       main_brain_hostname = options.server,
                       emulation_reconstructor = emulation_reconstructor,
                       debug_drop = options.debug_drop,
                       debug_acquire = options.debug_acquire,
                       use_mode = options.mode_num,
                       num_buffers = options.num_buffers,
                       mask_images = options.mask_images,
                       n_sigma = options.n_sigma,
                       )

    if options.wx:
        import camnodewx
        app=camnodewx.WxApp()
        app_state.append_chain( klass = camnodewx.DisplayCamData )
        app.post_init(call_often = app_state.update)
        app_state.set_quit_function( app.OnQuit )
    else:
        app=ConsoleApp(call_often = app_state.update)
        app_state.set_quit_function( app.OnQuit )
    app.MainLoop()

if __name__=='__main__':
    if 0:
        # profile

        # seems useless -- doesn't profile other threads?
        import hotshot
        prof = hotshot.Profile("profile.hotshot")
        res = prof.runcall(main)
        prof.close()
    else:
        # don't profile
        main()

