#emacs, this is -*-Python-*- mode

import threading
import time
import socket
import sys
import numarray as na
cimport c_numarray
import Pyro.core, Pyro.errors
import FlyMovieFormat
cimport c_cam_iface

# The Numeric API requires this function to be called before
# using any Numeric facilities in an extension module.
c_numarray.import_libnumarray()

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

CAM_CONTROLS = {'shutter':c_cam_iface.SHUTTER,
                'gain':c_cam_iface.GAIN,
                'brightness':c_cam_iface.BRIGHTNESS}

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class CamIFaceError(Exception):
    pass

def _check_error():
    if c_cam_iface.cam_iface_have_error():
        err_str=c_cam_iface.cam_iface_get_error_string()
        c_cam_iface.cam_iface_clear_error()
        raise CamIFaceError(err_str)

ctypedef class Camera:
    cdef c_cam_iface.CamContext* cval

    def __init__(self,int device_number, int num_buffers):
        self.cval = c_cam_iface.new_CamContext(device_number,num_buffers)
        _check_error()
        
    def __del__(self):
        c_cam_iface.delete_CamContext(self.cval)
        _check_error()
        
    def set_camera_property(self,
                            cameraProperty,
                            long ValueA,
                            long ValueB,
                            int Auto):
        c_cam_iface.CamContext_set_camera_property(self.cval,
                                                   cameraProperty,
                                                   ValueA,
                                                   ValueB,
                                                   Auto )
        _check_error()

    def get_camera_property(self,cameraProperty):
        cdef long ValueA
        cdef long ValueB
        cdef int Auto
        c_cam_iface.CamContext_get_camera_property(self.cval,
                                                   cameraProperty,
                                                   &ValueA,
                                                   &ValueB,
                                                   &Auto )
        _check_error()
        return (ValueA, ValueB, Auto)

    def get_camera_property_range(self,
                                  cameraProperty):
        cdef int Present
        cdef long Min
        cdef long Max
        cdef long Default
        cdef int Auto
        cdef int Manual
        
        c_cam_iface.CamContext_get_camera_property_range(self.cval,
                                                         cameraProperty,
                                                         &Present,
                                                         &Min,
                                                         &Max,
                                                         &Default,
                                                         &Auto,
                                                         &Manual)
        _check_error()
        return (Present,Min,Max,Default,Auto,Manual)
    
    cdef grab_next_frame_blocking(self,unsigned char *out_bytes):
        c_cam_iface.CamContext_grab_next_frame_blocking(self.cval,
                                                        out_bytes)
        _check_error()
    
    cdef point_next_frame_blocking(self,unsigned char **buf_ptr):
        c_cam_iface.CamContext_point_next_frame_blocking(self.cval,
                                                         buf_ptr)
        _check_error()

    cdef unpoint_frame(self):
        c_cam_iface.CamContext_unpoint_frame(self.cval)
        _check_error()

    def start_camera(self):
        c_cam_iface.CamContext_start_camera(self.cval)
        _check_error()

    def get_max_height(self):
        return self.cval.max_height

    def get_max_width(self):
        return self.cval.max_width    

    def get_last_timestamp(self):
        cdef double timestamp
        c_cam_iface.CamContext_get_last_timestamp(self.cval,&timestamp)
        _check_error()
        return timestamp
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

cdef class GrabClass:
    cdef Camera cam

    cdef set_camera(self,Camera cam):
        self.cam = cam
        
    def grab_func(self,globals):
        cdef unsigned char* buf_ptr
        cdef c_cam_iface.CamContext* cc
        cdef c_numarray._numarray buf
        cdef long height
        cdef long width

        # speed up by eliminating namespace lookups
        app_quit_event_isSet = globals['app_quit_event'].isSet
        acquire_lock = globals['incoming_frames_lock'].acquire
        release_lock = globals['incoming_frames_lock'].release
        sleep = time.sleep
        collecting_background_frames = None
        bg_image = None
        collect_background_start_isSet = globals['collect_background_start'].isSet
        collect_background_start_clear = globals['collect_background_start'].clear
        height = self.cam.get_max_height()
        width = self.cam.get_max_width()
        try:
            while not app_quit_event_isSet():
                # get pointer to data from camera driver
                self.cam.point_next_frame_blocking(&buf_ptr)

                # allow use of this memory area as numarray
                buf = <c_numarray._numarray>c_numarray.NA_NewArray(
                    buf_ptr, c_numarray.tUInt8, 2,
                    height, width)

                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()

                # now
                pre_cpy=time_func()

                # copy the data out of camwire's buffer
                my_buffer = buf.copy()

                # XXX need to Py_DECREF(buf) ??

                # return camwire's buffer
                self.cam.unpoint_frame()

                # make appropriate references to our copy of the data
                globals['most_recent_frame'] = my_buffer
                acquire_lock()
                globals['incoming_frames'].append(
                    (my_buffer,timestamp) ) # save it
                release_lock()

                # now
                post_cpy=time_func()

                if collect_background_start_isSet():
                    print 'started collecting background'
                    collecting_background_frames = 0
                    collect_background_start_clear()

                if collecting_background_frames is not None:
                    if collecting_background_frames >= 100: # average 100 frames
                        collecting_background_frames = None
                        print 'stopped collecting background'
                    else:
                        collecting_background_frames = collecting_background_frames+1

                sleep(0.00001) # yield processor
        finally:
            globals['app_quit_event'].set()
            globals['grab_thread_done'].set()

class FromMainBrainAPI( Pyro.core.ObjBase ):
    
    # ----------------------------------------------------------------
    #
    # Methods called locally
    #
    # ----------------------------------------------------------------
    
    def post_init(self, cam, cam_id, main_brain,globals):
        self.cam_id = cam_id
        self.main_brain = main_brain
        self.cam = cam
        self.globals = globals

    def listen(self,daemon):
        """thread mainloop"""
        self_app_quit_event_isSet = self.globals['app_quit_event'].isSet
        hr = daemon.handleRequests
        try:
            while not self_app_quit_event_isSet():
                hr(0.1) # block on select for n seconds
                
        finally:
            self.globals['app_quit_event'].set()
            self.globals['listen_thread_done'].set()

    # ----------------------------------------------------------------
    #
    # Methods called remotely from main_brain
    #
    # These all get called in their own thread.  Don't call across
    # the thread boundary without using locks.
    #
    # ----------------------------------------------------------------

    def send_most_recent_frame(self):
        self.main_brain.set_image(self.cam_id,self.globals['most_recent_frame'])

    def get_most_recent_frame(self):
        return self.globals['most_recent_frame']

    def start_recording(self,filename):
        fly_movie_lock = threading.Lock()
        self.globals['record_status_lock'].acquire()
        try:
            fly_movie = FlyMovieFormat.FlyMovieSaver(filename,version=1)
            self.globals['record_status'] = ('save',fly_movie,fly_movie_lock)
            print "starting to record to %s"%filename
        finally:
            self.globals['record_status_lock'].release()        

    def stop_recording(self):
        cmd=None
        self.globals['record_status_lock'].acquire()
        try:
            if self.globals['record_status']:
                cmd,fly_movie,fly_movie_lock = self.globals['record_status']
            self.globals['record_status'] = None
        finally:
            self.globals['record_status_lock'].release()
            
        if cmd == 'save':
            fly_movie_lock.acquire()
            fly_movie.close()
            fly_movie_lock.release()
            print "stopping recording"
        else:
            print "got stop recording command, but not recording!"

    def prints(self,value):
        print value

    def quit(self):
        print 'received quit command'
        self.globals['app_quit_event'].set()

    def collect_background(self):
        print 'received collect_background command'
        self.globals['collect_background_start'].set()

    def set_camera_property(self,property_name,value):
        enum = CAM_CONTROLS[property_name]
        self.cam.set_camera_property(enum,value,0,0)
        return True

cdef class App:
    cdef Camera cam
    cdef object globals
    cdef object main_brain
    cdef object cam_id
    cdef object from_main_brain_api
    
    def __init__(self):
        cdef GrabClass grabber
        
        # ----------------------------------------------------------------
        #
        # Initialize "global" variables
        #
        # ----------------------------------------------------------------

        self.globals = {}
        self.globals['incoming_frames']=[]
        self.globals['record_status']=None
        self.globals['most_recent_frame']=None
        
        # control flow events for threading model
        self.globals['app_quit_event'] = threading.Event()
        self.globals['listen_thread_done'] = threading.Event()
        self.globals['grab_thread_done'] = threading.Event()
        self.globals['incoming_frames_lock'] = threading.Lock()
        self.globals['collect_background_start'] = threading.Event()
        self.globals['record_status_lock'] = threading.Lock()

        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------

        assert c_cam_iface.cam_iface_get_num_cameras()==1
        self.cam = Camera(0,30)

        self.cam.set_camera_property(c_cam_iface.SHUTTER,300,0,0)
        self.cam.set_camera_property(c_cam_iface.GAIN,72,0,0)
        self.cam.set_camera_property(c_cam_iface.BRIGHTNESS,783,0,0)

        # ----------------------------------------------------------------
        #
        # Initialize network connections
        #
        # ----------------------------------------------------------------

        Pyro.core.initServer(banner=0,storageCheck=0)
        hostname = socket.gethostbyname(socket.gethostname())
        fqdn = socket.getfqdn(hostname)
        port = 9834

        # where is the "main brain" server?
        try:
            main_brain_hostname = socket.gethostbyname('mainbrain')
        except:
            # try localhost
            main_brain_hostname = socket.gethostbyname(socket.gethostname())
        port = 9833
        name = 'main_brain'

        main_brain_URI = "PYROLOC://%s:%d/%s" % (main_brain_hostname,port,name)
        print 'searching for',main_brain_URI
        self.main_brain = Pyro.core.getProxyForURI(main_brain_URI)
        print 'found'

        # inform brain that we're connected before starting camera thread
        scalar_control_info = {}
        for name, enum_val in CAM_CONTROLS.items():
            current_value = self.cam.get_camera_property(enum_val)[0]
            tmp = self.cam.get_camera_property_range(enum_val)
            min_value = tmp[1]
            max_value = tmp[2]
            scalar_control_info[name] = (current_value, min_value, max_value)

        driver = c_cam_iface.cam_iface_get_driver_name()

        self.cam_id = self.main_brain.register_new_camera(scalar_control_info)
        self.main_brain._setOneway(['set_image','set_fps','close'])

        # ---------------------------------------------------------------
        #
        # start local Pyro server
        #
        # ---------------------------------------------------------------

        port=9834
        daemon = Pyro.core.Daemon(host=hostname,port=port)
        self.from_main_brain_api = FromMainBrainAPI()
        self.from_main_brain_api.post_init(self.cam,self.cam_id,self.main_brain,
                                           self.globals)
        URI=daemon.connect(self.from_main_brain_api,'camera_server')
        print 'URI:',URI

        # create and start listen thread
        listen_thread=threading.Thread(target=self.from_main_brain_api.listen,
                                       args=(daemon,))
        listen_thread.start()

        # ----------------------------------------------------------------
        #
        # start camera thread
        #
        # ----------------------------------------------------------------

        grabber = GrabClass()
        grabber.set_camera(self.cam)
        grab_thread=threading.Thread(target=grabber.grab_func,
                                     args=(self.globals,))
        self.cam.start_camera()  # start camera
        grab_thread.start() # start grabbing frames from camera

    def mainloop(self):
        grabbed_frames = []

        last_measurement_time = time.time()
        last_return_info_check = 0.0 # never
        n_frames = 0
        try:
            try:
                while not self.globals['app_quit_event'].isSet():
                    now = time.time()

                    # calculate and send FPS
                    elapsed = now-last_measurement_time
                    if elapsed > 5.0:
                        fps = n_frames/elapsed
                        self.main_brain.set_fps(self.cam_id,fps)
                        last_measurement_time = now
                        n_frames = 0

                    # get new frames from grab thread
                    self.globals['incoming_frames_lock'].acquire()
                    len_if = len(self.globals['incoming_frames'])
                    if len_if:
                        n_frames = n_frames+len_if
                        grabbed_frames.extend( self.globals['incoming_frames'] )
                        self.globals['incoming_frames'] = []
                    self.globals['incoming_frames_lock'].release()

                    # process asynchronous commands
                    cmds=self.main_brain.get_and_clear_commands(self.cam_id)
                    for key in cmds.keys():
                        if key == 'set':
                            for param,value in cmds['set'].iteritems():
                                self.from_main_brain_api.set_camera_property(param,value)
                        elif key == 'get_im': # low priority get image (for streaming)
                            self.from_main_brain_api.send_most_recent_frame() # mimic call

                    # handle saving movie if needed
                    cmd=None
                    self.globals['record_status_lock'].acquire()
                    try:
                        if self.globals['record_status']:
                            cmd,fly_movie,fly_movie_lock = self.globals['record_status']
                    finally:
                        self.globals['record_status_lock'].release()

                    if len(grabbed_frames):
                        if cmd=='save':
                            fly_movie_lock.acquire()
                            try:
                                for frame,timestamp in grabbed_frames:
                                    fly_movie.add_frame(frame,timestamp)
                            finally:
                                fly_movie_lock.release()

                        grabbed_frames = []

                    time.sleep(0.05)

            finally:
                self.globals['app_quit_event'].set() # make sure other threads close
                print 'telling main_brain to close cam_id'
                self.main_brain.close(self.cam_id)
                print 'closed'
                print
                print 'waiting for grab thread to quit'
                self.globals['grab_thread_done'].wait() # block until thread is done...
                print 'closed'
                print
                print 'waiting for camera_server to close'
                self.globals['listen_thread_done'].wait() # block until thread is done...
                print 'closed'
                print
                print 'quitting'
        except Pyro.errors.ConnectionClosedError:
            print 'unexpected connection closure...'

def main():
    c_cam_iface.cam_iface_startup()
    try:
        app=App()
        app.mainloop()
    finally:
        c_cam_iface.cam_iface_shutdown()
        
