#!/usr/bin/env python

DUMMY=0

import threading
import time
import socket
import sys
import numarray as na
import Pyro.core, Pyro.errors
import FlyMovieFormat

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

if not DUMMY:
    import cam_iface
else:
    import cam_iface_dummy
    cam_iface = cam_iface_dummy

CAM_CONTROLS = {'shutter':cam_iface.SHUTTER,
            'gain':cam_iface.GAIN,
            'brightness':cam_iface.BRIGHTNESS}

incoming_frames = []
record_status = None
most_recent_frame = None

def grab_func(cam,app_quit_event,grab_thread_done,incoming_frames_lock):
    # transfer data from camera
    # (this could be in C)
    global incoming_frames, most_recent_frame
    buf = na.zeros( (cam.max_height,cam.max_width), na.UInt8 ) # allocate buffer

    # speed up by eliminating namespace lookups
    app_quit_event_isSet = app_quit_event.isSet
    grab = cam.grab_next_frame_blocking
    get_last_timestamp = cam.get_last_timestamp
    acquire_lock = incoming_frames_lock.acquire
    copy_data = buf.copy
    release_lock = incoming_frames_lock.release
    sleep = time.sleep
    try:
        while not app_quit_event_isSet():
            grab(buf) # grab frame and stick in buf
            now=get_last_timestamp()
            #now=time_func()
            acquire_lock()
            most_recent_frame = copy_data() # copy buffer out of FIFO
            incoming_frames.append( (most_recent_frame,now) ) # save it
            release_lock()
            sleep(0.00001) # yield processor
            
    finally:
        app_quit_event.set()
        grab_thread_done.set()

class FromMainBrainAPI( Pyro.core.ObjBase ):

    # ----------------------------------------------------------------
    #
    # Methods called locally
    #
    # ----------------------------------------------------------------
    
    def post_init(self, cam, cam_id, main_brain,
                  app_quit_event, listen_thread_done,
                  record_status_lock):
        self.cam_id = cam_id
        self.main_brain = main_brain
        self.app_quit_event = app_quit_event
        self.listen_thread_done = listen_thread_done
        self.cam = cam
        self.record_status_lock = record_status_lock

    def listen(self,daemon):
        """thread mainloop"""
        self_app_quit_event_isSet = self.app_quit_event.isSet
        hr = daemon.handleRequests
        try:
            while not self_app_quit_event_isSet():
                hr(0.1) # block on select for n seconds
                
        finally:
            self.app_quit_event.set()
            self.listen_thread_done.set()

    # ----------------------------------------------------------------
    #
    # Methods called remotely from main_brain
    #
    # These all get called in their own thread.  Don't call across
    # the thread boundary without using locks.
    #
    # ----------------------------------------------------------------

    def send_most_recent_frame(self):
        global most_recent_frame
        self.main_brain.set_image(self.cam_id,most_recent_frame)
##        most_recent_frame=None

    def get_most_recent_frame(self):
        global most_recent_frame
        return most_recent_frame

    def start_recording(self,filename):
        global record_status
        fly_movie_lock = threading.Lock()
        self.record_status_lock.acquire()
        try:
            fly_movie = FlyMovieFormat.FlyMovieSaver(filename,version=1)
            record_status = ('save',fly_movie,fly_movie_lock)
            print "starting to record to %s"%filename
        finally:
            self.record_status_lock.release()        

    def stop_recording(self):
        global record_status
        cmd=None
        self.record_status_lock.acquire()
        try:
            if record_status:
                cmd,fly_movie,fly_movie_lock = record_status
            record_status = None
        finally:
            self.record_status_lock.release()
            
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
        self.app_quit_event.set()

    def set_camera_property(self,property_name,value):
        enum = CAM_CONTROLS[property_name]
        self.cam.set_camera_property(enum,value,0,0)
        return True

class App:
    def __init__(self):
        global incoming_frames, record_status

        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------

        num_buffers = 30
        for device_number in range(cam_iface.cam_iface_get_num_cameras()):
            try:
                self.cam = cam_iface.CamContext(device_number,num_buffers)
                break # found a camera
            except Exception, x:
                if not x.args[0].startswith('The requested resource is in use.'):
                    raise

        self.cam.set_camera_property(cam_iface.SHUTTER,300,0,0)
        self.cam.set_camera_property(cam_iface.GAIN,72,0,0)
        self.cam.set_camera_property(cam_iface.BRIGHTNESS,783,0,0)

        # ----------------------------------------------------------------
        #
        # Initialize variables
        #
        # ----------------------------------------------------------------

        # control flow events for threading model
        self.app_quit_event = threading.Event()
        self.listen_thread_done = threading.Event()
        self.grab_thread_done = threading.Event()

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

        driver = cam_iface.cam_iface_get_driver_name()

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
        self.record_status_lock = threading.Lock()
        self.from_main_brain_api.post_init(self.cam,self.cam_id,self.main_brain,
                                           self.app_quit_event,self.listen_thread_done,
                                           self.record_status_lock)
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

        self.incoming_frames_lock = threading.Lock()
        grab_thread=threading.Thread(target=grab_func,
                                     args=(self.cam,
                                           self.app_quit_event,
                                           self.grab_thread_done,
                                           self.incoming_frames_lock))
        self.cam.start_camera()  # start camera
        grab_thread.start() # start grabbing frames from camera

    def mainloop(self):
        global incoming_frames, record_status

        grabbed_frames = []

        last_measurement_time = time.time()
        last_return_info_check = 0.0 # never
        n_frames = 0
        try:
            try:
                while not self.app_quit_event.isSet():
                    now = time.time()

                    # calculate and send FPS
                    elapsed = now-last_measurement_time
                    if elapsed > 5.0:
                        fps = n_frames/elapsed
                        self.main_brain.set_fps(self.cam_id,fps)
                        last_measurement_time = now
                        n_frames = 0

                    # get new frames from grab thread
                    self.incoming_frames_lock.acquire()
                    if len(incoming_frames):
                        n_frames += len(incoming_frames)
                        grabbed_frames.extend( incoming_frames )
                        incoming_frames = []
                    self.incoming_frames_lock.release()

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
                    self.record_status_lock.acquire()
                    try:
                        if record_status:
                            cmd,fly_movie,fly_movie_lock = record_status
                    finally:
                        self.record_status_lock.release()

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
                self.app_quit_event.set() # make sure other threads close
                print 'telling main_brain to close cam_id'
                self.main_brain.close(self.cam_id)
                print 'closed'
                print
                print 'waiting for grab thread to quit'
                self.grab_thread_done.wait() # block until thread is done...
                print 'closed'
                print
                print 'waiting for camera_server to close'
                self.listen_thread_done.wait() # block until thread is done...
                print 'closed'
                print
                print 'quitting'
        except Pyro.errors.ConnectionClosedError:
            print 'unexpected connection closure...'
    
if __name__=='__main__':
    app=App()
    app.mainloop()
