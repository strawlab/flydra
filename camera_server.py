#!/usr/bin/env python

DUMMY=0

import threading
import time
import socket
import sys
import numarray as na
import Pyro.core, Pyro.errors

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
most_recent_frame = None

def grab_func(cam,app_quit_event,grab_thread_done,incoming_frames_lock):
    # transfer data from camera
    # (this could be in C)
    global incoming_frames, most_recent_frame
    buf = na.zeros( (cam.max_height,cam.max_width), na.UInt8 ) # allocate buffer

    # speed up by eliminating namespace lookups
    app_quit_event_isSet = app_quit_event.isSet
    grab = cam.grab_next_frame_blocking
    acquire_lock = incoming_frames_lock.acquire
    copy_data = buf.copy
    release_lock = incoming_frames_lock.release
    sleep = time.sleep
    try:
        while not app_quit_event_isSet():
            grab(buf) # grab frame and stick in buf
            acquire_lock()
            most_recent_frame = copy_data() # copy buffer out of FIFO
            incoming_frames.append( most_recent_frame ) # save it
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
                  app_quit_event, listen_thread_done):
        self.cam_id = cam_id
        self.main_brain = main_brain
        self.app_quit_event = app_quit_event
        self.listen_thread_done = listen_thread_done
        self.cam = cam

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

    def prints(self,value):
        print value

    def quit(self):
        print 'received quit command'
        self.app_quit_event.set()

    def set_camera_property(self,property_name,value):
        enum = CAM_CONTROLS[property_name]
        self.cam.set_camera_property(enum,value,0,0)
        return True
           
def main():
    global incoming_frames

    # ----------------------------------------------------------------
    #
    # Setup cameras
    #
    # ----------------------------------------------------------------

    num_buffers = 30
    for device_number in range(cam_iface.cam_iface_get_num_cameras()):
        try:
            cam = cam_iface.CamContext(device_number,num_buffers)
            break # found a camera
        except Exception, x:
            if not x.args[0].startswith('The requested resource is in use.'):
                raise

    cam.set_camera_property(cam_iface.SHUTTER,300,0,0)
    cam.set_camera_property(cam_iface.GAIN,72,0,0)
    cam.set_camera_property(cam_iface.BRIGHTNESS,783,0,0)
    
    # ----------------------------------------------------------------
    #
    # Initialize variables
    #
    # ----------------------------------------------------------------
 
    start = time.time()
    now = start
    
    grabbed_frames = []

    # control flow events for threading model
    app_quit_event = threading.Event()
    listen_thread_done = threading.Event()
    grab_thread_done = threading.Event()

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
        main_brain_hostname = socket.gethostbyname('flydra-server')
    except:
        # try localhost
        main_brain_hostname = socket.gethostbyname(socket.gethostname())
    port = 9833
    name = 'main_brain'
    
    main_brain_URI = "PYROLOC://%s:%d/%s" % (main_brain_hostname,port,name)
    print 'searching for',main_brain_URI
    main_brain = Pyro.core.getProxyForURI(main_brain_URI)
    print 'found'
    
    # inform brain that we're connected before starting camera thread
    scalar_control_info = {}
    for name, enum_val in CAM_CONTROLS.items():
        current_value = cam.get_camera_property(enum_val)[0]
        tmp = cam.get_camera_property_range(enum_val)
        min_value = tmp[1]
        max_value = tmp[2]
        scalar_control_info[name] = (current_value, min_value, max_value)

    driver = cam_iface.cam_iface_get_driver_name()

    cam_id = main_brain.register_new_camera(scalar_control_info)
    main_brain._setOneway(['set_image','set_fps','close'])
    
    # ---------------------------------------------------------------
    #
    # start local Pyro server
    #
    # ---------------------------------------------------------------

    port=9834
    daemon = Pyro.core.Daemon(host=hostname,port=port)
    from_main_brain_api = FromMainBrainAPI();
    from_main_brain_api.post_init(cam,cam_id,main_brain,
                                  app_quit_event,listen_thread_done)
    URI=daemon.connect(from_main_brain_api,'camera_server')
    print 'URI:',URI
    
    # create and start listen thread
    listen_thread=threading.Thread(target=from_main_brain_api.listen,
                                   args=(daemon,))
    listen_thread.start()
        
    # ----------------------------------------------------------------
    #
    # start camera thread
    #
    # ----------------------------------------------------------------
    
    incoming_frames_lock = threading.Lock()
    grab_thread=threading.Thread(target=grab_func,
                                 args=(cam,
                                       app_quit_event,
                                       grab_thread_done,
                                       incoming_frames_lock))
    cam.start_camera()  # start camera
    grab_thread.start() # start grabbing frames from camera

    last_measurement_time = time.time()
    last_return_info_check = 0.0 # never
    n_frames = 0
    quit = False
    try:
        try:
            while not app_quit_event.isSet():
                now = time.time()

                # calculate and send FPS
                elapsed = now-last_measurement_time
                if elapsed > 5.0:
                    fps = n_frames/elapsed
                    main_brain.set_fps(cam_id,fps)
                    last_measurement_time = now
                    n_frames = 0

                # get new frames from grab thread
                if len(incoming_frames):
                    n_frames += len(incoming_frames)
                    incoming_frames_lock.acquire()
                    grabbed_frames.extend( incoming_frames )
                    incoming_frames = []
                    incoming_frames_lock.release()

                # send most recent image
                if len(grabbed_frames):
                    #main_brain.set_image(cam_id,grabbed_frames[-1])
                    grabbed_frames = []

                cmds=main_brain.get_and_clear_commands(cam_id)
                for key in cmds.keys():
                    if key == 'set':
                        for param,value in cmds['set'].iteritems():
                            from_main_brain_api.set_camera_property(param,value)
                    elif key == 'get_im': # low priority get image (for streaming)
                        from_main_brain_api.send_most_recent_frame() # mimic call

                time.sleep(0.05)

        finally:
            app_quit_event.set() # make sure other threads close
            print 'telling main_brain to close cam_id'
            main_brain.close(cam_id)
            print 'closed'
            print
            print 'waiting for grab thread to quit'
            grab_thread_done.wait() # block until thread is done...
            print 'closed'
            print
            print 'waiting for camera_server to close'
            listen_thread_done.wait() # block until thread is done...
            print 'closed'
            print
            print 'quitting'
    except Pyro.errors.ConnectionClosedError:
        print 'unexpected connection closure...'
    
if __name__=='__main__':
    main()
