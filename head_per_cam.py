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

def grab_func(cam,quit_now,thread_done,incoming_frames_lock):
    # transfer data from camera
    # (this could be in C)
    global incoming_frames
    buf = na.zeros( (cam.max_height,cam.max_width), na.UInt8 ) # allocate buffer
    try:
        while not quit_now.isSet():
            cam.grab_next_frame_blocking(buf,0) # grab frame and stick in buf
            incoming_frames_lock.acquire()
            incoming_frames.append( buf.copy() ) # save a copy of the buffer
            incoming_frames_lock.release()
    ##        sys.stdout.write('_')
    ##        sys.stdout.flush()
            time.sleep(0.00001) # yield processor
    finally:
        thread_done.set()

def main():
    global incoming_frames

    start = time.time()
    now = start
    num_buffers = 30
    
    grabbed_frames = []

    # open network stuff
    Pyro.core.initClient(banner=0)
    # where is the server
    try:
        hostname = socket.gethostbyname('flydra-server')
    except:
        # try localhost
        hostname = socket.gethostbyname(socket.gethostname())
    port = 9833
    name = 'main_brain'
    
    main_brain_URI = "PYROLOC://%s:%d/%s" % (hostname,port,name)
    print 'searching for',main_brain_URI
    main_brain = Pyro.core.getProxyForURI(main_brain_URI)
    print 'found'

    for device_number in range(cam_iface.cam_iface_get_num_cameras()):
        try:
            cam = cam_iface.CamContext(device_number,num_buffers)
            break # found a camera
        except Exception, x:
            if not x.args[0].startswith('The requested resource is in use.'):
                raise

    cam.set_camera_property(cam_iface.GAIN,28,0,0)
    cam.set_camera_property(cam_iface.SHUTTER,498,0,0)
    cam.set_camera_property(cam_iface.BRIGHTNESS,717,0,0)
    
    cam.start_camera()

    # build scalar_control_info dict
    scalar_control_info = {}
    for name, enum_val in CAM_CONTROLS.items():
        current_value = cam.get_camera_property(enum_val)[0]
        tmp = cam.get_camera_property_range(enum_val)
        min_value = tmp[1]
        max_value = tmp[2]
        scalar_control_info[name] = (current_value, min_value, max_value)

    hostname = socket.gethostbyname(socket.gethostname())
    driver = cam_iface.cam_iface_get_driver_name()

    # inform brain that we're connected before starting camera thread
    cam_id = main_brain.register_new_camera(scalar_control_info)
    main_brain._setOneway(['set_image','set_fps','close'])
    
    # start camera thread
    thread_done = threading.Event()
    quit_now = threading.Event()
    incoming_frames_lock = threading.Lock()
    grab_thread=threading.Thread(target=grab_func,
                                 args=(cam,quit_now,thread_done,incoming_frames_lock))
    grab_thread.start()

    last_measurement_time = time.time()
    last_return_info_check = 0.0 # never
    n_frames = 0
    quit = False
    try:
        try:
            while not quit:
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
##                    sys.stdout.write('y')
##                    sys.stdout.flush()

                # send most recent image
                if len(grabbed_frames):
                    main_brain.set_image(cam_id,grabbed_frames[-1])
                    grabbed_frames = []

                # poll for commands
                if now - last_return_info_check > 1.0:
                    updates = main_brain.get_commands(cam_id)
                    last_return_info_check = now

                    for key,value in updates:
                        if key in CAM_CONTROLS:
                            enum = CAM_CONTROLS[key]
                            cam.set_camera_property(enum,value,0,0)
                        # more commands here
                        elif key == 'quit':
                            quit = value
                        else:
                            raise RuntimeError ('Unknown command: %s'%repr(ud))
                time.sleep(0.01)
                if thread_done.isSet():
                    quit = True

        finally:
            print 'telling grab thread to quit'
            quit_now.set()
            print 'waiting for grab thread to quit'
            thread_done.wait() # block until thread is done...
            print 'telling main_brain to close cam_id'
            main_brain.close(cam_id)
            print 'quitting'
    except Pyro.errors.ConnectionClosedError:
        print 'unexpected connection closure...'
    
if __name__=='__main__':
    main()
