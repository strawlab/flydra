#!/usr/bin/env python
import thread
import time
import socket
import sys
import numarray as na
import Pyro.core, Pyro.errors

DUMMY=1

if not DUMMY:
    if sys.platform == 'win32':
        import cam_iface_bcam
        cam_iface = cam_iface_bcam
    elif sys.platform.startswith('linux'):
        import cam_iface_dc1394
        cam_iface = cam_iface_dc1394
    else:
        raise NotImplementedError('only win32 and linux support implemented')
else:
    import cam_iface_dummy
    cam_iface = cam_iface_dummy
    
CAM_CONTROLS = {'shutter':cam_iface.SHUTTER,
                'gain':cam_iface.GAIN,
                'brightness':cam_iface.BRIGHTNESS}

incoming_frames = []

def grab_thread(cam,quit_now,thread_done):
    # transfer data from camera
    # (this could be in C)
    global incoming_frames
    thread_done.acquire()
    buf = na.zeros( (cam.max_height,cam.max_width), na.UInt8 ) # allocate buffer
    while quit_now.locked():
        cam.grab_next_frame_blocking(buf) # grab frame and stick in buf
        incoming_frames.append( buf.copy() ) # save a copy of the buffer
    thread_done.release()

def main():
    global incoming_frames

    start = time.time()
    now = start
    num_buffers = 3
    
    grabbed_frames = []

    # open network stuff
    Pyro.core.initClient(banner=0)
    # where is the server
    hostname = socket.gethostbyname(socket.gethostname())
    port = 9832
    name = 'flydra_brain'
    flydra_brain_URI = "PYROLOC://%s:%d/%s" % (hostname,port,name)
    flydra_brain = Pyro.core.getProxyForURI(flydra_brain_URI)

    for device_number in range(cam_iface.get_num_cameras()):
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
    driver = cam_iface.get_driver_abbreviation()
    cam_id = '%(hostname)s %(driver)s'%locals()
    #cam_id = '%(hostname)s %(driver)s %(device_number)d %(start)f'%locals()
    # inform brain that we're connected before starting camera thread
    servlet_URI = flydra_brain.get_URI_for_new_camera_servlet()
    camera_servlet = Pyro.core.getProxyForURI(servlet_URI)
    camera_servlet._setOneway(['push_image','set_current_fps'])
    
    camera_servlet.set_cam_info(cam_id,scalar_control_info)

    # start camera thread
    thread_done = thread.allocate_lock()
    quit_now = thread.allocate_lock()
    quit_now.acquire()
    thread.start_new_thread(grab_thread,(cam,quit_now,thread_done))

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
                    camera_servlet.set_current_fps(fps)
                    last_measurement_time = now
                    n_frames = 0

                # get new frames from grab thread
                if len(incoming_frames):
                    n_frames += len(incoming_frames)
                    grabbed_frames.extend( incoming_frames )
                    incoming_frames = []

                # send image
                if len(grabbed_frames):
                    camera_servlet.push_image(grabbed_frames[-1])
                    grabbed_frames = []

                # poll for commands
                if now - last_return_info_check > 1.0:
                    updates = camera_servlet.get_updates()
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

                time.sleep(0.05)

        finally:
            quit_now.release()
            camera_servlet.close()
            del camera_servlet
            flydra_brain.delete_servlet_by_URI(servlet_URI)
            thread_done.acquire() # block until thread is done...
    except Pyro.errors.ConnectionClosedError:
        pass
    
if __name__=='__main__':
    main()
