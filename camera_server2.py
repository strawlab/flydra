#emacs, this is -*-Python-*- mode
# $Id$

import threading
import time
import socket
import sys
import Pyro.core, Pyro.errors
import FlyMovieFormat
import struct
import numarray as nx
import pyx_cam_iface as cam_iface
try:
    import realtime_image_analysis
except ImportError, x:
    if str(x).startswith('libippcore.so: cannot open shared object file'):
        print 'WARNING: IPP not loaded, proceeding without it'
        import realtime_image_analysis_noipp as realtime_image_analysis
    else:
        raise x

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

CAM_CONTROLS = {'shutter':cam_iface.SHUTTER,
                'gain':cam_iface.GAIN,
                'brightness':cam_iface.BRIGHTNESS}

# where is the "main brain" server?
try:
    main_brain_hostname = socket.gethostbyname('mainbrain')
except:
    # try localhost
    main_brain_hostname = socket.gethostbyname(socket.gethostname())

class GrabClass(object):
    def __init__(self, cam, coord_port):
        self.cam = cam
        self.coord_port = coord_port

        # get coordinates for region of interest
        height = self.cam.get_max_height()
        width = self.cam.get_max_width()
        self.realtime_analyzer = realtime_image_analysis.RealtimeAnalyzer(width, height)

    def get_clear_threshold(self):
        return self.realtime_analyzer.clear_threshold
    def set_clear_threshold(self, value):
        self.realtime_analyzer.clear_threshold = value
    clear_threshold = property( get_clear_threshold, set_clear_threshold )
    
    def get_diff_threshold(self):
        return self.realtime_analyzer.diff_threshold
    def set_diff_threshold(self, value):
        self.realtime_analyzer.diff_threshold = value
    diff_threshold = property( get_diff_threshold, set_diff_threshold )

    def get_use_arena(self):
        return self.realtime_analyzer.use_arena
    def set_use_arena(self, value):
        self.realtime_analyzer.use_arena = value
    use_arena = property( get_use_arena, set_use_arena )

    def get_roi(self):
        return self.realtime_analyzer.roi
    def set_roi(self, lbrt):
        self.realtime_analyzer.roi = lbrt
    roi = property( get_roi, set_roi )
    
    def grab_func(self,globals):
        n_bg_samples = 100
        
        # questionable optimization: speed up by eliminating namespace lookups
        cam_quit_event_isSet = globals['cam_quit_event'].isSet
        acquire_lock = globals['incoming_frames_lock'].acquire
        release_lock = globals['incoming_frames_lock'].release
        sleep = time.sleep
        bg_frame_number = -1
        rot_frame_number = -1
        collect_background_start_isSet = globals['collect_background_start'].isSet
        collect_background_start_clear = globals['collect_background_start'].clear
        clear_background_start_isSet = globals['clear_background_start'].isSet
        clear_background_start_clear = globals['clear_background_start'].clear
        find_rotation_center_start_isSet = globals['find_rotation_center_start'].isSet
        find_rotation_center_start_clear = globals['find_rotation_center_start'].clear
        debug_isSet = globals['debug'].isSet
        height = self.cam.get_max_height()
        width = self.cam.get_max_width()
        buf_ptr_step = width
        flip = False

        buf = nx.zeros( (self.cam.max_height,self.cam.max_width), nx.UInt8 ) # allocate buffer
        coord_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        old_ts = time.time()
        try:
            while not cam_quit_event_isSet():
                self.cam.grab_next_frame_blocking(buf) # grab frame and stick in buf
                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()
                framenumber=self.cam.get_last_framenumber()

                diff = timestamp-old_ts
                if diff > 0.02:
                    print 'warning: IFI is',diff
                    flip = False # reset to synchronize across all cameras
                globals['last_frame_timestamp']=timestamp
                old_ts = timestamp
                
                points = self.realtime_analyzer.do_work( buf, timestamp, framenumber )
                
                if debug_isSet():
                    if flip:
                        show_image = self.realtime_analyzer.get_working_image()
                    else:
                        show_image = buf.copy()
                    flip = not flip
                else:
                    show_image = buf.copy()
                
                # make appropriate references to our copy of the data
                globals['most_recent_frame'] = show_image
                globals['most_recent_frame_and_points'] = show_image, points
                acquire_lock()
                globals['incoming_frames'].append(
                    (show_image,timestamp,framenumber) ) # save it
                release_lock()

                if clear_background_start_isSet():
                    clear_background_start_clear()
                    self.realtime_analyzer.clear_background_image()
                    
                if collect_background_start_isSet():
                    bg_frame_number=0
                    collect_background_start_clear()
                    self.realtime_analyzer.clear_accumulator_image()
                    
                if bg_frame_number>=0:
                    self.realtime_analyzer.accumulate_last_image()
                    bg_frame_number += 1
                    if bg_frame_number>=n_bg_samples:
                        bg_frame_number=-1 # stop averaging frames
                        self.realtime_analyzer.convert_accumulator_to_bg_image(n_bg_samples)
                                
                if find_rotation_center_start_isSet():
                    find_rotation_center_start_clear()
                    rot_frame_number=0
                    self.realtime_analyzer.rotation_calculation_init()
                    
                if rot_frame_number>=0:
                    self.realtime_analyzer.rotation_update()
                    rot_frame_number += 1
                    if rot_frame_number>=n_rot_samples:
                        self.realtime_analyzer.rotation_end()
                        rot_frame_number=-1 # stop averaging frames
              
                n_pts = len(points)
                data = struct.pack('<dli',timestamp,framenumber,n_pts)
                for i in range(n_pts):
                    data = data + struct.pack('ffff',*points[i])
                coord_socket.sendto(data,
                                    (main_brain_hostname,self.coord_port))
                sleep(1e-6) # yield processor
        finally:

            globals['cam_quit_event'].set()
            globals['grab_thread_done'].set()

class App:
    
    def __init__(self):

        MAX_GRABBERS = 3
        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------

        self.num_cams = cam_iface.get_num_cameras()
        print 'Number of cameras detected:', self.num_cams
        assert self.num_cams <= MAX_GRABBERS
        if self.num_cams == 0:
            return

        # ----------------------------------------------------------------
        #
        # Initialize network connections
        #
        # ----------------------------------------------------------------

        Pyro.core.initClient(banner=0)
        
        port = 9833
        name = 'main_brain'
        main_brain_URI = "PYROLOC://%s:%d/%s" % (main_brain_hostname,port,name)
        print 'connecting to',main_brain_URI
        self.main_brain = Pyro.core.getProxyForURI(main_brain_URI)
        self.main_brain._setOneway(['set_image','set_fps','close'])
        self.main_brain_lock = threading.Lock()

        # ----------------------------------------------------------------
        #
        # Initialize each camera
        #
        # ----------------------------------------------------------------

        self.globals = []
        self.all_cam_ids = []

        self.all_cams = []
        self.all_grabbers = []
        
        for cam_no in range(self.num_cams):
            cam = cam_iface.CamContext(cam_no,30)
            self.all_cams.append( cam )

            height = cam.get_max_height()
            width = cam.get_max_width()

            # ----------------------------------------------------------------
            #
            # Initialize "global" variables
            #
            # ----------------------------------------------------------------

            self.globals.append({})
            globals = self.globals[cam_no] # shorthand

            globals['incoming_frames']=[]
            globals['currently_saving_fly_movie']=None
            globals['most_recent_frame']=None
            globals['most_recent_frame_and_points']=None

            # control flow events for threading model
            globals['cam_quit_event'] = threading.Event()
            globals['listen_thread_done'] = threading.Event()
            globals['grab_thread_done'] = threading.Event()
            globals['incoming_frames_lock'] = threading.Lock()
            globals['collect_background_start'] = threading.Event()
            globals['clear_background_start'] = threading.Event()
            globals['find_rotation_center_start'] = threading.Event()
            globals['debug'] = threading.Event()

            globals['last_frame_timestamp']=None

            # set defaults
            cam.set_camera_property(cam_iface.SHUTTER,300,0,0)
            cam.set_camera_property(cam_iface.GAIN,72,0,0)
            cam.set_camera_property(cam_iface.BRIGHTNESS,783,0,0)

            # get settings
            scalar_control_info = {}
            for name, enum_val in CAM_CONTROLS.items():
                current_value = cam.get_camera_property(enum_val)[0]
                tmp = cam.get_camera_property_range(enum_val)
                min_value = tmp[1]
                max_value = tmp[2]
                scalar_control_info[name] = (current_value, min_value, max_value)
            diff_threshold = 8.1
            scalar_control_info['diff_threshold'] = diff_threshold
            clear_threshold = 0.0
            scalar_control_info['clear_threshold'] = clear_threshold
            
            scalar_control_info['width'] = width
            scalar_control_info['height'] = height
            scalar_control_info['roi'] = 0,0,width-1,height-1

            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            self.main_brain_lock.acquire()
            self.all_cam_ids.append(
                self.main_brain.register_new_camera(cam_no,
                                                    scalar_control_info,
                                                    port))
            coord_port = self.main_brain.get_coord_port(self.all_cam_ids[cam_no])
            self.main_brain_lock.release()
            
            # ---------------------------------------------------------------
            #
            # start local Pyro server
            #
            # ---------------------------------------------------------------

            hostname = socket.gethostname()
            if hostname == 'flygate':
                hostname = 'mainbrain' # serve on internal network
            print 'hostname',hostname
            host = socket.gethostbyname(hostname)
            daemon = Pyro.core.Daemon(host=host,port=port)

            # ----------------------------------------------------------------
            #
            # start camera thread
            #
            # ----------------------------------------------------------------

            grabber = GrabClass(cam,coord_port)
            self.all_grabbers.append( grabber )
            
            grabber.diff_threshold = diff_threshold
            grabber.clear_threshold = clear_threshold
            
            grabber.use_arena = False
            globals['use_arena'] = grabber.use_arena
            
            grab_thread=threading.Thread(target=grabber.grab_func,
                                         args=(globals,))
            cam.start_camera()  # start camera
            grab_thread.start() # start grabbing frames from camera

    def handle_commands(self, cam_no, cmds):
        cam = self.all_cams[cam_no]
        grabber = self.all_grabbers[cam_no]
        cam_id = self.all_cam_ids[cam_no]
        globals = self.globals[cam_no]
        
        for key in cmds.keys():
            if key == 'set':
                for property_name,value in cmds['set'].iteritems():
                    if property_name in CAM_CONTROLS:
                        enum = CAM_CONTROLS[property_name]
                        if type(value) == tuple: # setting whole thing
                            tmp = cam.get_camera_property_range(enum)
                            assert value[1] == tmp[1]
                            assert value[2] == tmp[2]
                            value = value[0]
                        cam.set_camera_property(enum,value,0,0)
                    elif property_name == 'roi':
                        grabber.roi = value 
                    elif property_name == 'diff_threshold':
                        grabber.diff_threshold = value
                    elif property_name == 'clear_threshold':
                        grabber.clear_threshold = value
                    elif property_name == 'width':
                        assert cam.get_max_width() == value
                    elif property_name == 'height':
                        assert cam.get_max_height() == value
            elif key == 'get_im':
                self.main_brain.set_image(cam_id, globals['most_recent_frame'])
            elif key == 'use_arena':
                grabber.use_arena = cmds[key]
                globals['use_arena'] = grabber.use_arena
            elif key == 'quit':
                globals['cam_quit_event'].set()
            elif key == 'collect_bg':
                globals['collect_background_start'].set()
            elif key == 'clear_bg':
                globals['clear_background_start'].set()
            elif key == 'find_r_center':
                globals['find_rotation_center_start'].set()
            elif key == 'stop_recording':
                if globals['currently_saving_fly_movie']:
                    fly_movie = globals['currently_saving_fly_movie']
                    fly_movie.close()
                    print 'stopped recording'
                globals['currently_saving_fly_movie'] = None
            elif key == 'start_recording':
                filename = cmds[key]
                fly_movie = FlyMovieFormat.FlyMovieSaver(filename,version=1)
                globals['currently_saving_fly_movie'] = fly_movie
                print "starting to record to %s"%filename
            elif key == 'debug':
                if cmds[key]: globals['debug'].set()
                else: globals['debug'].clear()
                
    def mainloop(self):
        # per camera variables
        grabbed_frames = []

        last_measurement_time = []
        last_return_info_check = []
        n_frames = []
        
        if self.num_cams == 0:
            return

        for cam_no in range(self.num_cams):
            grabbed_frames.append( [] )

            last_measurement_time.append( time_func() )
            last_return_info_check.append( 0.0 ) # never
            n_frames.append( 0 )
            
        try:
            try:
                cams_in_operation = self.num_cams
                while cams_in_operation>0:
                    cams_in_operation = 0
                    for cam_no in range(self.num_cams):
                        globals = self.globals[cam_no] # shorthand

                        # check if camera running
                        if globals['cam_quit_event'].isSet():
                            continue

                        cams_in_operation = cams_in_operation + 1

                        cam = self.all_cams[cam_no]
                        cam_id = self.all_cam_ids[cam_no]
                        
                        now = time_func()
                        lft = globals['last_frame_timestamp']
                        if lft is not None:
                            if (now-lft) > 1.0:
                                print 'WARNING: last frame was %f seconds ago'%(now-lft,)
                                print '(Is the grab thread dead?)'
                                globals['last_frame_timestamp'] = None

                        # calculate and send FPS
                        elapsed = now-last_measurement_time[cam_no]
                        if elapsed > 5.0:
                            fps = n_frames[cam_no]/elapsed
                            self.main_brain_lock.acquire()
                            self.main_brain.set_fps(cam_id,fps)
                            self.main_brain_lock.release()
                            last_measurement_time[cam_no] = now
                            n_frames[cam_no] = 0

                        # get new frames from grab thread
                        lock = globals['incoming_frames_lock']
                        lock.acquire()
                        t1=time_func()
                        gif = globals['incoming_frames']
                        len_if = len(gif)
                        if len_if:
                            n_frames[cam_no] = n_frames[cam_no]+len_if
                            grabbed_frames[cam_no].extend( gif )
                            globals['incoming_frames'] = []
                        lock.release()
                        t2=time_func()
                        diff = t2-t1
                        if diff > 0.005:
                            print '                        Held lock for %f msec'%(diff*1000.0,)

                        # process asynchronous commands
                        self.main_brain_lock.acquire()
                        cmds=self.main_brain.get_and_clear_commands(cam_id)
                        self.main_brain_lock.release()
                        self.handle_commands(cam_no,cmds)
                            
                        # handle saving movie if needed
                        fly_movie = globals['currently_saving_fly_movie']

                        gfcn = grabbed_frames[cam_no]
                        len_gfcn = len(gfcn)
                        if len_gfcn:
                            if fly_movie is not None:
                                t1=time_func()
                                if 1:
                                    for frame,timestamp,framenumber in gfcn:
                                        fly_movie.add_frame(frame,timestamp)
                                    sz= frame.shape[1]*frame.shape[0]
                                else:
                                    frames, timestamps, framenumbers = zip(*gfcn)
                                    fly_movie.add_frames(frames,timestamps)
                                    sz= frames[0].shape[1]*frames[0].shape[0]
                                t2=time_func()
                                tdiff = t2-t1
                                mb_per_sec = len_gfcn*sz/(1024*1024)/tdiff

                            grabbed_frames[cam_no] = []

                    time.sleep(0.05)

            finally:
                self.main_brain_lock.acquire()
                for cam_id in self.all_cam_ids:
                    self.main_brain.close(cam_id)
                self.main_brain_lock.release()
                for cam_no in range(self.num_cams):
                    self.globals[cam_no]['cam_quit_event'].set()                    
        except Pyro.errors.ConnectionClosedError:
            print 'unexpected connection closure...'

def main():
    app=App()
    app.mainloop()
    print

if __name__=='__main__':
    main()
        
