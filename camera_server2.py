#emacs, this is -*-Python-*- mode
# $Id$

import threading, time, socket, sys, struct, os
import Pyro.core, Pyro.errors
import FlyMovieFormat
import numarray as nx
import pyx_cam_iface_numarray as cam_iface
import reconstruct_utils
import Queue
if os.name == 'posix':
    import posix_sched
import math

from common_variables import REALTIME_UDP

try:
    import realtime_image_analysis
except ImportError, x:
    if str(x).startswith('libippcore.so: cannot open shared object file'):
        print 'WARNING: IPP not loaded, proceeding without it'
        import realtime_image_analysis_noipp as realtime_image_analysis
    else:
        raise

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time

pt_fmt = '<dddddddddBB'
    
Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

CAM_CONTROLS = {'shutter':cam_iface.SHUTTER,
                'gain':cam_iface.GAIN,
                'brightness':cam_iface.BRIGHTNESS}

ALPHA = 1.0/10 # relative importance of each new frame
BG_FRAME_INTERVAL = 20 # every N frames, add a new BG image to the accumulator

# where is the "main brain" server?
try:
    main_brain_hostname = socket.gethostbyname('brain1')
except:
    # try localhost
    main_brain_hostname = socket.gethostbyname(socket.gethostname())

class GrabClass(object):
    def __init__(self, cam, cam2mainbrain_port, cam_id):
        self.cam = cam
        self.cam2mainbrain_port = cam2mainbrain_port
        self.cam_id = cam_id

        # get coordinates for region of interest
        max_width = self.cam.get_max_width()
        max_height = self.cam.get_max_height()

        hw_roi_w, hw_roi_h = self.cam.get_frame_size()
        hw_roi_l, hw_roi_b = self.cam.get_frame_offset()
        self.new_roi = threading.Event()
        self.new_roi_data = None
        self.realtime_analyzer = realtime_image_analysis.RealtimeAnalyzer(max_width,
                                                                          max_height,
                                                                          hw_roi_w,
                                                                          hw_roi_h,
                                                                          hw_roi_l,
                                                                          hw_roi_b,
                                                                          ALPHA)

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

    def get_pmat(self):
        return self.realtime_analyzer.pmat
    def set_pmat(self,value):
        self.realtime_analyzer.pmat = value
    pmat = property( get_pmat, set_pmat )

    def get_use_arena(self):
        return self.realtime_analyzer.use_arena
    def set_use_arena(self, value):
        self.realtime_analyzer.use_arena = value
    use_arena = property( get_use_arena, set_use_arena )

    def get_roi(self):
        return self.realtime_analyzer.roi
    def set_roi(self, lbrt):
        self.new_roi_data = lbrt
        self.new_roi.set()
    roi = property( get_roi, set_roi )

    def make_reconstruct_helper(self, intlin, intnonlin):
        fc1 = intlin[0,0]
        fc2 = intlin[1,1]
        cc1 = intlin[0,2]
        cc2 = intlin[1,2]
        k1, k2, p1, p2 = intnonlin
        
        helper = reconstruct_utils.ReconstructHelper(
            fc1, fc2, cc1, cc2, k1, k2, p1, p2 )
        
        self.realtime_analyzer.set_reconstruct_helper( helper )
    
    def grab_func(self,globals):
        n_bg_samples = 100
        
        # questionable optimization: speed up by eliminating namespace lookups
        cam_quit_event_isSet = globals['cam_quit_event'].isSet
        sleep = time.sleep
        bg_frame_number = 0
        rot_frame_number = -1
        clear_background_isSet = globals['clear_background'].isSet
        clear_background_clear = globals['clear_background'].clear
        take_background_isSet = globals['take_background'].isSet
        take_background_clear = globals['take_background'].clear
        collecting_background_isSet = globals['collecting_background'].isSet
        find_rotation_center_start_isSet = globals['find_rotation_center_start'].isSet
        find_rotation_center_start_clear = globals['find_rotation_center_start'].clear
        debug_isSet = globals['debug'].isSet
        height = self.cam.get_max_height()
        width = self.cam.get_max_width()
        buf_ptr_step = width
        bg_changed = True
        use_roi2_isSet = globals['use_roi2'].isSet

        if REALTIME_UDP:
            coord_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            coord_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            coord_socket.connect((main_brain_hostname,self.cam2mainbrain_port))

        old_ts = time.time()
        old_fn = 0
        n_rot_samples = 560*60 # 1 minute -- WARNING: not valid for all framerates!
        points = []

        if os.name == 'posix':
            try:
                max_priority = posix_sched.get_priority_max( posix_sched.FIFO )
                sched_params = posix_sched.SchedParam(max_priority)
                posix_sched.setscheduler(0, posix_sched.FIFO, sched_params)
                print 'excellent, grab thread running in maximum prioity mode'
            except Exception, x:
                print 'WARNING: could not run in maximum priority mode:', str(x)
        
        self.cam.start_camera()  # start camera

        try:
            while not cam_quit_event_isSet():
                try:
                    framebuffer = self.cam.grab_next_frame_blocking_numarray()
                except cam_iface.BuffersOverflowed:
                    print >> sys.stderr , 'ERROR: buffers overflowed on %s at %s'%(self.cam_id,time.asctime())
                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()
                framenumber=self.cam.get_last_framenumber()

                diff = timestamp-old_ts
                if diff > 0.02:
                    print >> sys.stderr, 'Warning: IFI is %f on %s at %s'%(diff,self.cam_id,time.asctime())
                if framenumber-old_fn > 1:
                    print >> sys.stderr, '  frames apparently skipped:', framenumber-old_fn
                old_ts = timestamp
                old_fn = framenumber
                
                points = self.realtime_analyzer.do_work(
                    framebuffer, timestamp, framenumber, use_roi2_isSet() )
                n_pts = len(points)
                raw_image = framebuffer
                
                # make appropriate references to our copy of the data
                if debug_isSet():
                    debug_image = self.realtime_analyzer.get_working_image()
                    globals['most_recent_frame'] = (0,0), debug_image
                else:
                    l,b= self.cam.get_frame_offset()
                    globals['most_recent_frame'] = (l,b), raw_image
                    
                globals['incoming_raw_frames'].put(
                    (raw_image,timestamp,framenumber,n_pts) ) # save it

                if collecting_background_isSet():
                    if bg_frame_number % BG_FRAME_INTERVAL == 0:
                        self.realtime_analyzer.accumulate_last_image()
                        bg_changed = True
                        bg_frame_number = 0
                    bg_frame_number += 1
                
                if take_background_isSet():
                    self.realtime_analyzer.take_background_image()
                    bg_changed = True
                    take_background_clear()
                    
                if clear_background_isSet():
                    self.realtime_analyzer.clear_background_image()
                    bg_changed = True
                    clear_background_clear()

                if bg_changed:
                    bg_image = self.realtime_analyzer.get_background_image()
                    globals['current_bg_frame_and_timestamp']=bg_image,timestamp
                    globals['incoming_bg_frames'].put(
                        (bg_image,timestamp,framenumber) ) # save it
                    bg_changed = False
                    
                if find_rotation_center_start_isSet():
                    find_rotation_center_start_clear()
                    rot_frame_number=0
                    self.realtime_analyzer.rotation_calculation_init( n_rot_samples )
                    
                if rot_frame_number>=0:
                    find_rotation_center_start_clear()
                    if n_pts != 0:
                        pt = points[0]
                        x0, y0, slope = pt[0],pt[1],pt[3]
                    else:
                        x0 = y0 = slope = 0.0
                    if slope != 0.0:
                        rise = 1.0
                        run = rise/slope
                        orientation = math.atan2(rise,run)
                        orientation = orientation + math.pi/2.0
                    else:
                        orientation = math.pi/2.0
                    self.realtime_analyzer.rotation_update(x0,y0,orientation,timestamp)
                    rot_frame_number += 1
                    if rot_frame_number>=n_rot_samples:
                        self.realtime_analyzer.rotation_end()
                        rot_frame_number=-1 # stop averaging frames
              
                # XXX could speed this with a join operation I think
                data = struct.pack('<dli',timestamp,framenumber,n_pts)
                for point_tuple in points:
                    try:
                        data = data + struct.pack(pt_fmt,*point_tuple)
                    except:
                        print 'error-causing data: ',point_tuple
                        raise
                if REALTIME_UDP:
                    coord_socket.sendto(data,
                                        (main_brain_hostname,self.cam2mainbrain_port))
                else:
                    coord_socket.send(data)
                    
                if self.new_roi.isSet():
                    self.cam.stop_camera()  # start camera
                    print 'stopped camera'
                    lbrt = self.new_roi_data
                    self.new_roi_data = None
                    l,b,r,t=lbrt
                    w = r-l+1
                    h = t-b+1
                    self.realtime_analyzer.roi = lbrt
                    print 'camera setting size to',w,h
                    print 'camera setting offset to',l,b
                    self.cam.set_frame_size(w,h)
                    self.cam.set_frame_offset(l,b)
                    w,h = self.cam.get_frame_size()
                    l,b= self.cam.get_frame_offset()
                    print 'set to',w,h,l,b
                    self.realtime_analyzer.set_hw_roi(w,h,l,b)

                    print 're-starting camera'
                    self.new_roi.clear()
                    self.cam.start_camera()  # start camera
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
        if self.num_cams <= 0:
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
            num_buffers = 20
            cam = cam_iface.Camera(cam_no,num_buffers,7,0,0) # last 3 parameters only used on window drivers for now
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

            globals['incoming_raw_frames']=Queue.Queue()
            globals['incoming_bg_frames']=Queue.Queue()
            globals['raw_fmf_and_bg_fmf']=None
            globals['most_recent_frame']=None
            globals['saved_bg_frame']=False
            globals['current_bg_frame_and_timestamp']=None

            # control flow events for threading model
            globals['cam_quit_event'] = threading.Event()
            globals['listen_thread_done'] = threading.Event()
            globals['grab_thread_done'] = threading.Event()
            globals['take_background'] = threading.Event()
            globals['clear_background'] = threading.Event()
            globals['collecting_background'] = threading.Event()
            globals['collecting_background'].set()
            globals['find_rotation_center_start'] = threading.Event()
            globals['debug'] = threading.Event()
            globals['use_roi2'] = threading.Event()

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

            try:
                scalar_control_info['trigger_source'] = cam.get_trigger_source()
            except cam_iface.CamIFaceError:
                scalar_control_info['trigger_source'] = 0
            scalar_control_info['roi2'] = globals['use_roi2'].isSet()
            
            scalar_control_info['width'] = width
            scalar_control_info['height'] = height
            scalar_control_info['roi'] = 0,0,width-1,height-1
            scalar_control_info['max_framerate'] = cam.get_framerate()

            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            self.main_brain_lock.acquire()
            cam_id = self.main_brain.register_new_camera(cam_no,
                                                         scalar_control_info,
                                                         port)

            self.all_cam_ids.append(cam_id)
            cam2mainbrain_port = self.main_brain.get_cam2mainbrain_port(self.all_cam_ids[cam_no])
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

            grabber = GrabClass(cam,cam2mainbrain_port,cam_id)
            self.all_grabbers.append( grabber )
            
            grabber.diff_threshold = diff_threshold
            grabber.clear_threshold = clear_threshold
            
            grabber.use_arena = False
            globals['use_arena'] = grabber.use_arena
            
            grab_thread=threading.Thread(target=grabber.grab_func,
                                         args=(globals,))
            grab_thread.setDaemon(True) # quit that thread if it's the only one left...
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
                    elif property_name == 'trigger_source':
                        cam.set_trigger_source( value )
                    elif property_name == 'roi2':
                        if value: globals['use_roi2'].set()
                        else: globals['use_roi2'].clear()
                    elif property_name == 'max_framerate':
                        cam.set_framerate(value)
            elif key == 'get_im':
                self.main_brain.set_image(cam_id, globals['most_recent_frame'])
            elif key == 'use_arena':
                grabber.use_arena = cmds[key]
                globals['use_arena'] = grabber.use_arena
            elif key == 'quit':
                globals['cam_quit_event'].set()
            elif key == 'take_bg':
                globals['take_background'].set()
            elif key == 'clear_bg':
                globals['clear_background'].set()
            elif key == 'collecting_bg':
                if cmds[key]:
                    globals['collecting_background'].set()
                else:
                    globals['collecting_background'].clear()
            elif key == 'find_r_center':
                globals['find_rotation_center_start'].set()
            elif key == 'stop_recording':
                if globals['raw_fmf_and_bg_fmf'] is not None:
                    raw_movie, bg_movie = globals['raw_fmf_and_bg_fmf']
                    raw_movie.close()
                    bg_movie.close()
                    print 'stopped recording'
                    globals['saved_bg_frame']=False
                    globals['raw_fmf_and_bg_fmf'] = None
            elif key == 'start_recording':
                raw_filename, bg_filename = cmds[key]
                raw_movie = FlyMovieFormat.FlyMovieSaver(raw_filename,version=1)
                bg_movie = FlyMovieFormat.FlyMovieSaver(bg_filename,version=1)
                globals['raw_fmf_and_bg_fmf'] = raw_movie, bg_movie
                globals['saved_bg_frame']=False
                print "starting to record to %s"%raw_filename
                print "  background to %s"%bg_filename
            elif key == 'debug':
                if cmds[key]: globals['debug'].set()
                else: globals['debug'].clear()
            elif key == 'cal':
                pmat, intlin, intnonlin = cmds[key]
                grabber.pmat = pmat
                grabber.make_reconstruct_helper(intlin, intnonlin)
            else:
                raise ValueError("Unknown key '%s'"%key)
                
    def mainloop(self):
        # per camera variables
        last_measurement_time = []
        last_return_info_check = []
        n_raw_frames = []
        last_found_timestamp = [0.0]*self.num_cams
        
        if self.num_cams == 0:
            return

        for cam_no in range(self.num_cams):
            last_measurement_time.append( time_func() )
            last_return_info_check.append( 0.0 ) # never
            n_raw_frames.append( 0 )
            
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

                        # calculate and send FPS every 5 sec
                        elapsed = now-last_measurement_time[cam_no]
                        if elapsed > 5.0:
                            fps = n_raw_frames[cam_no]/elapsed
                            self.main_brain_lock.acquire()
                            self.main_brain.set_fps(cam_id,fps)
                            self.main_brain_lock.release()
                            last_measurement_time[cam_no] = now
                            n_raw_frames[cam_no] = 0

                        # Are we saving movies?
                        raw_fmf_and_bg_fmf = globals['raw_fmf_and_bg_fmf']
                        if raw_fmf_and_bg_fmf is None:
                            raw_movie = None
                            bg_movie = None
                        else:
                            raw_movie, bg_movie = raw_fmf_and_bg_fmf
                            
                        # Get new raw frames from grab thread.
                        get_raw_frame_nowait = globals['incoming_raw_frames'].get_nowait
                        try:
                            qsize = globals['incoming_raw_frames'].qsize()
                            while 1:
                                frame,timestamp,framenumber,n_pts = get_raw_frame_nowait() # this may raise Queue.Empty
                                if n_pts>0:
                                    last_found_timestamp[cam_no] = timestamp
                                # save movie for 1 second after I found anything
                                if (timestamp - last_found_timestamp[cam_no]) < 1.0 and raw_movie is not None:
                                    raw_movie.add_frame(frame,timestamp)
                                n_raw_frames[cam_no] += 1
                        except Queue.Empty:
                            pass

                        # Get new BG frames from grab thread.
                        get_bg_frame_nowait = globals['incoming_bg_frames'].get_nowait
                        try:
                            while 1:
                                frame,timestamp,framenumber = get_bg_frame_nowait() # this may raise Queue.Empty
                                if bg_movie is not None:
                                    bg_movie.add_frame(frame,timestamp)
                                    globals['saved_bg_frame'] = True
                        except Queue.Empty:
                            pass

                        # make sure a BG frame is saved at beginning of movie
                        if bg_movie is not None and not globals['saved_bg_frame']:
                            frame,timestamp = globals['current_bg_frame_and_timestamp']
                            bg_movie.add_frame(frame,timestamp)
                            globals['saved_bg_frame'] = True

                        # process asynchronous commands
                        self.main_brain_lock.acquire()
                        cmds=self.main_brain.get_and_clear_commands(cam_id)
                        self.main_brain_lock.release()
                        self.handle_commands(cam_no,cmds)
                            
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
    if app.num_cams <= 0:
        return
    app.mainloop()
    print

if __name__=='__main__':
    main()
        
