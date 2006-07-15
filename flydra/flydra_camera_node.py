#emacs, this is -*-Python-*- mode
# $Id: $

import os
BENCHMARK = int(os.environ.get('FLYDRA_BENCHMARK',0))

import threading, time, socket, sys, struct, select, math
import Queue
import numpy as nx

import FlyMovieFormat
import cam_iface
if not BENCHMARK:
    import Pyro.core, Pyro.errors
    Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!
    Pyro.config.PYRO_TRACELEVEL = 0
    Pyro.config.PYRO_USER_TRACELEVEL = 0
    Pyro.config.PYRO_DETAILED_TRACEBACK = 1
    Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1
    ConnectionClosedError = Pyro.errors.ConnectionClosedError
else:
    class NonExistantError(Exception):
        pass
    ConnectionClosedError = NonExistantError
import flydra.reconstruct_utils as reconstruct_utils
import FastImage
if os.name == 'posix':
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
REALTIME_UDP = flydra.common_variables.REALTIME_UDP

import flydra_ipp.realtime_image_analysis4 as realtime_image_analysis

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time

pt_fmt = '<dddddddddBBdd'
small_datafile_fmt = '<dII'
    
CAM_CONTROLS = {'shutter':cam_iface.SHUTTER,
                'gain':cam_iface.GAIN,
                'brightness':cam_iface.BRIGHTNESS}

ALPHA = 1.0/50 # relative importance of each new frame
BG_FRAME_INTERVAL = 50 # every N frames, add a new BG image to the accumulator

# where is the "main brain" server?
try:
    main_brain_hostname = socket.gethostbyname('brain1')
except:
    # try localhost
    main_brain_hostname = socket.gethostbyname(socket.gethostname())

def TimestampEcho():
    # create listening socket
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj.setblocking(0)
    hostname = ''
    port = flydra.common_variables.timestamp_echo_listener_port
    sockobj.bind(( hostname, port))
    sendto_port = flydra.common_variables.timestamp_echo_gatherer_port
    fmt = flydra.common_variables.timestamp_echo_fmt_diff
    while 1:
        in_ready, trash1, trash2 = select.select( [sockobj], [], [], 0.0 )
        if not len(in_ready):
            continue
        buf, (orig_host,orig_port) = sockobj.recvfrom(4096)
        newbuf = buf + struct.pack( fmt, time.time() )
        sender.sendto(newbuf,(orig_host,sendto_port))

class GrabClass(object):
    def __init__(self, cam, cam2mainbrain_port, cam_id, log_message_queue):
        self.cam = cam
        self.cam2mainbrain_port = cam2mainbrain_port
        self.cam_id = cam_id
        self.log_message_queue = log_message_queue

        self.new_roi = threading.Event()
        self.new_roi_data = None
        max_num_points = 3
        self.realtime_analyzer = realtime_image_analysis.RealtimeAnalyzer(self.cam.get_max_width(),
                                                                          self.cam.get_max_height(),
                                                                          max_num_points)

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

        print 'received lens distortion information...'
        self.realtime_analyzer.set_reconstruct_helper( helper )
    
    def grab_func(self,globals):
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
        max_frame_size = FastImage.Size(self.cam.get_max_width(), self.cam.get_max_height())

        hw_roi_w, hw_roi_h = self.cam.get_frame_size()
        cur_fisize = FastImage.Size(hw_roi_w, hw_roi_h)
        
        bg_changed = True
        use_roi2_isSet = globals['use_roi2'].isSet
        fi8ufactory = FastImage.FastImage8u
        use_cmp_isSet = globals['use_cmp'].isSet
        return_first_xy = 0
        
        hw_roi_frame = fi8ufactory( cur_fisize )

        if BENCHMARK:
            coord_socket = DummySocket()
        else:
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
                msg = 'excellent, grab thread running in maximum prioity mode'
            except Exception, x:
                msg = 'WARNING: could not run in maximum priority mode:', str(x)
            self.log_message_queue.put((self.cam_id,time.time(),msg))
            print msg

        
        #FastImage.set_debug(1) # let us see any images malloced, should only happen on hardware ROI size change
        
        self.cam.start_camera()  # start camera

        # take first image to set background and so on
        try:
            self.cam.grab_next_frame_into_buf_blocking(hw_roi_frame)
        except cam_iface.BuffersOverflowed:
            print >> sys.stderr , 'On start warning: buffers overflowed on %s at %s'%(self.cam_id,time.asctime())

        running_mean8u_im = self.realtime_analyzer.get_image_view('mean') # this is a view we write into

        # allocate images and initialize if necessary
        running_mean_im = hw_roi_frame.get_32f_copy(max_frame_size)
        running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )

        fastframef32_tmp = FastImage.FastImage32f(max_frame_size)
        
        mean2 = FastImage.FastImage32f(max_frame_size)
        running_stdframe = FastImage.FastImage32f(max_frame_size)
        compareframe = FastImage.FastImage32f(max_frame_size)
        compareframe8u = self.realtime_analyzer.get_image_view('cmp') # this is a view we write into
        
        running_sumsqf = FastImage.FastImage32f(max_frame_size)
        running_sumsqf.set_val(0,max_frame_size)
        
        noisy_pixels_mask = FastImage.FastImage8u(max_frame_size)
        mean_duration_no_bg = 0.0053 # starting value
        mean_duration_bg = 0.020 # starting value
        
        incoming_raw_frames_queue_put = globals['incoming_raw_frames'].put
        if BENCHMARK:
            benchmark_start_time = time.time()
        try:
            while not cam_quit_event_isSet():
                try:
                    self.cam.grab_next_frame_into_buf_blocking(hw_roi_frame)
                except cam_iface.BuffersOverflowed:
                    now = time.time()
                    msg = 'ERROR: buffers overflowed on %s at %s'%(self.cam_id,time.asctime(time.localtime(now)))
                    self.log_message_queue.put((self.cam_id,now,msg))
                    print >> sys.stderr, msg

                received_time = time.time()

                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()
                framenumber=self.cam.get_last_framenumber()

                if BENCHMARK:
                    if (framenumber%100) == 0:
                        dur = received_time-benchmark_start_time
                        print '%.1f msec for 100 frames'%(dur*1000.0)
                        benchmark_start_time = received_time
                else:
                    diff = timestamp-old_ts
                    if diff > 0.02:
                        msg = 'Warning: IFI is %f on %s at %s'%(diff,self.cam_id,time.asctime())
                        self.log_message_queue.put((self.cam_id,time.time(),msg))
                        print >> sys.stderr, msg
                    if framenumber-old_fn > 1:
                        msg = '  frames apparently skipped: %d'%(framenumber-old_fn,)
                        self.log_message_queue.put((self.cam_id,time.time(),msg))
                        print >> sys.stderr, msg
                old_ts = timestamp
                old_fn = framenumber

                work_start_time = time.time()
                points = self.realtime_analyzer.do_work(hw_roi_frame,
                                                        timestamp, framenumber, use_roi2_isSet(),
                                                        use_cmp_isSet(), return_first_xy,
                                                        0.010, # maximum 10 msec in here
                                                        )
                work_done_time = time.time()
                
                # allow other thread to see images
                imname = globals['export_image_name'] # figure out what is wanted # XXX theoretically could have threading issue
                if imname == 'raw':
                    export_image = hw_roi_frame
                else:
                    export_image = self.realtime_analyzer.get_image_view(imname) # get image
                globals['most_recent_frame_potentially_corrupt'] = (0,0), export_image # give view of image, receiver must be careful

                tp1 = time.time()
                
                # allow other thread to see raw image always (for saving)
                incoming_raw_frames_queue_put(
                    (hw_roi_frame.get_8u_copy(hw_roi_frame.size), # save a copy
                     timestamp,
                     framenumber,
                     points,
                     self.realtime_analyzer.roi,
                     ) )

                tp2 = time.time()
                did_expensive = False
                if collecting_background_isSet():
                    if bg_frame_number % BG_FRAME_INTERVAL == 0:
                        if cur_fisize != max_frame_size:
                            # set to full ROI and take full image if necessary
                            raise NotImplementedError("background collection while using hardware ROI not implemented")
                        if 1:

                            # maintain running average
                            running_mean_im.toself_add_weighted( hw_roi_frame, max_frame_size, ALPHA )
                            # maintain 8bit unsigned background image
                            running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )

                            # standard deviation calculation
                            hw_roi_frame.get_32f_copy_put(fastframef32_tmp,max_frame_size)
                            fastframef32_tmp.toself_square(max_frame_size) # current**2
                            running_sumsqf.toself_add_weighted( fastframef32_tmp, max_frame_size, ALPHA)
                            running_mean_im.get_square_put(mean2,max_frame_size)
                            running_sumsqf.get_subtracted_put(mean2,running_stdframe,max_frame_size)

                            # now create frame for comparison
                            running_stdframe.toself_multiply(6.0,max_frame_size)
                            running_stdframe.get_8u_copy_put(compareframe8u,max_frame_size)

                            # now we do hack, erm, heuristic for bright points, which aren't gaussian.
                            running_mean8u_im.get_compare_put( 200, noisy_pixels_mask, max_frame_size, FastImage.CmpGreater)
                            compareframe8u.set_val_masked(25, noisy_pixels_mask, max_frame_size)
                        else:
                            realtime_image_analysis.bg_help( running_mean_im,
                                                             fastframef32_tmp,
                                                             running_sumsqf,
                                                             mean2,
                                                             running_stdframe,
                                                             running_mean8u_im,
                                                             hw_roi_frame,
                                                             noisy_pixels_mask,
                                                             compareframe8u,
                                                             max_frame_size,
                                                             ALPHA,
                                                             6.0)
                        did_expensive = True
                        bg_changed = True
                        bg_frame_number = 0
                    bg_frame_number += 1
                
                tp3 = time.time()
                
                if take_background_isSet():
                    # reset background image with current frame as mean and 0 STD
                    hw_roi_frame.get_32f_copy_put( running_mean_im, max_frame_size )
                    running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )

                    if 1:
                        running_sumsqf.set_val( 0, max_frame_size )
                        compareframe8u.set_val(0, max_frame_size )
                    else:
                        # XXX TODO: cleanup
                        hw_roi_frame.get_32f_copy_put(running_sumsqf,max_frame_size)
                        running_sumsqf.toself_square(max_frame_size)
                        
                        running_mean_im.get_square_put(mean2,max_frame_size)
                        running_sumsqf.get_subtracted_put(mean2,running_stdframe,max_frame_size)
                        
                        compareframe8u.set_val(0, max_frame_size )
                    
                    bg_changed = True
                    take_background_clear()
                    
                tp4 = time.time()
                
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
                        bg_image = running_mean8u_im.get_8u_copy(running_mean8u_im.size)
                        std_image = compareframe8u.get_8u_copy(compareframe8u.size)
                    elif 0:
                        bg_image = nx.array(running_mean8u_im) # make copy (we don't want to send live versions of image
                        std_image = nx.array(compareframe8u) # make copy (we don't want to send live versions of image
                    else:
                        bg_image = running_mean8u_im
                        std_image = compareframe8u
                    globals['current_bg_frame_and_timestamp']=bg_image,std_image,timestamp # only used when starting to save
                    globals['incoming_bg_frames'].put(
                        (bg_image,std_image,timestamp,framenumber) ) # save it
                    bg_changed = False
                    
                if find_rotation_center_start_isSet():
                    find_rotation_center_start_clear()
                    rot_frame_number=0
                    self.realtime_analyzer.rotation_calculation_init( n_rot_samples )
                    
                if rot_frame_number>=0:
                    find_rotation_center_start_clear()
                    if len(points) > 0:
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
                data = struct.pack('<dli',timestamp,framenumber,len(points))
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
                    self.cam.stop_camera()  # stop camera
                    lbrt = self.new_roi_data
                    self.new_roi_data = None
                    l,b,r,t=lbrt
                    w = r-l+1
                    h = t-b+1
                    self.realtime_analyzer.roi = lbrt
                    self.cam.set_frame_size(w,h)
                    self.cam.set_frame_offset(l,b)
                    w,h = self.cam.get_frame_size()
                    l,b= self.cam.get_frame_offset()
                    cur_fisize = FastImage.Size(w, h)
                    hw_roi_frame = fi8ufactory( cur_fisize )
                    self.realtime_analyzer.roi = (l,b,r,t)

                    self.new_roi.clear()
                    self.cam.start_camera()  # start camera
                    
                bookkeeping_done_time = time.time()
                bookkeeping_dur = bookkeeping_done_time-received_time

                alpha = 0.01
                if did_expensive:
                    mean_duration_bg = (1-alpha)*mean_duration_bg + alpha*bookkeeping_dur
                else:
                    mean_duration_no_bg = (1-alpha)*mean_duration_no_bg + alpha*bookkeeping_dur

                if bookkeeping_dur > 0.050 and not BENCHMARK:
                    print 'TIME BUDGET:'
                    print '   % 5.1f start of work'%((work_start_time-received_time)*1000.0,)
                    print '   % 5.1f done with work'%((work_done_time-received_time)*1000.0,)
                    
                    print '   % 5.1f tp1'%((tp1-received_time)*1000.0,)
                    print '   % 5.1f tp2'%((tp2-received_time)*1000.0,)
                    if did_expensive:
                        print '     (did background/variance estimate)'
                    print '   % 5.1f tp3'%((tp3-received_time)*1000.0,)
                    print '   % 5.1f tp4'%((tp4-received_time)*1000.0,)
                    
                    print '   % 5.1f end of all'%(bookkeeping_dur*1000.0,)
                    print
                    print 'mean_duration_bg',mean_duration_bg*1000
                    print 'mean_duration_no_bg',mean_duration_no_bg*1000
                    print
                    
        finally:
            self.realtime_analyzer.close()
            #FastImage.set_debug(0)
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
        if not BENCHMARK:
            Pyro.core.initClient(banner=0)
        
        if BENCHMARK:
            self.main_brain = DummyMainBrain()
        else:
            port = 9833
            name = 'main_brain'
            main_brain_URI = "PYROLOC://%s:%d/%s" % (main_brain_hostname,port,name)
            print 'connecting to',main_brain_URI
            self.main_brain = Pyro.core.getProxyForURI(main_brain_URI)
            self.main_brain._setOneway(['set_image','set_fps','close','log_message'])
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
        self.reconstruct_helper = []        
        
        timestamp_echo_thread=threading.Thread(target=TimestampEcho)
        timestamp_echo_thread.setDaemon(True) # quit that thread if it's the only one left...
        timestamp_echo_thread.start()
        
        for cam_no in range(self.num_cams):
            num_buffers = 205
            cam = cam_iface.Camera(cam_no,num_buffers,7,0,0) # last 3 parameters only used on window drivers for now
            print 'allocated %d buffers'%num_buffers
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
            globals['small_fmf']=None
            globals['most_recent_frame_potentially_corrupt']=None
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
            globals['export_image_name'] = 'raw'
            globals['use_roi2'] = threading.Event()
            globals['use_cmp'] = threading.Event()
            globals['use_cmp'].set()
            
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
            diff_threshold = 11
            scalar_control_info['diff_threshold'] = diff_threshold
            clear_threshold = 0.2
            scalar_control_info['clear_threshold'] = clear_threshold
            scalar_control_info['visible_image_view'] = 'raw'
            
            try:
                scalar_control_info['trigger_source'] = cam.get_trigger_source()
            except cam_iface.CamIFaceError:
                scalar_control_info['trigger_source'] = 0
            scalar_control_info['roi2'] = globals['use_roi2'].isSet()
            scalar_control_info['cmp'] = globals['use_cmp'].isSet()
            
            scalar_control_info['width'] = width
            scalar_control_info['height'] = height
            scalar_control_info['roi'] = 0,0,width-1,height-1
            scalar_control_info['max_framerate'] = cam.get_framerate()
            scalar_control_info['collecting_background']=globals['collecting_background'].isSet()
            
            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            self.main_brain_lock.acquire()
            cam_id = self.main_brain.register_new_camera(cam_no,
                                                         scalar_control_info,
                                                         port)

            self.all_cam_ids.append(cam_id)
            cam2mainbrain_port = self.main_brain.get_cam2mainbrain_port(self.all_cam_ids[cam_no])
            self.main_brain_lock.release()

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
            grabber = GrabClass(cam,cam2mainbrain_port,cam_id,self.log_message_queue)
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
                        print 'setting diff_threshold',value
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
                    elif property_name == 'cmp':
                        if value: globals['use_cmp'].set()
                        else: globals['use_cmp'].clear()
                    elif property_name == 'max_framerate':
                        cam.set_framerate(value)
                    elif property_name == 'collecting_background':
                        if value: globals['collecting_background'].set()
                        else: globals['collecting_background'].clear()
                    elif property_name == 'visible_image_view':
                        globals['export_image_name'] = value
                        print 'displaying',value,'image'
            elif key == 'get_im':
                val = globals['most_recent_frame_potentially_corrupt']
                if val is not None: # prevent race condition
                    lb, im = val
                    #nxim = nx.array(im) # copy to native nx form, not view of __array_struct__ form
                    nxim = nx.asarray(im) # view of __array_struct__ form
                    self.main_brain.set_image(cam_id, (lb, nxim))
            elif key == 'use_arena':
                grabber.use_arena = cmds[key]
                globals['use_arena'] = grabber.use_arena
            elif key == 'quit':
                globals['cam_quit_event'].set()
            elif key == 'take_bg':
                globals['take_background'].set()
            elif key == 'clear_bg':
                globals['clear_background'].set()
##            elif key == 'collecting_bg':
##                if cmds[key]:
##                    globals['collecting_background'].set()
##                    print 'set collecting'
##                else:
##                    globals['collecting_background'].clear()
##                    print 'cleared collecting'
            elif key == 'find_r_center':
                globals['find_rotation_center_start'].set()
            elif key == 'stop_recording':
                if globals['raw_fmf_and_bg_fmf'] is not None:
                    raw_movie, bg_movie, std_movie = globals['raw_fmf_and_bg_fmf']
                    raw_movie.close()
                    bg_movie.close()
                    std_movie.close()
                    print 'stopped recording'
                    globals['saved_bg_frame']=False
                    globals['raw_fmf_and_bg_fmf'] = None
            elif key == 'stop_small_recording':
                if globals['small_fmf'] is not None:
                    small_movie, small_datafile = globals['small_fmf']
                    small_movie.close()
                    small_datafile.close()
                    print 'stopped small recording'
                    globals['small_fmf'] = None
            elif key == 'start_recording':
                raw_filename, bg_filename = cmds[key]
                
                raw_filename = os.path.expanduser(raw_filename)
                bg_filename = os.path.expanduser(bg_filename)
                
                save_dir = os.path.split(raw_filename)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                std_filename = bg_filename.replace('_bg','_std')
                msg = 'WARNING: fly movie filenames will conflict if > 1 camera per computer'
                print msg
                raw_movie = FlyMovieFormat.FlyMovieSaver(raw_filename,version=1)
                bg_movie = FlyMovieFormat.FlyMovieSaver(bg_filename,version=1)
                std_movie = FlyMovieFormat.FlyMovieSaver(std_filename,version=1)
                globals['raw_fmf_and_bg_fmf'] = raw_movie, bg_movie, std_movie
                globals['saved_bg_frame']=False
                msg = "starting to record to %s\n"%raw_filename
                msg += "  background to %s\n"%bg_filename
                msg += "  comparison frames to %s"%std_filename
                print msg
            elif key == 'start_small_recording':
                small_movie_filename, small_datafile_filename = cmds[key]
                
                small_movie_filename = os.path.expanduser(small_movie_filename)
                small_datafile_filename = os.path.expanduser(small_datafile_filename)
                
                save_dir = os.path.split(small_movie_filename)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                small_movie = FlyMovieFormat.FlyMovieSaver(small_movie_filename,version=1)
                small_datafile = file( small_datafile_filename, mode='wb' )
                globals['small_fmf'] = small_movie, small_datafile
                print "starting to record small movies to %s"%small_movie_filename
##            elif key == 'debug': # kept for backwards compatibility
##                if cmds[key]: globals['export_image_name'] = 'absdiff'
##                else: globals['export_image_name'] = 'raw'
            elif key == 'cal':
                pmat, intlin, intnonlin = cmds[key]
                grabber.pmat = pmat
                grabber.make_reconstruct_helper(intlin, intnonlin) # let grab thread make one

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
                raise ValueError("Unknown key '%s'"%key)
                
    def mainloop(self):
        # per camera variables
        last_measurement_time = []
        last_return_info_check = []
        n_raw_frames = []
        last_found_timestamp = [0.0]*self.num_cams
        last_frames_by_cam = [ [] for c in range(self.num_cams) ]
        
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
                        last_frames = last_frames_by_cam[cam_no]

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
                            std_movie = None
                        else:
                            raw_movie, bg_movie, std_movie = raw_fmf_and_bg_fmf
                            
                        # Are we saving small movies?
                        small_fmf_and_small_datafile = globals['small_fmf']
                        if small_fmf_and_small_datafile is None:
                            small_movie = None
                            small_datafile = None
                        else:
                            small_movie, small_datafile = small_fmf_and_small_datafile
                            
                        # Get new raw frames from grab thread.
                        get_raw_frame_nowait = globals['incoming_raw_frames'].get_nowait
                        try:
                            while 1:
                                frame,timestamp,framenumber,points,lbrt = get_raw_frame_nowait() # this may raise Queue.Empty
                                # XXX could have option to skip frames if a newer frame is available
                                last_frames.append( (frame,timestamp,framenumber,points) )
                                while len(last_frames)>1000:
                                    del last_frames[0]
                                n_pts = len(points)
                                if n_pts>0:
                                    last_found_timestamp[cam_no] = timestamp
                                # save movie for 1 second after I found anything
                                if ((raw_movie is not None) and
                                    (timestamp - last_found_timestamp[cam_no]) < 1.0):
                                    raw_movie.add_frame(frame,timestamp)
                                if small_movie is not None and n_pts>0:
                                    pt = points[0] # save only first found point currently
                                    
                                    x0, y0 = pt[0],pt[1] # absolute values, distorted
                                    l,b,r,t = lbrt # absolute values
                                    hw_roi_w = r-l
                                    hw_roi_h = t-b
                                    small_width = 20 # width
                                    small_height = 20 # height
                                    small_width2 = small_width//2 # half of total width
                                    small_height2 = small_height//2 # half of total height
                                    if ((small_width > hw_roi_w) or
                                        (small_height > hw_roi_h)):
                                        raise RuntimeError('FMF frame size (for small movie) is bigger than hardware ROI')
                                    
                                    save_l = int(round(x0 - small_width2))
                                    if save_l < l:
                                        save_l = l
                                    save_r = save_l+small_width
                                    if save_r > r:
                                        save_r = r
                                        save_l = save_r-small_width

                                    save_b = int(round(y0 - small_height2))
                                    if save_b < b:
                                        save_b = b
                                    save_t = save_b+small_height
                                    if save_t > t:
                                        save_t = t
                                        save_b = save_t-small_height

                                    if isinstance(frame,FastImage.FastImageBase):
                                        small_frame = frame.roi( save_l, save_b,
                                                                 FastImage.Size( small_width, small_height ) )
                                    else:
                                        warnings.warn('memory leak!')
                                        nxframe = nx.asarray(frame)
                                        small_frame = nxframe[save_b:save_t,save_l:save_r]
                                        
                                    small_movie.add_frame(small_frame,timestamp)
                                    small_datafile.write(
                                        struct.pack( small_datafile_fmt,
                                                     timestamp, save_l, save_b) )
                                n_raw_frames[cam_no] += 1
                        except Queue.Empty:
                            pass

                        # Get new BG frames from grab thread.
                        get_bg_frame_nowait = globals['incoming_bg_frames'].get_nowait
                        try:
                            while 1:
                                bg_frame,std_frame,timestamp,framenumber = get_bg_frame_nowait() # this may raise Queue.Empty
                                if bg_movie is not None:
                                    bg_movie.add_frame(bg_frame,timestamp)
                                    std_movie.add_frame(std_frame,timestamp)
                                    globals['saved_bg_frame'] = True
                        except Queue.Empty:
                            pass

                        try:
                            while 1:
                                args = self.log_message_queue.get_nowait()
                                self.main_brain.log_message(*args)
                        except Queue.Empty:
                            pass

                        # make sure a BG frame is saved at beginning of movie
                        if bg_movie is not None and not globals['saved_bg_frame']:
                            bg_frame,std_frame,timestamp = globals['current_bg_frame_and_timestamp']
                            bg_movie.add_frame(bg_frame,timestamp)
                            std_movie.add_frame(std_frame,timestamp)
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
        except ConnectionClosedError:
            print 'unexpected connection closure...'

def main():
    app=App()
    if app.num_cams <= 0:
        return
    app.mainloop()
    print

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
        
