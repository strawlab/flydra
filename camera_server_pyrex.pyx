#emacs, this is -*-Python-*- mode
# $Id$

import threading
import time
import socket
import sys
import Pyro.core, Pyro.errors
import FlyMovieFormat
import warnings
import struct
import numarray as nx

include "../cam_iface/src/pyx_cam_iface.pyx"
# this has the following side effects:
# cimport c_numarray # CamIFaceError, Camera, c_numarray
# c_numarray.import_libnumarray()

cimport c_lib

# start of IPP-requiring code
cimport ipp
cimport c_fit_params

class IPPError(Exception):
    pass

cdef void CHK( ipp.IppStatus errval ) except *:
    if errval != 0:
        raise IPPError("IPP status %d"%(errval,))

cdef void print_info_8u(ipp.Ipp8u* im, int im_step, ipp.IppiSize sz, object prefix):
    cdef ipp.Ipp32f minVal, maxVal
    cdef ipp.IppiPoint minIdx, maxIdx
    
    CHK(
        ipp.ippiMinMaxIndx_8u_C1R( im, im_step, sz,
                                   &minVal, &maxVal,
                                   &minIdx, &maxIdx ))
    print prefix,'min: %f, max: %f, minIdx: %d,%d, maxIdx: %d,%d'%(minVal,maxVal,
                                                                   minIdx.x,minIdx.y,
                                                                   maxIdx.x,maxIdx.y)
# end of IPP-requiring code

cimport arena_control

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

CAM_CONTROLS = {'shutter':c_cam_iface.SHUTTER,
                'gain':c_cam_iface.GAIN,
                'brightness':c_cam_iface.BRIGHTNESS}

# where is the "main brain" server?
try:
    main_brain_hostname = socket.gethostbyname('mainbrain')
except:
    # try localhost
    main_brain_hostname = socket.gethostbyname(socket.gethostname())

cdef class GrabClass:
    cdef Camera cam
    cdef int coord_port
    cdef int left, bottom, right, top
    cdef float diff_threshold
    cdef int use_arena
    
    cdef void set_camera_and_coord_port(self, Camera cam, object coord_port):
        self.cam = cam
        self.coord_port = coord_port
        
    def grab_func(self,globals):
        cdef unsigned char* buf_ptr
        cdef c_numarray._numarray buf
        cdef int height
        cdef int buf_ptr_step, width
        cdef int heightwidth[2]
        cdef int collecting_background_frames
        cdef int bg_frame_number
        cdef double new_x_cent, new_y_cent
        cdef int n_frames4stats
##        cdef double tval1, tval2, tval3, latency1, latency2
        cdef int i, j
        cdef double timestamp
        cdef long framenumber
        cdef int have_arena_control
        
        # start of IPP-requiring code
        cdef int index_x,index_y
#        cdef ipp.Ipp32f max_val, std_val
        cdef ipp.Ipp8u max_val, std_val
        cdef int im1_step, im2_step, sum_image_step, bg_img_step
        cdef int mean_image_step, std_image_step, sq_image_step, std_img_step
        cdef int centroid_search_radius
        cdef int n_bg_samples
        cdef int n_rot_samples
        cdef ipp.Ipp32f alpha
        
        cdef int w,h
        
        cdef ipp.Ipp32f v32f
        cdef ipp.Ipp8u  v8u
        
        cdef ipp.Ipp8u *im1, *im2 # current image
        cdef ipp.Ipp8u *bg_img, *std_img  # 8-bit background
        cdef ipp.Ipp32f *sum_image, *sq_image # FP accumulators
        cdef ipp.Ipp32f *mean_image, *std_image # FP background

        cdef ipp.IppiSize roi_sz
        
        cdef double x0, y0 # centroid
        cdef double orientation
        # end of IPP-requiring code

##        COORD_PORT = None
        n_bg_samples = 100
        n_rot_samples = 100*60 # 1 minute
        centroid_search_radius = 50
        alpha = 1.0/n_bg_samples
        # questionable optimization: speed up by eliminating namespace lookups
        # very questionable! doesn't the compiler resolve these? -JB
        cam_quit_event_isSet = globals['cam_quit_event'].isSet
        acquire_lock = globals['incoming_frames_lock'].acquire
        release_lock = globals['incoming_frames_lock'].release
        sleep = time.sleep
        bg_frame_number = -1
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

        # start of IPP-requiring code
        # allocate IPP memory

        # pre- and post-processed images of every frame
        im1=ipp.ippiMalloc_8u_C1( width, height, &im1_step )
        if im1==NULL:
            raise MemoryError("Error allocating memory by IPP")
        im2=ipp.ippiMalloc_8u_C1( width, height, &im2_step )
        if im2==NULL:
            raise MemoryError("Error allocating memory by IPP")

        # 8u background, std images
        bg_img=ipp.ippiMalloc_8u_C1( width, height, &bg_img_step )
        if bg_img==NULL:
            raise MemoryError("Error allocating memory by IPP")
        std_img=ipp.ippiMalloc_8u_C1( width, height, &std_img_step )
        if std_img==NULL:
            raise MemoryError("Error allocating memory by IPP")
        
        roi_sz.width = width
        roi_sz.height = height

        # 32f statistics and accumulator images for background collection
        sum_image=ipp.ippiMalloc_32f_C1( width, height,
                                        &sum_image_step )
        if sum_image==NULL:
            raise MemoryError("Error allocating memory by IPP")

        sq_image=ipp.ippiMalloc_32f_C1( width, height,
                                        &sq_image_step )
        if sq_image==NULL:
            raise MemoryError("Error allocating memory by IPP")

        mean_image=ipp.ippiMalloc_32f_C1( width, height,
                                        &mean_image_step )
        if mean_image==NULL:
            raise MemoryError("Error allocating memory by IPP")

        std_image=ipp.ippiMalloc_32f_C1( width, height,
                                        &std_image_step )
        if std_image==NULL:
            raise MemoryError("Error allocating memory by IPP")

        # image moment calculation initialization
        if c_fit_params.init_moment_state() != 0:
            raise RuntimeError("could not init moment state")

        # initialize background images
        CHK( ipp.ippiSet_8u_C1R(0,bg_img,bg_img_step, roi_sz))
        CHK( ipp.ippiSet_8u_C1R(0,std_img,std_img_step, roi_sz))

        CHK( ipp.ippiSet_32f_C1R(0.0,mean_image,mean_image_step, roi_sz))
        CHK( ipp.ippiSet_32f_C1R(0.0,std_image,std_image_step, roi_sz))

        # end of IPP-requiring code

        try:
            arena_error = arena_control.arena_initialize()
            have_arena_control = 1
            if arena_error != 0:
                print "WARNING: could not initialize arena control"
                have_arena_control = 0
        except NameError:
            have_arena_control = 0

        coord_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not cam_quit_event_isSet():
                # get pointer to data from camera driver
                self.cam.point_next_frame_blocking(&buf_ptr)
                
                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()
                framenumber=self.cam.get_last_framenumber()

                # now
                tval1=time_func()

                # start of IPP-requiring code
                # copy image to IPP memory
                for i from 0 <= i < height:
                    c_lib.memcpy(im1+im1_step*i,buf_ptr+width*i,width)

                # do background subtraction & find max pixel in ROI
                roi_sz.width = self.right-self.left+1
                roi_sz.height = self.top-self.bottom+1

                CHK( ipp.ippiAbsDiff_8u_C1R(
                    (bg_img + self.bottom*bg_img_step + self.left), bg_img_step,
                    (im1 + self.bottom*im1_step + self.left), im1_step,
                    (im2 + self.bottom*im2_step + self.left), im2_step, roi_sz))
                # (to avoid big moment arm:) if pixel < .8*max(pixel): pixel=0
                CHK( ipp.ippiThreshold_Val_8u_C1IR(
                    (im2 + self.bottom*im2_step + self.left), im2_step,
                    roi_sz, max_val*0.8, 0, ipp.ippCmpLess))
                CHK( ipp.ippiMaxIndx_8u_C1R(
                    (im2 + self.bottom*im2_step + self.left), im2_step,
                    roi_sz, &max_val, &index_x,&index_y))
#                CHK( ipp.ippiSqr_8u_C1IRSfs(
#                    (im2 + self.bottom*im2_step + self.left), im2_step,
#                    roi_sz, 5 ))

                if max_val < self.diff_threshold:
                    x0=-1
                    y0=-1
                else:
                    # compute centroid -=-=-=-=-=-=-=-=-=-=-=-=
                    
                    # index_x, index_y, centroid_search_radius,

                    c_fit_params.fit_params( &x0, &y0, &orientation,
                                roi_sz.width, roi_sz.height,
                                (im2 + self.bottom*im2_step + self.left), im2_step )
                    # note that x0 and y0 are now relative to the ROI origin
                    orientation = orientation + 1.57079632679489661923 # (pi/2)

                #print 'max_val %f (% 8.1f,% 8.1f)'%(max_val,x0,y0)
                
                # now
                tval2=time_func()
                
                buf_ptr=im2
                buf_ptr_step=im2_step
                # end of IPP-requiring code                    

                # allocate new numarray memory
                buf = <c_numarray._numarray>c_numarray.NA_NewArray(
                    NULL, c_numarray.tUInt8, 2,
                    height, width)
                heightwidth[0]=height
                heightwidth[1]=width

                # copy image to numarray
                for i from 0 <= i < height:
                    c_lib.memcpy(buf.data+width*i,
                                 buf_ptr+buf_ptr_step*i,
                                 width)

                # return camwire's buffer
                self.cam.unpoint_frame()

                points = [ (x0 + self.left,
                            y0 + self.bottom,
                            orientation) ]

##                if debug_isSet():
##                    print points

                if self.use_arena: # call out to arena feedback function
                    if have_arena_control:
                        arena_control.arena_update(
                            x0, y0, orientation, timestamp, framenumber )
                    else:
                        print 'ERROR: no arena control'
                    
                # make appropriate references to our copy of the data
                globals['most_recent_frame'] = buf
                globals['most_recent_frame_and_points'] = buf, points
                acquire_lock()
                globals['incoming_frames'].append(
                    (buf,timestamp,framenumber) ) # save it
                release_lock()

                #
                #
                # -=-=-=-=-= Clear BG image -=-=-=-=
                #
                #
                
                if clear_background_start_isSet():
                    clear_background_start_clear()
                    CHK( ipp.ippiSet_8u_C1R( 0,
                        (bg_img + self.bottom*bg_img_step + self.left),
                        bg_img_step, roi_sz))
                    CHK( ipp.ippiSet_8u_C1R( 0,
                        (std_img + self.bottom*std_img_step + self.left),
                        std_img_step, roi_sz))

                #
                #
                # -=-=-=-=-= Start collecting BG images -=-=-=-=
                #
                #
                
                if collect_background_start_isSet():
                    bg_frame_number=0
                    collect_background_start_clear()
                    # start of IPP-requiring code
                    CHK( ipp.ippiSet_32f_C1R( 0.0, 
                        (sum_image + self.bottom*sum_image_step/4 + self.left),
                        sum_image_step, roi_sz)) # divide by 4 because 32f = 4 bytes
                    CHK( ipp.ippiSet_32f_C1R( 0.0,
                        (sq_image + self.bottom*sq_image_step/4 + self.left),
                        sq_image_step, roi_sz))
                    # end of IPP-requiring code
                    
                #
                #
                # -=-=-=-=-= Collect BG images -=-=-=-=
                #
                #
                
                if bg_frame_number>=0:
                    
                    # start of IPP-requiring code
                    CHK( ipp.ippiAdd_8u32f_C1IR(
                        (im1 + self.bottom*im1_step + self.left), im1_step,
                        (sum_image + self.bottom*sum_image_step/4 + self.left),
                        sum_image_step, roi_sz))
                    CHK( ipp.ippiAddSquare_8u32f_C1IR(
                        (im1 + self.bottom*im1_step + self.left), im1_step,
                        (sq_image + self.bottom*sq_image_step/4 + self.left),
                        sq_image_step, roi_sz))
                    # end of IPP-requiring code
                    
                    bg_frame_number = bg_frame_number+1
                    if bg_frame_number>=n_bg_samples:

                        #
                        #
                        # -=-=-=-=-= Finish collecting BG images -=-=-=-=
                        #
                        #
                        
                        bg_frame_number=-1 # stop averaging frames
                        
                        # start of IPP-requiring code

                        # find mean
                        CHK( ipp.ippiMulC_32f_C1R(
                            (sum_image + self.bottom*sum_image_step/4 + self.left),
                            sum_image_step, 1.0/n_bg_samples,
                            (mean_image + self.bottom*mean_image_step/4 + self.left),
                            mean_image_step, roi_sz))
                        CHK( ipp.ippiConvert_32f8u_C1R(
                            (mean_image + self.bottom*mean_image_step/4 + self.left),
                            mean_image_step,
                            (bg_img + self.bottom*bg_img_step + self.left),
                            bg_img_step, roi_sz, ipp.ippRndNear ))

                        # find STD (use sum_image as temporary variable
                        CHK( ipp.ippiSqr_32f_C1R(
                            (mean_image + self.bottom*mean_image_step/4 + self.left),
                            mean_image_step,
                            (sum_image + self.bottom*sum_image_step/4 + self.left),
                            sum_image_step, roi_sz))
                        CHK( ipp.ippiMulC_32f_C1R(
                            (sq_image + self.bottom*sq_image_step/4 + self.left),
                            sq_image_step, 1.0/n_bg_samples,
                            (std_image + self.bottom*std_image_step/4 + self.left),
                            std_image_step, roi_sz))
                        CHK( ipp.ippiSub_32f_C1IR(
                            (sum_image + self.bottom*sum_image_step/4 + self.left),
                            sum_image_step,
                            (std_image + self.bottom*std_image_step/4 + self.left),
                            std_image_step, roi_sz))
                        CHK( ipp.ippiSqrt_32f_C1IR(
                            (std_image + self.bottom*std_image_step/4 + self.left),
                            std_image_step, roi_sz))
                        
##                        CHK(
##                            ipp.ippiMulC_32f_C1IR(3.0,std_image, std_image_step,sz))

                        CHK( ipp.ippiConvert_32f8u_C1R(
                            (std_image + self.bottom*std_image_step/4 + self.left),
                            std_image_step,
                            (std_img + self.bottom*std_img_step + self.left),
                            std_img_step, roi_sz, ipp.ippRndNear ))

                        # end of IPP-requiring code

                #
                #
                # -=-=-=-=-= Start finding center of rotation -=-=-=-=
                #
                #

                
                # start of IPP-requiring code
                if find_rotation_center_start_isSet():
                    find_rotation_center_start_clear()
                    rot_frame_number=0
                    arena_control.rotation_calculation_init()
                    c_fit_params.start_center_calculation( n_rot_samples )
                    
                #
                #
                # -=-=-=-=-= Collect data to find rotation center -=-=-=-=
                #
                #
                
                if rot_frame_number>=0:
                    c_fit_params.update_center_calculation( x0, y0, orientation )
                    arena_control.rotation_update()

                    rot_frame_number = rot_frame_number+1
                    if rot_frame_number>=n_rot_samples:

                        #
                        #
                        # -=-=-=-=-= Finish and calculate rotation center -=-=-=-=
                        #
                        #
                        
                        c_fit_params.end_center_calculation( &new_x_cent, &new_y_cent )
                        arena_control.rotation_calculation_finish( new_x_cent, new_y_cent )
                        rot_frame_number=-1 # stop averaging frames

                # end of IPP-requiring code
                
                tval3=time_func()

                latency1 = (tval1 - timestamp)*1000.0
                latency2 = (tval2 - timestamp)*1000.0
                latency3 = (tval3 - timestamp)*1000.0
##                print 'max_val %f i(% 5d,% 5d) f(% 8.1f,% 8.1f)'%(max_val,index_x,index_y,x0,y0)
##                print ('%.1f'%latency1).rjust(10),('%.1f'%latency2).rjust(10),('%.1f'%latency3).rjust(10)

##                if COORD_PORT is None:
##                    COORD_PORT = self.coord_port
##                else:
                if 1:
                    n_pts = len(points)
                    data = struct.pack('<dli',timestamp,framenumber,n_pts)
                    for i in range(n_pts):
                        data = data + struct.pack('<fff',*points[i])
                    coord_socket.sendto(data,
                                        (main_brain_hostname,self.coord_port))
##                    coord_socket.sendto(data,
##                                        (main_brain_hostname,COORD_PORT))
                sleep(1e-6) # yield processor
        finally:
            if have_arena_control:
                arena_control.arena_finish()

            # start of IPP-requiring code
            ipp.ippiFree(im1)
            ipp.ippiFree(im2)
            ipp.ippiFree(bg_img)
            ipp.ippiFree(std_img)
            ipp.ippiFree(sum_image)
            ipp.ippiFree(sq_image)
            ipp.ippiFree(mean_image)
            ipp.ippiFree(std_image)
            c_fit_params.free_moment_state()
            # end of IPP-requiring code

            globals['cam_quit_event'].set()
            globals['grab_thread_done'].set()

class FromMainBrainAPI( Pyro.core.ObjBase ):
    # "camera server"
    
    # ----------------------------------------------------------------
    #
    # Methods called locally
    #
    # ----------------------------------------------------------------
    
    def post_init(self, cam_id, main_brain, main_brain_lock, globals):
        if type(cam_id) != type(''):
            raise TypeError('cam_id must be a string')
        self.cam_id = cam_id
        self.main_brain = main_brain
        self.main_brain_lock = main_brain_lock
        self.globals = globals

    def listen(self,daemon):
        """thread mainloop"""
        self_cam_quit_event_isSet = self.globals['cam_quit_event'].isSet
        hr = daemon.handleRequests
        try:
            while not self_cam_quit_event_isSet():
                hr(0.1) # block on select for n seconds
                
        finally:
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
        """Trigger asynchronous send of image"""
        self.main_brain_lock.acquire()
        self.main_brain.set_image(self.cam_id, self.globals['most_recent_frame'])
        self.main_brain_lock.release()

    def get_most_recent_frame(self):
        """Return (synchronous) image"""
        return self.globals['most_recent_frame_and_points']

    def get_roi(self):
        """Return region of interest"""
        return self.globals['lbrt']

    def get_widthheight(self):
        """Return width and height of camera"""
        return self.globals['width'], self.globals['height']

    def is_ipp_enabled(self):
        result = False
        # start of IPP-requiring code
        result = True
        # end of IPP-requiring code
        return result

    def start_debug(self):
        self.globals['debug'].set()
        print '-='*20,'ENTERING DEBUG MODE'

    def stop_debug(self):
        self.globals['debug'].clear()
        print '-='*20,'LEAVING DEBUG MODE'

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
##            if self.globals['record_status']:
##                cmd,fly_movie,fly_movie_lock = self.globals['record_status']
            self.globals['record_status'] = None
        finally:
            self.globals['record_status_lock'].release()
            
        if cmd == 'save':
            fly_movie_lock.acquire()
            fly_movie.close()
            fly_movie_lock.release()
            print "stopping recording"
        else:
            # still saving data...
            #print "got stop recording command, but not recording!"
            pass

    def no_op(self):
        """used to test connection"""
        return None

    def quit(self):
        self.globals['cam_quit_event'].set()

    def collect_background(self):
        self.globals['collect_background_start'].set()

    def clear_background(self):
        self.globals['clear_background_start'].set()

    def get_diff_threshold(self):
        return self.globals['diff_threshold']

    def find_r_center(self):
        self.globals['find_rotation_center_start'].set()
    
cdef class App:
    cdef object globals
    cdef object cam_id
    cdef object from_main_brain_api
    
    cdef object main_brain
    cdef object main_brain_lock
    cdef int num_cams
    
    # MAX_GRABBERS = 3
    cdef Camera cam0
    cdef Camera cam1
    cdef Camera cam2
    
    cdef GrabClass grabber0
    cdef GrabClass grabber1
    cdef GrabClass grabber2
    
    def __init__(self):
        cdef Camera cam
        cdef GrabClass grabber

        MAX_GRABBERS = 3
        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------

        self.num_cams = c_cam_iface.cam_iface_get_num_cameras()
        print 'Number of cameras detected:', self.num_cams
        assert self.num_cams <= MAX_GRABBERS
        if self.num_cams == 0:
            return

        # ----------------------------------------------------------------
        #
        # Initialize network connections
        #
        # ----------------------------------------------------------------

        Pyro.core.initServer(banner=0,storageCheck=1)
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
        self.cam_id = []
        self.from_main_brain_api = []
        
        for cam_no in range(self.num_cams):
            cam = Camera(cam_no,30)

            height = cam.get_max_height()
            width = cam.get_max_width()

            if cam_no == 0:
                self.cam0=cam
            elif cam_no == 1:
                self.cam1=cam
            elif cam_no == 2:
                self.cam2=cam
            # add more if MAX_GRABBERS increases
                
            # ----------------------------------------------------------------
            #
            # Initialize "global" variables
            #
            # ----------------------------------------------------------------

            self.globals.append({})
            globals = self.globals[cam_no] # shorthand

            globals['incoming_frames']=[]
            globals['record_status']=None
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
            globals['record_status_lock'] = threading.Lock()

            globals['lbrt'] = 0,0,width-1,height-1
            globals['width'] = width
            globals['height'] = height

            # set defaults
            cam.set_camera_property(c_cam_iface.SHUTTER,300,0,0)
            cam.set_camera_property(c_cam_iface.GAIN,72,0,0)
            cam.set_camera_property(c_cam_iface.BRIGHTNESS,783,0,0)

            # get settings
            scalar_control_info = {}
            for name, enum_val in CAM_CONTROLS.items():
                current_value = cam.get_camera_property(enum_val)[0]
                tmp = cam.get_camera_property_range(enum_val)
                min_value = tmp[1]
                max_value = tmp[2]
                scalar_control_info[name] = (current_value, min_value, max_value)
            diff_threshold = 8.1
            scalar_control_info['initial_diff_threshold'] = diff_threshold

            # register self with remote server
            port = 9834 + cam_no # for local Pyro server
            self.main_brain_lock.acquire()
            self.cam_id.append(
                self.main_brain.register_new_camera(cam_no,
                                                    scalar_control_info,
                                                    port))
            coord_port = self.main_brain.get_coord_port(self.cam_id[cam_no])
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
            self.from_main_brain_api.append( FromMainBrainAPI() )
            self.from_main_brain_api[cam_no].post_init(
                                                       self.cam_id[cam_no],
                                                       self.main_brain,
                                                       self.main_brain_lock,
                                                       globals)
            URI=daemon.connect(self.from_main_brain_api[cam_no],'camera_server')
            print 'listening locally at',URI
            
            # create and start listen thread
            listen_thread=threading.Thread(target=self.from_main_brain_api[cam_no].listen,
                                           args=(daemon,))
            listen_thread.start()

            # ----------------------------------------------------------------
            #
            # start camera thread
            #
            # ----------------------------------------------------------------

            grabber = GrabClass()
            grabber.left = 0
            grabber.right = width-1
            grabber.bottom = 0
            grabber.top = height-1
            grabber.set_camera_and_coord_port(cam,coord_port)
            
            grabber.diff_threshold = diff_threshold
            # shadow grabber value
            globals['diff_threshold'] = grabber.diff_threshold
            
            grabber.use_arena = 0
            globals['use_arena'] = grabber.use_arena
            
            grab_thread=threading.Thread(target=grabber.grab_func,
                                         args=(globals,))
            cam.start_camera()  # start camera
            grab_thread.start() # start grabbing frames from camera

            print 'grab thread started'
            if cam_no == 0:
                self.grabber0=grabber
            elif cam_no == 1:
                self.grabber1=grabber
            elif cam_no == 2:
                self.grabber2=grabber
            print 'set grabber'
            # add more if MAX_GRABBERS increases

    def mainloop(self):
        cdef Camera cam
        cdef GrabClass grabber
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

                        if cam_no == 0:
                            cam=self.cam0
                        elif cam_no == 1:
                            cam=self.cam1
                        elif cam_no == 2:
                            cam=self.cam2

                        cam_id = self.cam_id[cam_no]
                        
                        now = time_func()

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
                        globals['incoming_frames_lock'].acquire()
                        len_if = len(globals['incoming_frames'])
                        if len_if:
                            n_frames[cam_no] = n_frames[cam_no]+len_if
                            grabbed_frames[cam_no].extend( globals['incoming_frames'] )
                            globals['incoming_frames'] = []
                        globals['incoming_frames_lock'].release()

                        # process asynchronous commands
                        self.main_brain_lock.acquire()
                        cmds=self.main_brain.get_and_clear_commands(cam_id)
                        self.main_brain_lock.release()
                        for key in cmds.keys():
                            if key == 'set':
                                for property_name,value in cmds['set'].iteritems():
                                    enum = CAM_CONTROLS[property_name]
                                    cam.set_camera_property(enum,value,0,0)
                            elif key == 'get_im': # low priority get image (for streaming)
                                self.from_main_brain_api[cam_no].send_most_recent_frame() # mimic call
                            else:
                                if cam_no == 0:
                                    grabber=self.grabber0
                                elif cam_no == 1:
                                    grabber=self.grabber1
                                elif cam_no == 2:
                                    grabber=self.grabber2
                                # add more if MAX_GRABBERS increases
                                if key == 'roi':
                                    l,b,r,t = cmds[key]
                                    grabber.left = l
                                    grabber.bottom = b
                                    grabber.right = r
                                    grabber.top = t
                                    globals['lbrt']=l,b,r,t
                                elif key == 'diff_threshold':
                                    grabber.diff_threshold = cmds[key]
                                    # shadow grabber value
                                    globals['diff_threshold'] = grabber.diff_threshold
                                elif key == 'use_arena':
                                    grabber.use_arena = cmds[key]
                                    globals['use_arena'] = grabber.use_arena
                                
                        # handle saving movie if needed
                        cmd=None
                        globals['record_status_lock'].acquire()
                        try:
                            if globals['record_status']:
                                cmd,fly_movie,fly_movie_lock = globals['record_status']
                        finally:
                            globals['record_status_lock'].release()

                        if len(grabbed_frames[cam_no]):
                            if cmd=='save':
                                #print 'saving %d frames'%(len(grabbed_frames[cam_no]),)
                                fly_movie_lock.acquire()
                                try:
                                    for frame,timestamp,framenumber in grabbed_frames[cam_no]:
                                        fly_movie.add_frame(frame,timestamp)
                                finally:
                                    fly_movie_lock.release()

                            grabbed_frames[cam_no] = []

                    time.sleep(0.05)

            finally:
##                self.globals[cam_no]['cam_quit_event'].set() # make sure other threads close
                self.main_brain_lock.acquire()
                for cam_id in self.cam_id:
                    self.main_brain.close(cam_id)
                self.main_brain_lock.release()
##                for cam_no in range(self.num_cams):
##                    self.globals[cam_no]['grab_thread_done'].wait() # block until thread is done...
##                    self.globals[cam_no]['listen_thread_done'].wait() # block until thread is done...
        except Pyro.errors.ConnectionClosedError:
            print 'unexpected connection closure...'

def main():
    app=App()
    app.mainloop()
        
