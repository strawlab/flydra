#emacs, this is -*-Python-*- mode

import threading
import time
import socket
import sys
import numarray as na
import Pyro.core, Pyro.errors
import FlyMovieFormat
import warnings
import struct


include "../cam_iface/src/pyx_cam_iface.pyx"
# CamIFaceError, Camera, c_numarray
#c_numarray.import_libnumarray()

cimport c_lib

# start of IPP-requiring code
cimport ipp

class IPPError(Exception):
    pass

cdef object IppStatus2str(ipp.IppStatus status):
    return 'IppStatus: %d'%status

cdef void CHK(ipp.IppStatus status) except *:
    if (status < ipp.ippStsNoErr):
        raise IPPError(IppStatus2str(status))
    elif (status > ipp.ippStsNoErr):
        warnings.warn(IppStatus2str(status))

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
    
    cdef void set_camera_and_coord_port(self, Camera cam, object coord_port):
        self.cam = cam
        self.coord_port = coord_port
        
    def grab_func(self,globals):
        cdef unsigned char* buf_ptr
        cdef c_numarray._numarray buf
        cdef int height
        cdef int buf_ptr_step, width
        cdef int collecting_background_frames
        cdef int bg_frame_number
        cdef int n_frames4stats
##        cdef double tval1, tval2, tval3, latency1, latency2
        cdef int i, j
        cdef double timestamp
        cdef long framenumber

        # start of IPP-requiring code
        cdef int index_x,index_y
        cdef ipp.Ipp32f max_val, std_val
        cdef int im1_step, im2_step, sum_image_step#, bg_step
        cdef int mean_image_step, std_image_step, sq_image_step
        cdef int im1_32f_step, im2_32f_step
        cdef int tmp1_step, tmp2_step, roi_step
        cdef int centroid_search_radius
        cdef int n_bg_samples
        cdef ipp.Ipp32f alpha
        
        cdef int w,h
        cdef int left, right, bottom, top
        
        cdef ipp.Ipp32f v32f
        cdef ipp.Ipp8u  v8u
        
        cdef ipp.Ipp8u *im1, *im2 # current image
        cdef ipp.Ipp8u *tmp1, *tmp2 # current image
        cdef ipp.Ipp8u* bg  # 8-bit background
        cdef ipp.Ipp32f *sum_image, *sq_image # FP background
        cdef ipp.Ipp32f *mean_image, *std_image # FP background
        cdef ipp.Ipp32f *im1_32f, *im2_32f, *roi_start
        cdef ipp.IppiSize sz, roi_sz
        cdef ipp.IppiPoint roi_offset
        cdef ipp.IppiMomentState_64f *pState
        cdef ipp.Ipp64f Mu00, Mu10, Mu01
        cdef float x0, y0 # centroid
        # end of IPP-requiring code

##        COORD_PORT = None
        n_bg_samples = 100
        centroid_search_radius = 10
        alpha = 1.0/n_bg_samples
        # questionable optimization: speed up by eliminating namespace lookups
        cam_quit_event_isSet = globals['cam_quit_event'].isSet
        acquire_lock = globals['incoming_frames_lock'].acquire
        release_lock = globals['incoming_frames_lock'].release
        sleep = time.sleep
        bg_frame_number = -1
        collect_background_start_isSet = globals['collect_background_start'].isSet
        collect_background_start_clear = globals['collect_background_start'].clear
        clear_background_start_isSet = globals['clear_background_start'].isSet
        clear_background_start_clear = globals['clear_background_start'].clear
        height = self.cam.get_max_height()
        width = self.cam.get_max_width()
        buf_ptr_step = width

        # start of IPP-requiring code
        # allocate IPP memory
        im1=ipp.ippiMalloc_8u_C1( width, height, &im1_step )
        if im1==NULL:
            raise MemoryError("Error allocating memory by IPP")
        im2=ipp.ippiMalloc_8u_C1( width, height, &im2_step )
        if im2==NULL:
            raise MemoryError("Error allocating memory by IPP")

        tmp1=ipp.ippiMalloc_8u_C1( width, height, &tmp1_step )
        if tmp1==NULL:
            raise MemoryError("Error allocating memory by IPP")
        tmp2=ipp.ippiMalloc_8u_C1( width, height, &tmp2_step )
        if tmp2==NULL:
            raise MemoryError("Error allocating memory by IPP")
        
##        bg=ipp.ippiMalloc_8u_C1( width, height, &bg_step )
##        if bg==NULL:
##            raise MemoryError("Error allocating memory by IPP")
        
        sz.width = width
        sz.height = height
        
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

        im1_32f=ipp.ippiMalloc_32f_C1( width, height,
                                       &im1_32f_step )
        if im1_32f==NULL:
            raise MemoryError("Error allocating memory by IPP")

        im2_32f=ipp.ippiMalloc_32f_C1( width, height,
                                       &im2_32f_step )
        if im2_32f==NULL:
            raise MemoryError("Error allocating memory by IPP")

        CHK(ipp.ippiMomentInitAlloc_64f(&pState, ipp.ippAlgHintFast))

##        CHK(
##            ipp.ippiSet_8u_C1R(0,bg,bg_step,sz))

        CHK(
            ipp.ippiSet_32f_C1R(0,mean_image,mean_image_step,sz))

        CHK(
            ipp.ippiSet_32f_C1R(0,std_image,std_image_step,sz))

        # end of IPP-requiring code

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
                    
                # do background subtraction
                CHK(
                    ipp.ippiConvert_8u32f_C1R(im1, im1_step,
                                              im1_32f, im1_32f_step,sz))
                
                CHK(
                    ipp.ippiAbsDiff_32f_C1R(mean_image, mean_image_step,
                                            im1_32f,im1_32f_step,
                                            im2_32f,im2_32f_step,sz))
                CHK(
                    ipp.ippiMaxIndx_32f_C1R(im2_32f,im2_32f_step,sz,
                                            &max_val,
                                            &index_x,&index_y))

##                roi_sz.width = 20
##                roi_sz.height = 20
##                CHK(
##                    ipp.ippiSet_32f_C1R(255.0,im2_32f+240*sz.width+320,im2_32f_step,roi_sz))
                
                CHK(
                    ipp.ippiConvert_32f8u_C1R(im2_32f,im2_32f_step,
                                              im2,im2_step,
                                              sz,
                                              ipp.ippRndNear))
                
                if max_val < globals['diff_threshold']:
                    x0=-1
                    y0=-1
                else:
                    # compute centroid -=-=-=-=-=-=-=-=-=-=-=-=

                    left   = index_x - centroid_search_radius
                    right  = index_x + centroid_search_radius
                    bottom = index_y - centroid_search_radius
                    top    = index_y + centroid_search_radius

                    if left < 0:
                        left = 0
                    if right >= width:
                        right = width-1
                    if bottom < 0:
                        bottom = 0
                    if top >= height:
                        top = height-1

                    roi_sz.width = right-left+1
                    roi_sz.height = top-bottom+1

                    roi_start = im2_32f + (im2_32f_step/4)*bottom + left
                    CHK( ipp.ippiMoments64f_32f_C1R( roi_start, im2_32f_step, roi_sz, pState))

                    roi_offset.x = left
                    roi_offset.y = bottom
                    CHK( ipp.ippiGetSpatialMoment_64f( pState, 0, 0, 0, roi_offset, &Mu00 ))
                    if Mu00 == 0.0:
                        x0=-1
                        y0=-1
                    else:
                        CHK( ipp.ippiGetSpatialMoment_64f( pState, 1, 0, 0, roi_offset, &Mu10 ))
                        CHK( ipp.ippiGetSpatialMoment_64f( pState, 0, 1, 0, roi_offset, &Mu01 ))
                        
                        x0=Mu10/Mu00
                        y0=Mu01/Mu00
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

                # copy image to numarray
                for i from 0 <= i < height:
                    c_lib.memcpy(buf.data+width*i,buf_ptr+buf_ptr_step*i,width)

                # XXX need to Py_DECREF(buf) ??

                # return camwire's buffer
                self.cam.unpoint_frame()

                points = [ (x0,y0) ]

##                points = [ (x0,y0),
##                           (index_x, index_y) ]
                
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
##                    CHK(
##                        ipp.ippiSet_8u_C1R(0,bg,bg_step,sz))
                    CHK(
                        ipp.ippiSet_32f_C1R(0,mean_image,mean_image_step,sz))

                    CHK(
                        ipp.ippiSet_32f_C1R(0,std_image,std_image_step,sz))

                #
                #
                # -=-=-=-=-= Start collecting BG images -=-=-=-=
                #
                #
                
                if collect_background_start_isSet():
                    bg_frame_number=0
                    collect_background_start_clear()
                    # start of IPP-requiring code
                    CHK(
                        ipp.ippiSet_32f_C1R(0.0,sum_image,sum_image_step,sz))
                    CHK(
                        ipp.ippiSet_32f_C1R(0.0,sq_image,sq_image_step,sz))
                    # end of IPP-requiring code
                    
                #
                #
                # -=-=-=-=-= Collect BG images -=-=-=-=
                #
                #
                
                if bg_frame_number>=0:
                    
                    # start of IPP-requiring code
                    CHK(
                        ipp.ippiAdd_8u32f_C1IR(im1,im1_step,
                                               sum_image,sum_image_step,
                                               sz))
                    CHK(
                        ipp.ippiAddSquare_8u32f_C1IR(im1,im1_step,
                                                     sq_image,sq_image_step,
                                                     sz))
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
                        CHK(
                            ipp.ippiMulC_32f_C1R(sum_image, sum_image_step,
                                                 1.0/n_bg_samples,
                                                 mean_image, mean_image_step,sz))

                        # find STD (use sum_image as temporary variable
                        CHK(
                            ipp.ippiSqr_32f_C1R(mean_image, mean_image_step,
                                                sum_image, sum_image_step, sz))
                        CHK(
                            ipp.ippiMulC_32f_C1R(sq_image, sq_image_step,
                                                 1.0/n_bg_samples,
                                                 std_image, std_image_step,sz))
                        CHK(
                            ipp.ippiSub_32f_C1IR(sum_image, sum_image_step,
                                                 std_image, std_image_step,sz))
                        CHK(
                            ipp.ippiSqrt_32f_C1IR(std_image, std_image_step,sz))
                        
##                        CHK(
##                            ipp.ippiMulC_32f_C1IR(3.0,std_image, std_image_step,sz))
                        
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
                        data = data + struct.pack('<ff',*points[i])
                    coord_socket.sendto(data,
                                        (main_brain_hostname,self.coord_port))
##                    coord_socket.sendto(data,
##                                        (main_brain_hostname,COORD_PORT))
                sleep(1e-6) # yield processor
        finally:
            # start of IPP-requiring code
            ipp.ippiFree(im1)
            ipp.ippiFree(bg)
            # end of IPP-requiring code

            globals['cam_quit_event'].set()
            globals['grab_thread_done'].set()

class FromMainBrainAPI( Pyro.core.ObjBase ):
    
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

    def is_ipp_enabled(self):
        result = False
        # start of IPP-requiring code
        result = True
        # end of IPP-requiring code
        return result

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

    def no_op(self):
        """used to test connection"""
        return None

    def quit(self):
        self.globals['cam_quit_event'].set()

    def collect_background(self):
        self.globals['collect_background_start'].set()

    def clear_background(self):
        self.globals['clear_background_start'].set()

    def set_diff_threshold(self,value):
        self.globals['diff_threshold'] = value

    def get_diff_threshold(self,value):
        return self.globals['diff_threshold']

cdef class App:
    cdef Camera cam0
    cdef Camera cam1
    cdef Camera cam2
    
    cdef object globals
    cdef object cam_id
    cdef object from_main_brain_api
    
    cdef object main_brain
    cdef object main_brain_lock
    cdef int num_cams
    
    def __init__(self):
        cdef Camera cam
        cdef GrabClass grabber
        
        # ----------------------------------------------------------------
        #
        # Setup cameras
        #
        # ----------------------------------------------------------------

        self.num_cams = c_cam_iface.cam_iface_get_num_cameras()
        print 'Number of cameras detected:', self.num_cams

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

            if cam_no == 0:
                self.cam0=cam
            elif cam_no == 1:
                self.cam1=cam
            elif cam_no == 2:
                self.cam2=cam
                
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
            globals['record_status_lock'] = threading.Lock()

            globals['diff_threshold'] = 8.1

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
            scalar_control_info['threshold'] = globals['diff_threshold']

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
            grabber.set_camera_and_coord_port(cam,coord_port)
            grab_thread=threading.Thread(target=grabber.grab_func,
                                         args=(globals,))
            cam.start_camera()  # start camera
            grab_thread.start() # start grabbing frames from camera

    def mainloop(self):
        cdef Camera cam
        # per camera variables
        grabbed_frames = []

        last_measurement_time = []
        last_return_info_check = []
        n_frames = []
        
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
        
