#emacs, this is -*-Python-*- mode

import threading
import time
import socket
import sys
import numarray as na
cimport c_numarray
import Pyro.core, Pyro.errors
import FlyMovieFormat
import warnings
cimport c_cam_iface
cimport c_lib

include "../cam_iface/src/pyx_cam_iface.pyx"

# start of IPP-requiring code
cimport ipp

class IPPError(Exception):
    pass

cdef object IppStatus2str(ipp.IppStatus status):
    return 'IppStatus: %d'%status

cdef void _ipp_check(ipp.IppStatus status):
    if (status < ipp.ippStsNoErr):
        raise IPPError(IppStatus2str(status))
    elif (status > ipp.ippStsNoErr):
        warnings.warn(IppStatus2str(status))

cdef object get_cpu_type():
    cdef ipp.IppCpuType t
    t=ipp.ippCoreGetCpuType()
    if t==ipp.ippCpuUnknown:
        s='ippCpuUnknown'
    elif t==ipp.ippCpuPP:
        s='ippCpuPP'
    elif t==ipp.ippCpuPMX:
        s='ippCpuPMX'
    elif t==ipp.ippCpuPPR:
        s='ippCpuPPR'
    elif t==ipp.ippCpuPII:
        s='ippCpuPII'
    elif t==ipp.ippCpuPIII:
        s='ippCpuPIII'
    elif t==ipp.ippCpuP4:
        s='ippCpuP4'
    elif t==ipp.ippCpuP4HT:
        s='ippCpuP4HT'
    elif t==ipp.ippCpuP4HT2:
        s='ippCpuP4HT2'
    elif t==ipp.ippCpuCentrino:
        s='ippCpuCentrino'
    elif t==ipp.ippCpuITP:
        s='ippCpuITP'
    elif t==ipp.ippCpuITP2:
        s='ippCpuITP2'
    return s
print 'CPU type:', get_cpu_type()


cdef void print_info_8u(ipp.Ipp8u* im, int im_step, ipp.IppiSize sz, object prefix):
    cdef ipp.Ipp32f minVal, maxVal
    cdef ipp.IppiPoint minIdx, maxIdx
    
    _ipp_check(
        ipp.ippiMinMaxIndx_8u_C1R( im, im_step, sz,
                                   &minVal, &maxVal,
                                   &minIdx, &maxIdx ))
    print prefix,'min: %f, max: %f, minIdx: %d,%d, maxIdx: %d,%d'%(minVal,maxVal,
                                                                   minIdx.x,minIdx.y,
                                                                   maxIdx.x,maxIdx.y)
    
# end of IPP-requiring code

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

cdef class GrabClass:
    cdef Camera cam

    cdef set_camera(self,Camera cam):
        self.cam = cam
        
    def grab_func(self,globals):
        cdef unsigned char* buf_ptr
        cdef c_cam_iface.CamContext* cc
        cdef c_numarray._numarray buf
        cdef int height
        cdef int buf_ptr_step, width
        cdef int collecting_background_frames
        cdef int bg_frame_number
        cdef int n_frames4stats
        cdef double tval1, tval2, tval3, latency1, latency2
        cdef int i, j

        # start of IPP-requiring code
        cdef int index_x,index_y
        cdef ipp.Ipp8u max_val
        cdef int im1_step, im2_step, avg_image_step, bg_step
        cdef int tmp1_step, tmp2_step
        cdef int n_bg_samples
        cdef ipp.Ipp32f alpha
        
        cdef ipp.Ipp32f v32f
        cdef ipp.Ipp8u  v8u
        
        cdef ipp.Ipp8u *im1, *im2 # current image
        cdef ipp.Ipp8u *tmp1, *tmp2 # current image
        cdef ipp.Ipp8u* bg  # 8-bit background
        cdef ipp.Ipp32f* avg_image # FP background
        cdef ipp.IppiSize sz
        # end of IPP-requiring code
        
        n_bg_samples = 100
        alpha = 1.0/n_bg_samples
        # questionable optimization: speed up by eliminating namespace lookups
        app_quit_event_isSet = globals['app_quit_event'].isSet
        acquire_lock = globals['incoming_frames_lock'].acquire
        release_lock = globals['incoming_frames_lock'].release
        sleep = time.sleep
        bg_frame_number = -1
        collect_background_start_isSet = globals['collect_background_start'].isSet
        collect_background_start_clear = globals['collect_background_start'].clear
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
        buf_ptr_step = im2_step

        tmp1=ipp.ippiMalloc_8u_C1( width, height, &tmp1_step )
        if tmp1==NULL:
            raise MemoryError("Error allocating memory by IPP")
        tmp2=ipp.ippiMalloc_8u_C1( width, height, &tmp2_step )
        if tmp2==NULL:
            raise MemoryError("Error allocating memory by IPP")
        
        bg=ipp.ippiMalloc_8u_C1( width, height, &bg_step )
        if bg==NULL:
            raise MemoryError("Error allocating memory by IPP")
        
        sz.width = width
        sz.height = height
        
        avg_image=ipp.ippiMalloc_32f_C1( width, height,
                                        &avg_image_step )
        if avg_image==NULL:
            raise MemoryError("Error allocating memory by IPP")

        _ipp_check(
            ipp.ippiSet_8u_C1R(0,bg,bg_step,sz))

        # end of IPP-requiring code
        
        try:
            while not app_quit_event_isSet():
                # get pointer to data from camera driver
                self.cam.point_next_frame_blocking(&buf_ptr)

                # get best guess as to when image was taken
                timestamp=self.cam.get_last_timestamp()

                # now
                tval1=time_func()

                # start of IPP-requiring code
                # copy image to IPP memory
                for i from 0 <= i < height:
                    c_lib.memcpy(im1+im1_step*i,buf_ptr+width*i,width)
                    
                # do background subtraction

                if 0:
                    _ipp_check(
                        ipp.ippiCompare_8u_C1R(im1,im1_step,
                                               bg,bg_step,
                                               tmp1,tmp1_step,
                                               sz, ipp.ippCmpLess))
                    _ipp_check(
                        ipp.ippiDivC_8u_C1IRSfs(2, tmp1, tmp1_step, sz, 1))
                    
                    _ipp_check(
                        ipp.ippiCompare_8u_C1R(im1,im1_step,
                                               bg,bg_step,
                                               im2,im2_step,
                                               sz, ipp.ippCmpGreater))
                    _ipp_check(
                        ipp.ippiAdd_8u_C1IRSfs(tmp1,tmp1_step,im2,im2_step,sz,1))
                elif 1:
                    _ipp_check(
                        ipp.ippiSub_8u_C1RSfs(bg,bg_step,
                                              im1,im1_step,
                                              im2,im2_step,
                                              sz,1))
                    
                _ipp_check(
                        ipp.ippiMaxIndx_8u_C1R(im2,im2_step,sz,
                                               &max_val,
                                               &index_x,&index_y))
                print "x,y:",index_x,index_y
                buf_ptr=im2
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

                # make appropriate references to our copy of the data
                globals['most_recent_frame'] = buf
                acquire_lock()
                globals['incoming_frames'].append(
                    (buf,timestamp) ) # save it
                release_lock()

                # now
                tval2=time_func()

                #
                #
                # -=-=-=-=-= Start collecting BG images -=-=-=-=
                #
                #
                
                if collect_background_start_isSet():
                    bg_frame_number=0
                    collect_background_start_clear()
                    # start of IPP-requiring code
                    _ipp_check(
                        ipp.ippiConvert_8u32f_C1R(im1,im1_step,
                                                  avg_image,avg_image_step,
                                                  sz))
                    # end of IPP-requiring code
                    
                #
                #
                # -=-=-=-=-= Collect BG images -=-=-=-=
                #
                #
                
                if bg_frame_number>=0:
                    
                    # start of IPP-requiring code

                    if 1:
                        _ipp_check(ipp.ippiAddWeighted_8u32f_C1IR(im1,im1_step,
                                                                  avg_image,avg_image_step,
                                                                  sz,alpha))
                    else:
                        _ipp_check(
                            ipp.ippiAdd_8u32f_C1IR(im1,im1_step,
                                                   avg_image,avg_image_step,
                                                   sz))
                    _ipp_check(
                        ipp.ippiConvert_32f8u_C1R(avg_image,avg_image_step,
                                                  bg,bg_step,
                                                  sz,
                                                  ipp.ippRndNear))
                    print 'conversion performed'
                    # end of IPP-requiring code
                    bg_frame_number = bg_frame_number+1
                    if bg_frame_number>=n_bg_samples:

                        #
                        #
                        # -=-=-=-=-= Finish collecting BG images -=-=-=-=
                        #
                        #
                        
                        bg_frame_number=-1 # stop averaging frames

                tval3=time_func()

                latency1 = (tval1 - timestamp)*1000.0
                latency2 = (tval2 - timestamp)*1000.0
                latency3 = (tval3 - timestamp)*1000.0
                print ('%.1f'%latency1).rjust(10),('%.1f'%latency2).rjust(10),('%.1f'%latency3).rjust(10)
                sleep(0.00001) # yield processor
        finally:
            # start of IPP-requiring code
            ipp.ippiFree(im1)
            ipp.ippiFree(bg)
            # end of IPP-requiring code

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
        
        self.globals['image_offset'] = 127
        
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
    app=App()
    app.mainloop()
        
