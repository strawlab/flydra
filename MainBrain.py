# $Id$
import threading
import time
import socket
import Pyro.core
import sys
import os
import copy
import socket
from flydra.reconstruct import Reconstructor
import struct
import math
import numarray as nx

import struct
import math
from numarray.ieeespecial import nan

Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!

Pyro.config.PYRO_TRACELEVEL = 1
Pyro.config.PYRO_USER_TRACELEVEL = 1
Pyro.config.PYRO_DETAILED_TRACEBACK = 0
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

# globals:
UDP_ports=[]

calib_data_lock = threading.Lock()
calib_IdMat = []
calib_points = []

realtime_coord_dict={}
realtime_coord_dict_lock=threading.Lock()

SAVE_2D_DATA = False
SAVE_2D_FMT = '<Bidddd'
SAVE_2D_CAMS = 0
SAVE_GLOBALS_LOCK = threading.Lock()
SAVE_GLOBALS = {}
save_2d_data_fd=None
save_2d_data_lock=threading.Lock()

SAVE_3D_DATA = False
SAVE_3D_FMT = '<iddd'
save_3d_data1={}
save_3d_data1_lock=threading.Lock()
save_3d_data2={}
save_3d_data2_lock=threading.Lock()

fastest_realtime_data=None
best_realtime_data=None

RESET_FRAMENUMBER_DURATION=1.0 # seconds

try:
    hostname = socket.gethostbyname('mainbrain')
except:
    hostname = socket.gethostbyname(socket.gethostname())

##try:
##    projector_hostname = socket.gethostbyname('projector')
##except:
##    projector_hostname = socket.gethostbyname(socket.gethostname())
projector_hostname = socket.gethostbyname(socket.gethostname())
projector_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FASTEST_DATA_PORT = 28931
BEST_DATA_PORT = 28932

def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def get_fastest_realtime_data():
    global fastest_realtime_data
    data = fastest_realtime_data
    fastest_realtime_data = None
    return data 

def get_best_realtime_data():
    global best_realtime_data
    data = best_realtime_data
    best_realtime_data = None
    return data 

def DEBUG():
    print 'line',sys._getframe().f_back.f_lineno,', thread', threading.currentThread()
    #for t in threading.enumerate():
    #    print '   ',t

class CoordReceiver(threading.Thread):
    def __init__(self,cam_id,main_brain):
        global SAVE_2D_CAMS, SAVE_GLOBALS, SAVE_GLOBALS_LOCK
        
        self.cam_id = cam_id
        self.hack_cam_no = SAVE_2D_CAMS
        print self.cam_id,'assigned to hack_cam_no',self.hack_cam_no
        SAVE_GLOBALS_LOCK.acquire()
        SAVE_GLOBALS[self.cam_id]={}
        SAVE_GLOBALS[self.cam_id]['cam_no']=self.hack_cam_no
        SAVE_GLOBALS_LOCK.release()

        SAVE_2D_CAMS += 1
        self.main_brain = main_brain
        self.last_timestamp=-10.0
        self.reconstructor = None

        # set up threading stuff
        self.quit_event = threading.Event()
        name = 'CoordReceiver for %s'%self.cam_id
        threading.Thread.__init__(self,name=name)

        # find UDP port number
        if len(UDP_ports)>0:
            self.port = max(UDP_ports)+1
        else:
            self.port = 34813
        UDP_ports.append( self.port )

        # create and bind socket to listen to
        self.recSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recSocket.bind((hostname, self.port))

    def get_port(self):
        return self.port

    def set_reconstructor(self,r):
        # This is called on a running thread...
        self.reconstructor = r

    def quit(self):
        self.quit_event.set()
        
        # send packet to wake listener and allow thread to quit
        tmp_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tmp_socket.sendto(struct.pack('<dlii',0.0,-1,-1,-1),(hostname,self.port))
    
    def run(self):
        global fastest_realtime_data, best_realtime_data
        global calib_IdMat, calib_points, calib_data_lock
        global SAVE_2D_CAMS, SAVE_2D_DATA, SAVE_GLOBALS, SAVE_GLOBALS_LOCK
        global save_2d_data_fd, save_2d_data_lock

        header_fmt = '<dli'
        header_size = struct.calcsize(header_fmt)
        pt_fmt = '<fff'
        pt_size = struct.calcsize(pt_fmt)
        while not self.quit_event.isSet():
            data, addr = self.recSocket.recvfrom(1024)
            
            header = data[:header_size]
            timestamp, framenumber, n_pts = struct.unpack(header_fmt,header)
            start=header_size
            points = []
            for i in range(n_pts):
                end=start+pt_size
                x,y,slope = struct.unpack(pt_fmt,data[start:end])
                points.append( (x,y,slope) )
                start=end

            if 0:
                now = time.time()
                latency = now-timestamp
                print (' '*self.hack_cam_no*10)+('% 11.1f'%( (now*1000.0)%1000.0, ))+(' '*(SAVE_2D_CAMS-self.hack_cam_no-1)*10),
                print '% 6.1f'%((timestamp*1000)%1000.0,),
                print '% 6.1f'%((latency*1000)%1000.0,)
            

            if framenumber==-1:
                continue # leftover in socket buffer from last run??
            
            if timestamp-self.last_timestamp > RESET_FRAMENUMBER_DURATION:
                self.framenumber_offset = framenumber
                if self.last_timestamp != -10.0:
                    print self.cam_id,'synchronized'
                    SAVE_GLOBALS_LOCK.acquire()
                    SAVE_GLOBALS[self.cam_id]['frame0']=timestamp
                    SAVE_GLOBALS_LOCK.release()
                else:
                    print self.cam_id,'first 2D coordinates received'

            self.last_timestamp=timestamp
            corrected_framenumber = framenumber-self.framenumber_offset

            if SAVE_2D_DATA:
                buf = struct.pack(SAVE_2D_FMT,
                                  self.hack_cam_no,
                                  corrected_framenumber,
                                  timestamp,
                                  points[0][0],
                                  points[0][1],
                                  points[0][2],
                                  )
                save_2d_data_lock.acquire()
                save_2d_data_fd.write( buf )
                save_2d_data_lock.release()

            realtime_coord_dict_lock.acquire()
            # clean up old frame records to save RAM
            if len(realtime_coord_dict)>100:
                k=realtime_coord_dict.keys()
                k.sort()
                for ki in k[:-50]:
                    del realtime_coord_dict[ki]
                    
            # save new frame record
            cur_framenumber_dict=realtime_coord_dict.setdefault(corrected_framenumber,{})
            # save x,y, not slope
            cur_framenumber_dict[self.cam_id]=points[0][:2] # XXX for now, only attempt 3D reconstruction of 1st point

            # make thread-local copy of results if 3D reconstruction possible
            if len(cur_framenumber_dict)>=2:
                data_dict = cur_framenumber_dict.copy()
            else:
                data_dict = None
            realtime_coord_dict_lock.release()

            if data_dict is not None:
                
                # do 3D reconstruction -=-=-=-=-=-=-=-=
                if self.reconstructor is not None:
                    d2 = {}
                    cams_in_count = 0
                    for cam_id, PT in data_dict.iteritems():
                        cams_in_count += 1
                        if PT[0] + 1 > 1e-6: # only use found points
                            d2[cam_id] = PT
                    if len(d2) >=2:
                        X = self.reconstructor.find3d(d2.items())
                        find3d_time = time.time()
                        x,y,z=X
                        if len(d2) == 2 and SAVE_3D_DATA:
                            save_3d_data1_lock.acquire()
                            save_3d_data1[corrected_framenumber]=x,y,z,find3d_time,2
                            save_3d_data1_lock.release()
                        data_packet = struct.pack('<fff',x,y,z)
                        try:
                            projector_socket.sendto(data_packet,
                                                    (projector_hostname,FASTEST_DATA_PORT))
                        except x:
                            print 'WARNING: could not send 3d point data to projector:'
                            print x.__class__, x
                            print
                        fastest_realtime_data = X
                        if cams_in_count == self.main_brain.get_num_cams():
                            best_realtime_data = X
                            try:
                                projector_socket.sendto(data_packet,
                                                        (projector_hostname,BEST_DATA_PORT))
                            except x:
                                print 'WARNING: could not send 3d point data to projector:'
                                print x.__class__, x
                                print
                            
                        if SAVE_3D_DATA:
                            save_3d_data2_lock.acquire()
                            save_3d_data2[corrected_framenumber]=x,y,z,find3d_time,len(d2)
                            save_3d_data2_lock.release()
                            
                # save calibration data -=-=-=-=-=-=-=-=
                if self.main_brain.currently_calibrating.isSet():
                    if len(data_dict) == self.main_brain.get_num_cams():
                        k = data_dict.keys()
                        k.sort()
                        ids = []
                        save_points = []
                        for cam_id in k:
                            pt = data_dict[cam_id]
                            if pt[0]+1<1e-6: # pt[0] == -1
                                save_pt = nan, nan, nan
                                id = 0
                            else:
                                save_pt = pt[0], pt[1], 1.0
                                id = 1
                            ids.append( id )
                            save_points.extend( save_pt )
                        # we now have data from all cameras
                        calib_data_lock.acquire()
                        calib_IdMat.append( ids )
                        calib_points.append( save_points )
                        calib_data_lock.release()
                    
           # XXX hack? make data available via cam_dict
            cam_dict = self.main_brain.remote_api.cam_info[self.cam_id]
            cam_dict['lock'].acquire()
            cam_dict['points']=points
            cam_dict['lock'].release()
        UDP_ports.remove( self.port )

class MainBrain(object):
    """Handle all camera network stuff and interact with application"""

    class RemoteAPI(Pyro.core.ObjBase):

        # ----------------------------------------------------------------
        #
        # Methods called locally
        #
        # ----------------------------------------------------------------

        def post_init(self, main_brain):
            """call after __init__"""
            # let Pyro handle __init__
            self.cam_info = {}
            self.cam_info_lock = threading.Lock()
            #self.cam_info_lock = LockProxy()
            self.changed_cam_lock = threading.Lock()
            self.no_cams_connected = threading.Event()
            self.no_cams_connected.set()
            self.changed_cam_lock.acquire()
            self.new_cam_ids = []
            self.old_cam_ids = []
            self.changed_cam_lock.release()
            self.main_brain = main_brain
            
            # threading control locks
            self.quit_now = threading.Event()
            self.thread_done = threading.Event()

        def external_get_and_clear_pending_cams(self):
            self.changed_cam_lock.acquire()
            new_cam_ids = self.new_cam_ids
            self.new_cam_ids = []
            old_cam_ids = self.old_cam_ids
            self.old_cam_ids = []
            self.changed_cam_lock.release()
            return new_cam_ids, old_cam_ids

        def external_get_cam_ids(self):
            self.cam_info_lock.acquire()
            cam_ids = self.cam_info.keys()
            self.cam_info_lock.release()
            cam_ids.sort()
            return cam_ids

        def external_get_info(self, cam_id):
            self.cam_info_lock.acquire()
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            scalar_control_info = copy.deepcopy(cam['scalar_control_info'])
            fqdn = cam['fqdn']
            port = cam['port']
            cam_lock.release()
            self.cam_info_lock.release()
            return scalar_control_info, fqdn, port

        def external_get_image_fps_points(self, cam_id):
            self.cam_info_lock.acquire()
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            image = cam['image']
            cam['image'] = None
            fps = cam['fps']
            cam['fps'] = None
            points = cam['points'][:]
            cam_lock.release()
            self.cam_info_lock.release()            
            return image, fps, points

        def external_send_set_camera_property( self, cam_id, property_name, value):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands'].setdefault('set',{})[property_name]=value
            old_value = cam['scalar_control_info'][property_name]
            if type(old_value) == tuple and type(value) == int:
                # brightness, gain, shutter
                cam['scalar_control_info'][property_name] = (value, old_value[1], old_value[2])
            else:
                cam['scalar_control_info'][property_name] = value
            cam_lock.release()
            self.cam_info_lock.release()

        def external_request_image_async(self, cam_id):
            self.cam_info_lock.acquire()
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['get_im']=None
            cam_lock.release()
            self.cam_info_lock.release()            

        def external_start_recording( self, cam_id, filename):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['start_recording']=filename
            cam_lock.release()
            self.cam_info_lock.release()

        def external_stop_recording( self, cam_id):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['stop_recording']=None
            cam_lock.release()
            self.cam_info_lock.release()

        def external_quit( self, cam_id):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['quit']=True
            cam_lock.release()
            self.cam_info_lock.release()

        def external_set_use_arena( self, cam_id, value):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['use_arena']=value
            cam_lock.release()
            self.cam_info_lock.release()

        def external_find_r_center( self, cam_id):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['find_r_center']=None
            cam_lock.release()
            self.cam_info_lock.release()

        def external_collect_background( self, cam_id):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['collect_bg']=None
            cam_lock.release()
            self.cam_info_lock.release()

        def external_clear_background( self, cam_id):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['clear_bg']=None
            cam_lock.release()
            self.cam_info_lock.release()

        def external_set_debug( self, cam_id, value):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['debug']=value
            cam_lock.release()
            self.cam_info_lock.release()

        # --- thread boundary -----------------------------------------

        def listen(self,daemon):
            """thread mainloop"""
            quit_now_isSet = self.quit_now.isSet
            hr = daemon.handleRequests
            while not quit_now_isSet():
                hr(0.1) # block on select for n seconds
                self.cam_info_lock.acquire()                
                cam_ids = self.cam_info.keys()
                self.cam_info_lock.release()
                for cam_id in cam_ids:
                    self.cam_info_lock.acquire()
                    connected = self.cam_info[cam_id]['caller'].connected
                    self.cam_info_lock.release()                    
                    if not connected:
                        print 'main_brain WARNING: lost camera',cam_id
                        self.close(cam_id)
            self.thread_done.set()
                                             
        # ----------------------------------------------------------------
        #
        # Methods called remotely from cameras
        #
        # These all get called in their own thread.  Don't call across
        # the thread boundary without using locks, especially to GUI
        # or OpenGL.
        #
        # ----------------------------------------------------------------

        def register_new_camera(self,cam_no,scalar_control_info,port):
            """register new camera, return cam_id (caller: remote camera)"""

            caller= self.daemon.getLocalStorage().caller # XXX Pyro hack??
            caller_addr= caller.addr
            caller_ip, caller_port = caller_addr
            fqdn = socket.getfqdn(caller_ip)
        
            #cam_id = '%s:%d:%d'%(fqdn,cam_no,caller_port)
            cam_id = '%s:%d'%(fqdn,cam_no)
            print 'cam_id',cam_id,'connected'
            
            coord_receiver = CoordReceiver(cam_id,self.main_brain)
            self.cam_info_lock.acquire()            
            self.cam_info[cam_id] = {'commands':{}, # command queue for cam
                                     'lock':threading.Lock(), # prevent concurrent access
                                     'image':None,  # most recent image from cam
                                     'fps':None,    # most recept fps from cam
                                     'points':[], # 2D image points
                                     'caller':caller,    # most recept fps from cam
                                     'scalar_control_info':scalar_control_info,
                                     'fqdn':fqdn,
                                     'port':port,
                                     'coord_receiver':coord_receiver,
                                     }
            self.cam_info_lock.release()
        
            coord_receiver.start()
            
            self.no_cams_connected.clear()
            
            self.changed_cam_lock.acquire()
            self.new_cam_ids.append(cam_id)
            self.changed_cam_lock.release()
            
            return cam_id

        def set_image(self,cam_id,image):
            """set most recent image (caller: remote camera)"""
            self.cam_info_lock.acquire()
            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            self.cam_info[cam_id]['image'] = image
            cam_lock.release()
            self.cam_info_lock.release()            

        def set_fps(self,cam_id,fps):
            """set most recent fps (caller: remote camera)"""
            self.cam_info_lock.acquire()
            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            self.cam_info[cam_id]['fps'] = fps
            cam_lock.release()
            self.cam_info_lock.release()            

        def get_and_clear_commands(self,cam_id):
            self.cam_info_lock.acquire()
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cmds = cam['commands']
            cam['commands'] = {}
            cam_lock.release()
            self.cam_info_lock.release()
            return cmds

        def get_coord_port(self,cam_id):
            """Send UDP port number to which camera should send realtime data"""
            self.cam_info_lock.acquire()
            port = self.cam_info[cam_id]['coord_receiver'].get_port()
            self.cam_info_lock.release()
            
            return port

        def close(self,cam_id):
            """gracefully say goodbye (caller: remote camera)"""
            self.cam_info_lock.acquire()
            self.cam_info[cam_id]['coord_receiver'].quit()
            del self.cam_info[cam_id]['coord_receiver']
            del self.cam_info[cam_id]
            if not len(self.cam_info):
                self.no_cams_connected.set()
            
            self.changed_cam_lock.acquire()
            self.old_cam_ids.append(cam_id)
            self.changed_cam_lock.release()
            self.cam_info_lock.release()
            
    def __init__(self):
        Pyro.core.initServer(banner=0)

        port = 9833

        # start Pyro server
        daemon = Pyro.core.Daemon(host=hostname,port=port)
        remote_api = MainBrain.RemoteAPI(); remote_api.post_init(self)
        URI=daemon.connect(remote_api,'main_brain')

        # create (but don't start) listen thread
        self.listen_thread=threading.Thread(target=remote_api.listen,
                                            name='RemoteAPI-Thread',
                                            args=(daemon,))

        self.remote_api = remote_api

        self._new_camera_functions = []
        self._old_camera_functions = []

        self.last_requested_image = {}
        self.pending_requests = {}
        self.last_set_param_time = {}
        
        self.num_cams = 0
        self.set_new_camera_callback(self.IncreaseCamCounter)
        self.set_old_camera_callback(self.DecreaseCamCounter)
        self.currently_calibrating = threading.Event()

    def IncreaseCamCounter(self,*args):
        print 'new camera:',args
        self.num_cams += 1

    def DecreaseCamCounter(self,*args):
        print 'old camera:',args
        self.num_cams -= 1

    def get_num_cams(self):
        return self.num_cams

    def get_widthheight(self, cam_id):
        sci, fqdn, port = self.remote_api.external_get_info(cam_id)
        w = sci['width']
        h = sci['height']
        return w,h

    def get_roi(self, cam_id):
        sci, fqdn, port = self.remote_api.external_get_info(cam_id)
        lbrt = sci['roi']
        return lbrt

    def get_all_params(self):
        cam_ids = self.remote_api.external_get_cam_ids()
        all = {}
        for cam_id in cam_ids:
            sci, fqdn, port = self.remote_api.external_get_info(cam_id)
            all[cam_id] = sci
        return all

    def Save3dData(self,fast_filename='raw_data_3d_fast.dat',best_filename='raw_data_3d_best.dat'):
        for typ in ['fast','best']:
            if typ == 'fast':
                fname = fast_filename
                lock = save_3d_data1_lock
                dikt = save_3d_data1
            elif typ == 'best':
                fname = best_filename
                lock = save_3d_data2_lock
                dikt = save_3d_data2
            fullpath = os.path.abspath(fname)
            print 'saving %s 3d data to "%s"...'%(typ,fullpath)
            fd=open(fname,'wb')
            lock.acquire()
            dd=dikt.copy()
            lock.release()

            keys=dd.keys()
            keys.sort()
            for k in keys:
                fd.write('%d %s\n'%(k,' '.join(map( str, dd[k]))))
            fd.close()
            print '  done'

    def SaveGlobals(self,filename='camera_data.dat'):
        fullpath = os.path.abspath(filename)
        print 'saving globals to',fullpath
        fd=open(filename,'wb')
        SAVE_GLOBALS_LOCK.acquire()
        dd=SAVE_GLOBALS.copy()
        SAVE_GLOBALS_LOCK.release()
        cam_ids=dd.keys()
        cam_ids.sort()
        for cam_id in cam_ids:
            fd.write('%s %d %s\n'%(cam_id,SAVE_GLOBALS[cam_id]['cam_no'],
                                   repr(SAVE_GLOBALS[cam_id]['frame0'])))
        fd.close()
        print 'globals saved'

    def start_listening(self):
        # start listen thread
        self.listen_thread.start()

    def set_new_camera_callback(self,handler):
        self._new_camera_functions.append(handler)

    def set_old_camera_callback(self,handler):
        self._old_camera_functions.append(handler)

    def start_calibrating(self, calib_dir):
        self.calibration_cam_ids = self.remote_api.external_get_cam_ids()
        self.calib_dir = calib_dir
        self.currently_calibrating.set()

    def stop_calibrating(self):
        global calib_IdMat, calib_points, calib_data_lock
        self.currently_calibrating.clear()
        
        cam_ids = self.remote_api.external_get_cam_ids()
        if len(cam_ids) != len(self.calibration_cam_ids):
            raise RuntimeError("Number of cameras changed during calibration")

        for cam_id in cam_ids:
            if cam_id not in self.calibration_cam_ids:
                raise RuntimeError("Cameras changed during calibration")

        cam_ids.sort()
                                
        calib_data_lock.acquire()
        
        IdMat = calib_IdMat
        calib_IdMat = []
        
        points = calib_points
        calib_points = []
        
        calib_data_lock.release()

        IdMat = nx.transpose(IdMat)
        points = nx.transpose(points)
        print 'saving %d points to %s'%(len(points),self.calib_dir)
        save_ascii_matrix(os.path.join(self.calib_dir,'IdMat.dat'),IdMat)
        save_ascii_matrix(os.path.join(self.calib_dir,'points.dat'),points)
        cam_ids = self.remote_api.external_get_cam_ids()
        Res = []
        for cam_id in cam_ids:
            sci, fqdn, port = self.remote_api.external_get_info(cam_id)
            width, height = self.get_widthheight(cam_id)
            Res.append( [width,height] )
        Res = nx.array( Res )
        save_ascii_matrix(os.path.join(self.calib_dir,'Res.dat'),Res)
        
        fd = open(os.path.join(self.calib_dir,'camera_order.txt'),'w')
        for cam_id in cam_ids:
            fd.write('%s\n'%cam_id)
        fd.close()

    def service_pending(self):
        new_cam_ids, old_cam_ids = self.remote_api.external_get_and_clear_pending_cams()

        for cam_id in new_cam_ids:
            if cam_id in old_cam_ids:
                continue # inserted and then removed
            scalar_control_info, fqdn, port = self.remote_api.external_get_info(cam_id)
            for new_cam_func in self._new_camera_functions:
                new_cam_func(cam_id,scalar_control_info,(fqdn,port))

        for cam_id in old_cam_ids:
            for old_cam_func in self._old_camera_functions:
                old_cam_func(cam_id)

    def get_last_image_fps(self, cam_id):
        return self.remote_api.external_get_image_fps_points(cam_id)

    def close_camera(self,cam_id):
        self.remote_api.external_quit( cam_id )

    def set_use_arena(self, cam_id, value):
        self.remote_api.external_set_use_arena( cam_id, value)

    def set_debug_mode(self, cam_id, value):
        self.remote_api.external_set_debug( cam_id, value)

    def collect_background(self,cam_id):
        self.remote_api.external_collect_background(cam_id)

    def clear_background(self,cam_id):
        self.remote_api.external_clear_background(cam_id)

    def find_r_center(self,cam_id):
        self.remote_api.external_find_r_center(cam_id)

    def send_set_camera_property(self, cam_id, property_name, value):
        self.remote_api.external_send_set_camera_property( cam_id, property_name, value)

    def request_image_async(self, cam_id):
        self.remote_api.external_request_image_async(cam_id)

    def start_recording(self, cam_id,filename):
        self.remote_api.external_start_recording( cam_id, filename)

    def stop_recording(self, cam_id):
        self.remote_api.external_stop_recording( cam_id)

    def quit(self):
        # XXX ----- non-isolated calls to remote_api being done ----
        # this may be called twice: once explicitly and once by __del__
        self.remote_api.cam_info_lock.acquire()
        cam_ids = self.remote_api.cam_info.keys()
        self.remote_api.cam_info_lock.release()
        
        for cam_id in cam_ids:
            try:
                self.close_camera(cam_id)
            except Pyro.errors.ProtocolError:
                # disconnection results in error
                print 'ignoring exception on',cam_id
                pass
        self.remote_api.no_cams_connected.wait(2.0)
        self.remote_api.quit_now.set() # tell thread to finish
        self.remote_api.thread_done.wait(0.5) # wait for thread to finish
        if not self.remote_api.no_cams_connected.isSet():
            cam_ids = self.remote_api.cam_info.keys()
            print 'cameras failed to quit cleanly: %s'%str(cam_ids)
            #raise RuntimeError('cameras failed to quit cleanly: %s'%str(cam_ids))

    def load_calibration(self,dirname):
        cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = Reconstructor(calibration_dir=dirname)

        # XXX this is naughty accessing remote_api
        self.remote_api.cam_info_lock.acquire()
        for cam_id in cam_ids:
            port = self.remote_api.cam_info[cam_id]['coord_receiver'].set_reconstructor(self.reconstructor)
        self.remote_api.cam_info_lock.release()
    
    def __del__(self):
        self.quit()

    def set_all_cameras_debug_mode( self, value ):
        cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_set_debug( cam_id, value)

    def get_save_2d_data(self):
        global SAVE_2D_DATA
        return SAVE_2D_DATA
    def set_save_2d_data(self,value):
        global SAVE_2D_DATA, save_2d_data_fd

        if value:
            if save_2d_data_fd is None:
                save_2d_data_fd = open('raw_data.dat','wb')
        SAVE_2D_DATA = value
        
    save_2d_data = property( get_save_2d_data, set_save_2d_data )
    
    def clear_2d_data(self):
        global save_2d_data_fd, save_2d_data_lock

        save_2d_data_lock.acquire()
        save_2d_data_fd.seek(0)
        save_2d_data_lock.release()

    def get_save_3d_data(self):
        global SAVE_3D_DATA
        return SAVE_3D_DATA
    def set_save_3d_data(self,value):
        global SAVE_3D_DATA
        SAVE_3D_DATA=value
    save_3d_data = property( get_save_3d_data, set_save_3d_data )
    
    def clear_3d_data(self):
        global save_3d_data1, save_3d_data1_lock, save_3d_data2, save_3d_data2_lock
        
        save_3d_data1_lock.acquire()
        save_3d_data1 = {}
        save_3d_data1_lock.release()

        save_3d_data2_lock.acquire()
        save_3d_data2 = {}
        save_3d_data2_lock.release()

