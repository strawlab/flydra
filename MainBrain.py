# $Id$
import threading, time, socket, select, sys, os, copy, struct, math
import sets
import Pyro.core
from flydra.reconstruct import Reconstructor
import numarray as nx
from numarray.ieeespecial import nan, inf

Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here...

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

calib_data_lock = threading.Lock()
calib_IdMat = []
calib_points = []

fastest_realtime_data=None
best_realtime_data=None

try:
    hostname = socket.gethostbyname('mainbrain')
except:
    hostname = socket.gethostbyname(socket.gethostname())

downstream_hostnames = []

use_projector = False
realtime_display = True

if use_projector:
    projector_hostname = socket.gethostbyname('projector')
    downstream_hostnames.append(projector_hostname)
if realtime_display:
    realtime_display_hostname = hostname
    downstream_hostnames.append(realtime_display_hostname)

FASTEST_DATA_PORT = 28931
BEST_DATA_PORT = 28932

if len(downstream_hostnames):
    outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    def __init__(self,main_brain):
        
        self.main_brain = main_brain
        
        self.cam_ids = []
        self.UDP_ports = []
        self.absolute_cam_nos = []
        self.last_timestamps = []
        self.listen_sockets = []
        self.framenumber_offsets = []
        
        self.reconstructor = None

        self.all_data_lock = threading.Lock()
        self.quit_event = threading.Event()
        
        self.max_absolute_cam_nos = -1
        self.RESET_FRAMENUMBER_DURATION=1.0 # seconds
        
        self.general_save_info = {}
        self.save_2d_data_fd = None
        self.save_3d_data_fastest = None
        self.save_3d_data_best = None
        
        name = 'CoordReceiver thread'
        threading.Thread.__init__(self,name=name)

    def get_UDP_port(self,cam_id):
        self.all_data_lock.acquire()
        try:
            i = self.cam_ids.index( cam_id )
            UDP_port = self.UDP_ports[i]
        finally:
            self.all_data_lock.release()

        return UDP_port

    def get_3d_data(self,typ):
        if typ == 'fast':
            d = self.save_3d_data_fastest
        elif typ == 'best':
            d = self.save_3d_data_best

        self.all_data_lock.acquire()
        try:
            if d is not None:
                result = d.copy()
            else:
                result = None
        finally:
            self.all_data_lock.release()

        return result

    def get_general_cam_info(self):
        self.all_data_lock.acquire()
        try:
            result = self.general_save_info.copy()
        finally:
            self.all_data_lock.release()
        return result

    def is_saving_2d_data(self):
        self.all_data_lock.acquire()
        try:
            result = self.save_2d_data_fd is not None
        finally:
            self.all_data_lock.release()
        return result
     
    def set_saving_2d_data(self,value):
        self.all_data_lock.acquire()
        try:
            if value:
                self.save_2d_data_fd = open('raw_data.dat','wb')
            else:
                self.save_2d_data_fd = None
        finally:
            self.all_data_lock.release()
    
    def clear_saved_2d_data(self):
        self.all_data_lock.acquire()
        try:
            if self.save_2d_data_fd is not None:
                self.save_2d_data_fd.seek(0)
                self.save_2d_data_fd.truncate()
        finally:
            self.all_data_lock.release()
    
    def set_reconstructor(self,r):
        self.all_data_lock.acquire()
        self.reconstructor = r    
        self.all_data_lock.release()

    def is_collecting_3d_data(self):
        self.all_data_lock.acquire()
        try:
            result = self.save_3d_data_fastest is not None
        finally:
            self.all_data_lock.release()
        return result
    
    def set_collect_3d_data(self,value):
        self.all_data_lock.acquire()
        try:
            if value:
                if self.save_3d_data_fastest is None:
                    # do not clear data if already present
                    self.save_3d_data_fastest = {}
                if self.save_3d_data_best is None:
                    self.save_3d_data_best = {}
            else:
                self.save_3d_data_fastest = None
                self.save_3d_data_best = None
        finally:
            self.all_data_lock.release()

    def clear_3d_data(self):
        self.all_data_lock.acquire()
        try:
            if self.save_3d_data_fastest is not None:
                self.save_3d_data_fastest = {}
            if self.save_3d_data_best is not None:
                self.save_3d_data_best = {}
        finally:
            self.all_data_lock.release()
            
    def connect(self,cam_id):
        global hostname
        
        self.all_data_lock.acquire()
        self.cam_ids.append(cam_id)
        
        # find UDP_port
        if len(self.UDP_ports)>0:
            UDP_port = max(self.UDP_ports)+1
        else:
            UDP_port = 34813
        self.UDP_ports.append( UDP_port )
        
        # find absolute_cam_no
        self.max_absolute_cam_nos += 1        
        absolute_cam_no = self.max_absolute_cam_nos
        self.absolute_cam_nos.append( absolute_cam_no )

        # create and bind socket to listen to
        sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockobj.bind((hostname, UDP_port))
        sockobj.setblocking(0)
        self.listen_sockets.append( sockobj )
        self.last_timestamps.append(-10.0) # arbitrary impossible number
        self.framenumber_offsets.append(0)
        self.general_save_info[cam_id] = {'cam_no':absolute_cam_no,
                                          'frame0':self.last_timestamps[-1]}
        self.all_data_lock.release()

        return UDP_port

    def disconnect(self,cam_id):
        i = self.cam_ids.index( cam_id )

        self.all_data_lock.acquire()
        
        del self.cam_ids[i]
        del self.UDP_ports[i]
        del self.absolute_cam_nos[i]
        self.listen_sockets[i].close()
        del self.listen_sockets[i]
        del self.last_timestamps[i]
        del self.framenumber_offsets[i]
        #del self.general_save_info[cam_id]
        self.all_data_lock.release()
    
    def quit(self):
        self.quit_event.set()
        
    def run(self):
        global downstream_hostnames, fastest_realtime_data, best_realtime_data
        global outgoing_UDP_socket, calib_data_lock, calib_IdMat, calib_points
        global calib_data_lock
        
        header_fmt = '<dli'
        header_size = struct.calcsize(header_fmt)
        pt_fmt = '<fffffffff'
        pt_size = struct.calcsize(pt_fmt)
        save_2d_fmt = 'Bidfffffffff'
        timeout = 0.1
        
        realtime_coord_dict = {}        

        while not self.quit_event.isSet():
            try:
                in_ready, out_ready, exc_ready = select.select( self.listen_sockets,
                                                                [], [], timeout )
            except Exception:
                raise
            except:
                # sometimes this code raises an old string error (not derived from Exception)
                print 'WARNING: CoordReceiver received an exception not derived from Exception'
                continue
            if not len(in_ready):
                continue
            
            self.all_data_lock.acquire()
            deferred_2d_data = []
            new_data_framenumbers = sets.Set()
            for sockobj in in_ready:
                try:
                    cam_idx = self.listen_sockets.index(sockobj)
                except ValueError:
                    # camera was dropped?
                    continue
                
                cam_id = self.cam_ids[cam_idx]
                absolute_cam_no = self.absolute_cam_nos[cam_idx]
                
                data, addr = sockobj.recvfrom(1024)

                header = data[:header_size]
                timestamp, framenumber, n_pts = struct.unpack(header_fmt,header)
                start=header_size
                points = []
                for i in range(n_pts):
                    end=start+pt_size
                    x,y,area,slope,eccentricity,p1,p2,p3,p4 = struct.unpack(pt_fmt,data[start:end])
                    points.append( (x,y,area,slope,eccentricity, p1,p2,p3,p4) )
                    start=end
                # XXX hack? make data available via cam_dict
                cam_dict = self.main_brain.remote_api.cam_info[cam_id]
                cam_dict['lock'].acquire()
                cam_dict['points']=points
                cam_dict['lock'].release()
                    
                if timestamp-self.last_timestamps[cam_idx] > self.RESET_FRAMENUMBER_DURATION:
                    self.framenumber_offsets[cam_idx] = framenumber
                    if self.last_timestamps[cam_idx] != -10.0:
                        print cam_id,'synchronized'
                        self.general_save_info[cam_id]['frame0']=timestamp
                        # XXX Could I make new absolute_cam_no to prevent data loss?
##                    else:
##                        print cam_id,'first 2D coordinates received'
                        
                self.last_timestamps[cam_idx]=timestamp
                corrected_framenumber = framenumber-self.framenumber_offsets[cam_idx]

                if self.save_2d_data_fd is not None:
                    args = ( absolute_cam_no,
                             corrected_framenumber,
                             timestamp,
                             points[0][0], # x
                             points[0][1], # y
                             points[0][2], # area
                             points[0][3], # slope
                             points[0][4], # eccentricity
                             points[0][5], # p1
                             points[0][6], # p2
                             points[0][7], # p3
                             points[0][8], # p4
                             )
                    deferred_2d_data.append(args) # defer saving to later
                    
                # save new frame data
                cur_framenumber_dict=realtime_coord_dict.setdefault(corrected_framenumber,{})
                cur_framenumber_dict[cam_id]=points[0] # XXX for now, only attempt 3D reconstruction of 1st point

                new_data_framenumbers.add( corrected_framenumber ) # insert into set

            # Now we've grabbed all data waiting on network. Now it's
            # time to calculate 3D info.
            
            # XXX could go for latest data first to minimize latency
            # on that data.

            for corrected_framenumber in new_data_framenumbers:
                if self.reconstructor is None:
                    # can't do any 3D math without calibration information
                    break
                data_dict = realtime_coord_dict[corrected_framenumber]
                num_cams_arrived = len(data_dict)
                d2 = {} # old "good" points will go in here
                for cam_id, PT in data_dict.iteritems():
                    if PT[0] + 1 > 1e-6: # only use found points
                        d2[cam_id] = PT
                num_good_images = len(d2)
                if num_good_images==2 or num_cams_arrived==len(self.cam_ids):
                    # either first possible moment or all data present
                    X, line3d = self.reconstructor.find3d(d2.items())
                    if line3d is None:
                        line3d = nan, nan, nan, nan, nan, nan
                    find3d_time = time.time()

                    x,y,z=X
                    outgoing_data = [x,y,z]
                    outgoing_data.extend( line3d ) # 6 component vector
                    outgoing_data.extend( (find3d_time,num_good_images) )

                    if len(downstream_hostnames):
                        data_packet = struct.pack('ifffffffffdi',corrected_framenumber,*outgoing_data)
                    if num_good_images==2:
                        # fastest 3d data
                        fastest_realtime_data = X, line3d
                        try:
                            for downstream_hostname in downstream_hostnames:
                                outgoing_UDP_socket.sendto(data_packet,(downstream_hostname,FASTEST_DATA_PORT))
                        except:
                            print 'WARNING: could not send 3d point data to projector:'
                            print
                        if self.save_3d_data_fastest is not None:
                            self.save_3d_data_fastest[corrected_framenumber]=outgoing_data
                    if num_cams_arrived==len(self.cam_ids):
                        # realtime 3d data
                        best_realtime_data = X, line3d
                        try:
                            for downstream_hostname in downstream_hostnames:
                                outgoing_UDP_socket.sendto(data_packet,(downstream_hostname,BEST_DATA_PORT))
                        except:
                            print 'WARNING: could not send 3d point data to projector:'
                            print
                        if self.save_3d_data_best is not None:
                            self.save_3d_data_best[corrected_framenumber]=outgoing_data

            # save calibration data -=-=-=-=-=-=-=-=
            if self.main_brain.currently_calibrating.isSet():
                for corrected_framenumber in new_data_framenumbers:
                    data_dict = realtime_coord_dict[corrected_framenumber]
                    if len(data_dict) == len(self.cam_ids):
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
                    
            # clean up old frame records to save RAM
            if len(realtime_coord_dict)>100:
                k=realtime_coord_dict.keys()
                k.sort()
                for ki in k[:-50]:
                    del realtime_coord_dict[ki]  

            # save data deferred from earlier...
            for args in deferred_2d_data:
                buf = struct.pack(save_2d_fmt,*args)
                self.save_2d_data_fd.write( buf )
            self.all_data_lock.release()
            
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
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands'].setdefault('set',{})[property_name]=value
                    old_value = cam['scalar_control_info'][property_name]
                    if type(old_value) == tuple and type(value) == int:
                        # brightness, gain, shutter
                        cam['scalar_control_info'][property_name] = (value, old_value[1], old_value[2])
                    else:
                        cam['scalar_control_info'][property_name] = value
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_request_image_async(self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['get_im']=None
                finally:
                    cam_lock.release()
            finally:
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
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['stop_recording']=None
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_quit( self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['quit']=True
                finally:
                    cam_lock.release()
            finally:
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

        def external_set_pmat( self, cam_id, value):
            self.cam_info_lock.acquire()            
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cam['commands']['pmat']=value
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
            
            UDP_port = self.main_brain.coord_receiver.connect(cam_id)
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
                                     'UDP_port':UDP_port,
                                     }
            self.cam_info_lock.release()
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
            UDP_port = self.main_brain.coord_receiver.get_UDP_port(cam_id)
            return UDP_port

        def close(self,cam_id):
            """gracefully say goodbye (caller: remote camera)"""
            self.cam_info_lock.acquire()
            self.main_brain.coord_receiver.disconnect(cam_id)
            #self.cam_info[cam_id]['coord_receiver'].quit()
            #del self.cam_info[cam_id]['coord_receiver']
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

        self.coord_receiver = CoordReceiver(self)
        self.coord_receiver.start()

    def IncreaseCamCounter(self,*args):
        self.num_cams += 1

    def DecreaseCamCounter(self,*args):
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
            if typ == 'fast': fname = fast_filename
            elif typ == 'best': fname = best_filename
            dikt = self.coord_receiver.get_3d_data(typ)
            
            fullpath = os.path.abspath(fname)
            print 'saving %s 3d data to "%s"...'%(typ,fullpath)
            fd=open(fname,'wb')

            keys=dikt.keys()
            keys.sort()
            for k in keys:
                fd.write('%d %s\n'%(k,' '.join(map( repr, dikt[k]))))
            fd.close()
            print '  done'

    def SaveGlobals(self,filename='camera_data.dat'):
        fullpath = os.path.abspath(filename)
        print 'saving globals to',fullpath
        fd=open(filename,'wb')
        dd=self.coord_receiver.get_general_cam_info()
        cam_ids=dd.keys()
        cam_ids.sort()
        for cam_id in cam_ids:
            fd.write('%s %d %s\n'%(cam_id,dd[cam_id]['cam_no'],
                                   repr(dd[cam_id]['frame0'])))
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
        print 'saving %d points to %s'%(points.shape[1],self.calib_dir)
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

        self.coord_receiver.quit()

    def load_calibration(self,dirname):
        cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = Reconstructor(calibration_dir=dirname)

        self.coord_receiver.set_reconstructor(self.reconstructor)
        
        for cam_id in cam_ids:
            self.remote_api.external_set_pmat( cam_id, self.reconstructor.get_pmat(cam_id))
    
    def __del__(self):
        self.quit()

    def set_all_cameras_debug_mode( self, value ):
        cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_set_debug( cam_id, value)

    def get_save_2d_data(self):
        return self.coord_receiver.is_saving_2d_data()
    def set_save_2d_data(self,value):
        self.coord_receiver.set_saving_2d_data(value)
    save_2d_data = property( get_save_2d_data, set_save_2d_data )
    
    def clear_2d_data(self):
        self.coord_receiver.clear_saved_2d_data()

    def is_collecting_3d_data(self):
        return self.coord_receiver.is_collecting_3d_data()
    def set_collect_3d_data(self,value):
        self.coord_receiver.set_collect_3d_data(value)
    collect_3d_data = property( is_collecting_3d_data, set_collect_3d_data )
    
    def clear_3d_data(self):
        self.coord_receiver.clear_3d_data()
