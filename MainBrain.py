import threading
import time
import socket
import Pyro.core
import os
import copy
import socket
import struct
from flydra.reconstruct import Reconstructor
reconstructor=Reconstructor()

Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

UDP_ports=[]
recent_data={}
recent_data_lock=threading.Lock()
zz={}
zz_lock=threading.Lock()
data_ready_event=threading.Event()
RESET_FRAMENUMBER_DURATION=1.0 # seconds

try:
    hostname = socket.gethostbyname('mainbrain')
except:
    hostname = socket.gethostbyname(socket.gethostname())

class CoordReceiver(threading.Thread):
    def __init__(self,new_data_event,cam_id,n_spaces):
        self.new_data_event = new_data_event
        self.cam_id = cam_id
        self.n_spaces=n_spaces # XXX tmp
        self.last_timestamp=0.0

        # set up threading stuff
        self.quit_event = threading.Event()
        threading.Thread.__init__(self)

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

    def quit(self):
        self.quit_event.set()
        
        # send packet to wake listener and allow thread to quit
        tmp_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tmp_socket.sendto(struct.pack('<dlii',0.0,-1,-1,-1),(hostname,self.port))
    
    def run(self):
        global recent_data, recent_data_lock
        
        while not self.quit_event.isSet():
            data, addr = self.recSocket.recvfrom(1024)
            incoming_data = struct.unpack('<dlii',data)

            timestamp, framenumber, index_x, index_y = incoming_data
            if framenumber==-1:
                continue # leftover in socket buffer from last run??
            
            if timestamp-self.last_timestamp > RESET_FRAMENUMBER_DURATION:
                self.framenumber_offset = framenumber
            self.last_timestamp=timestamp
            corrected_framenumber = framenumber-self.framenumber_offset

            recent_data_lock.acquire()
            recent_data[self.cam_id]= timestamp, corrected_framenumber, index_x, index_y
            recent_data_lock.release()

            zz_lock.acquire()
            # clean up old frame records to save RAM
            k=zz.keys()
            if len(k)>5:
                k.sort()
                for ki in k[:-3]:
                    del zz[ki]
                    
            # save new frame record
            cur_framenumber_dict=zz.setdefault(corrected_framenumber,{})
            cur_framenumber_dict[self.cam_id]=index_x,index_y

            # print results if 3D reconstruction possible
            if len(cur_framenumber_dict)>=2:
                data_dict = cur_framenumber_dict.copy()
            else:
                data_dict = None
            zz_lock.release()

            if data_dict is not None:
                t1 = time.time()
                print 'time.time() % 15d'%t1
                print ' framenumber %d:'%corrected_framenumber,cur_framenumber_dict
                cam_idx = []
                points2d = []
                k = data_dict.keys()
                k.sort()
                for cam_id in k:
                    cam_idx.append( int( cam_id[3] )-1 )
                    points2d.append( data_dict[cam_id] )
                X = reconstructor.find3d(cam_idx,points2d)
                t2 = time.time()
                latency = (t2-t1)*1000.0
                print ' 3d point:', X, '(3d calc duration % 4.1f msec)'%latency

            self.new_data_event.set()

##            timestamp, index_x, index_y = incoming_data
##            print "%s % 15f % 15f"%(self.cam_id, time.time(), timestamp)
            latency = (time.time()-timestamp)*1000.0
            print "% 15f %s% 6.1f: % 5d (% 3d % 3d)"%(time.time(),
                                               ' '*self.n_spaces,latency,corrected_framenumber,
                                                      index_x,index_y)
            #print 'RECV on port %d (from %s): %s'%(self.port,addr,repr(data))
        UDP_ports.remove( self.port )

class MainBrain:
    """Handle all camera network stuff and interact with application"""

    class RemoteAPI(Pyro.core.ObjBase):

        # ----------------------------------------------------------------
        #
        # Methods called locally
        #
        # ----------------------------------------------------------------

        def post_init(self):
            """call after __init__"""
            # let Pyro handle __init__
            self.cam_info = {}
            #self.cam_info_lock = threading.Lock() # XXX probably not needed: listen thread is only writer
            self.changed_cam_lock = threading.Lock()
            self.no_cams_connected = threading.Event()
            self.no_cams_connected.set()
            self.changed_cam_lock.acquire()
            self.new_cam_ids = []
            self.old_cam_ids = []
            self.changed_cam_lock.release()
            
            # threading control locks
            self.quit_now = threading.Event()
            self.thread_done = threading.Event()

        def listen(self,daemon):
            """thread mainloop"""
            quit_now_isSet = self.quit_now.isSet
            hr = daemon.handleRequests
            while not quit_now_isSet():
                hr(0.1) # block on select for n seconds
                cam_ids = self.cam_info.keys()
                for cam_id in cam_ids:
                    if not self.cam_info[cam_id]['caller'].connected:
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

        def register_new_camera(self,scalar_control_info):
            """register new camera, return cam_id (caller: remote camera)"""

            caller= self.daemon.getLocalStorage().caller # XXX Pyro hack??
            caller_addr= caller.addr
            caller_ip, caller_port = caller_addr
            fqdn = socket.getfqdn(caller_ip)
        
            cam_id = '%s:%d'%(fqdn,caller_port)

            cam_no = int(fqdn[3:])
            coord_receiver = CoordReceiver(data_ready_event,cam_id,(cam_no-1)*23)
            
            self.cam_info[cam_id] = {'commands':{}, # command queue for cam
                                     'lock':threading.Lock(), # prevent concurrent access
                                     'image':None,  # most recent image from cam
                                     'num_image_puts':0,
                                     'fps':None,    # most recept fps from cam
                                     'caller':caller,    # most recept fps from cam
                                     'scalar_control_info':scalar_control_info,
                                     'fqdn':fqdn,
                                     'coord_receiver':coord_receiver,
                                     }
            coord_receiver.start()
            
            self.no_cams_connected.clear()
            
            self.changed_cam_lock.acquire()
            self.new_cam_ids.append(cam_id)
            self.changed_cam_lock.release()
            
            return cam_id

        def set_image(self,cam_id,image):
            """set most recent image (caller: remote camera)"""
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            self.cam_info[cam_id]['image'] = image
            cam_lock.release()

        def set_fps(self,cam_id,fps):
            """set most recent fps (caller: remote camera)"""
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            self.cam_info[cam_id]['fps'] = fps
            cam_lock.release()

        def get_and_clear_commands(self,cam_id):
            cam = self.cam_info[cam_id]
            cam_lock = cam['lock']
            cam_lock.acquire()
            cmds = cam['commands']
            cam['commands'] = {}
            cam_lock.release()
            return cmds

        def get_coord_port(self,cam_id):
            """Send UDP port number to which camera should send realtime data"""
            return self.cam_info[cam_id]['coord_receiver'].get_port()

        def close(self,cam_id):
            """gracefully say goodbye (caller: remote camera)"""
            self.cam_info[cam_id]['coord_receiver'].quit()
            del self.cam_info[cam_id]['coord_receiver']
            del self.cam_info[cam_id]
            if not len(self.cam_info):
                self.no_cams_connected.set()
            
            self.changed_cam_lock.acquire()
            self.old_cam_ids.append(cam_id)
            self.changed_cam_lock.release()
            
    def __init__(self):
        Pyro.core.initServer(banner=0)

        port = 9833

        # start Pyro server
        daemon = Pyro.core.Daemon(host=hostname,port=port)
        remote_api = MainBrain.RemoteAPI(); remote_api.post_init()
        URI=daemon.connect(remote_api,'main_brain')

        # create (but don't start) listen thread
        self.listen_thread=threading.Thread(target=remote_api.listen,
                                            args=(daemon,))

        self.remote_api = remote_api

        self._new_camera_functions = []
        self._old_camera_functions = []

        self.camera_server = {} # dict of Pyro servers for each camera
        self.last_requested_image = {}
        self.pending_requests = {}
        self.last_set_param_time = {}
        self.set_new_camera_callback(self.AddCameraServer)
        self.set_old_camera_callback(self.RemoveCameraServer)
        
    def AddCameraServer(self, cam_id, scalar_control_info):
        time.sleep(0.1) # let camera server get started
        fqdn = self.remote_api.cam_info[cam_id]['fqdn'] # crosses thread boundary?
        port = 9834
        name = 'camera_server'
        
        camera_server_URI = "PYROLOC://%s:%d/%s" % (fqdn,port,name)
        print 'resolving',camera_server_URI,'at',time.time()
        camera_server = Pyro.core.getProxyForURI(camera_server_URI)
        print 'found'
        camera_server._setOneway(['send_most_recent_frame',
                                  'quit',
                                  'set_camera_property'])
        self.camera_server[cam_id] = camera_server
        self.camera_server[cam_id].prints('hello from main brain')
    
    def RemoveCameraServer(self, cam_id):
        del self.camera_server[cam_id]

    def start_listening(self):
        # start listen thread
        self.listen_thread.start()

    def set_new_camera_callback(self,handler):
        self._new_camera_functions.append(handler)

    def set_old_camera_callback(self,handler):
        self._old_camera_functions.append(handler)

    def service_pending(self):
        self.remote_api.changed_cam_lock.acquire()
        # release lock as quickly as possible
        new_cam_ids = self.remote_api.new_cam_ids
        self.remote_api.new_cam_ids = []
        old_cam_ids = self.remote_api.old_cam_ids
        self.remote_api.old_cam_ids = []
        self.remote_api.changed_cam_lock.release()

        for cam_id in new_cam_ids:
            if cam_id in old_cam_ids:
                continue # inserted and removed
            for new_cam_func in self._new_camera_functions:
                # get scalar_control_info
                cam = self.remote_api.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                scalar_control_info = copy.deepcopy(cam['scalar_control_info'])
                cam_lock.release()
                new_cam_func(cam_id,scalar_control_info)

        for cam_id in old_cam_ids:
            for old_cam_func in self._old_camera_functions:
                old_cam_func(cam_id)

    def get_last_image_fps(self, cam_id):
        cam = self.remote_api.cam_info[cam_id]
        cam_lock = cam['lock']
        cam_lock.acquire()
        image = cam['image']
        cam['image'] = None
        fps = cam['fps']
        cam['fps'] = None
        cam_lock.release()
        return image, fps

    def close_camera(self,cam_id):
        self.camera_server[cam_id].quit()

    def collect_background(self,cam_id):
        self.camera_server[cam_id].collect_background()

    def send_set_camera_property(self, cam_id, property_name, value):
        cam = self.remote_api.cam_info[cam_id]
        cam_lock = cam['lock']
        cam_lock.acquire()
        cam['commands'].setdefault('set',{})[property_name]=value
        cam_lock.release()

    def request_image_async(self, cam_id):
        cam = self.remote_api.cam_info[cam_id]
        cam_lock = cam['lock']
        cam_lock.acquire()
        cam['commands']['get_im']=None
        cam_lock.release()

    def get_image_sync(self, cam_id):
        return self.camera_server[cam_id].get_most_recent_frame()

    def start_recording(self, cam_id,filename):
        return self.camera_server[cam_id].start_recording(filename)

    def stop_recording(self, cam_id):
        return self.camera_server[cam_id].stop_recording()

    def quit(self):
        # this may be called twice: once explicitly and once by __del__
        print 'sending quit signal to cameras'
        cam_ids = self.remote_api.cam_info.keys()
        for cam_id in cam_ids:
            try:
                self.close_camera(cam_id)
            except Pyro.errors.ProtocolError:
                # disconnection results in error
                print 'ignoring exception on',cam_id
                pass
        print 'waiting for cameras to quit'
        self.remote_api.no_cams_connected.wait(2.0)
        print 'sending quit signal to listen_thread...'
        self.remote_api.quit_now.set() # tell thread to finish
        print 'waiting for listen_thread to quit...'
        self.remote_api.thread_done.wait(0.5) # wait for thread to finish
        if not self.remote_api.no_cams_connected.isSet():
            cam_ids = self.remote_api.cam_info.keys()
            print 'cameras failed to quit cleanly: %s'%str(cam_ids)
            #raise RuntimeError('cameras failed to quit cleanly: %s'%str(cam_ids))
    
    def __del__(self):
        self.quit()
