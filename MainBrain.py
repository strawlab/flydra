# $Id$
import threading, time, socket, select, sys, os, copy, struct, math
import sets, traceback
import Pyro.core
import flydra.reconstruct
import reconstruct_utils as ru
import numarray as nx
import numarray.records
from numarray.ieeespecial import nan, inf, getnan
import Queue
import tables as PT

THRESHOLD_MEAN_DIST = 5.0 # cutoff acceptable pixel distance to stop searching camera combinations
REALTIME_UDP = False

Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here...

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

IMPOSSIBLE_TIMESTAMP = -10.0

calib_data_lock = threading.Lock()
calib_IdMat = []
calib_points = []

XXX_framenumber = 0

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

# 2D data format for PyTables:
class Info2D(PT.IsDescription):
    camn         = PT.Int32Col(pos=0)
    frame        = PT.Int32Col(pos=1)
    timestamp    = PT.FloatCol(pos=2)
    x            = PT.Float32Col(pos=3)
    y            = PT.Float32Col(pos=4)
    area         = PT.Float32Col(pos=5)
    slope        = PT.Float32Col(pos=6)
    eccentricity = PT.Float32Col(pos=7)
    p1           = PT.Float32Col(pos=8)
    p2           = PT.Float32Col(pos=9)
    p3           = PT.Float32Col(pos=10)
    p4           = PT.Float32Col(pos=11)

class CamSyncInfo(PT.IsDescription):
    cam_id = PT.StringCol(16,pos=0)
    camn   = PT.Int32Col(pos=1)
    frame0 = PT.FloatCol(pos=2)

class MovieInfo(PT.IsDescription):
    cam_id             = PT.StringCol(16,pos=0)
    filename           = PT.StringCol(255,pos=1)
    approx_start_frame = PT.Int32Col(pos=2)
    approx_stop_frame  = PT.Int32Col(pos=3)

class AdditionalInfo(PT.IsDescription):
    cal_source_type      = PT.StringCol(20)
    cal_source           = PT.StringCol(80)
    minimum_eccentricity = PT.Float32Col() # record what parameter was used during reconstruction

class Info3D(PT.IsDescription):
    frame      = PT.Int32Col(pos=0)
    
    x          = PT.Float32Col(pos=1)
    y          = PT.Float32Col(pos=2)
    z          = PT.Float32Col(pos=3)
    
    p0         = PT.Float32Col(pos=4)
    p1         = PT.Float32Col(pos=5)
    p2         = PT.Float32Col(pos=6)
    p3         = PT.Float32Col(pos=7)
    p4         = PT.Float32Col(pos=8)
    p5         = PT.Float32Col(pos=9)
    
    timestamp  = PT.FloatCol(pos=10)
    
    camns_used = PT.StringCol(32,pos=11)
    mean_dist  = PT.Float32Col(pos=12) # mean 2D reconstruction error
    
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

class DebugLock:
    def __init__(self,name,verbose=False):
        self.name = name
        self._lock = threading.Lock()
        self.verbose = verbose

    def acquire(self):
        if self.verbose:
            print '-='*20
        print '*****',self.name,'request acquire by',threading.currentThread()
        if self.verbose:
            frame = sys._getframe()
            traceback.print_stack(frame)
            print '-='*20
        self._lock.acquire()
        print '*****',self.name,'acquired by',threading.currentThread()
        if self.verbose:
            traceback.print_stack(frame)
            print '-='*20

    def release(self):
        print '*****',self.name,'released by',threading.currentThread()
        self._lock.release()

class CoordReceiver(threading.Thread):
    def __init__(self,main_brain):
        
        self.main_brain = main_brain
        
        self.cam_ids = []
        self.cam2mainbrain_data_ports = []
        self.absolute_cam_nos = []
        self.last_timestamps = []
        self.last_framenumbers_delay = []
        self.last_framenumbers_skip = []
        self.listen_sockets = {}
        self.server_sockets = {}
        self.framenumber_offsets = []
        self.cam_id2cam_no = {}
        self.reconstructor = None

        self.all_data_lock = threading.Lock()
        self.quit_event = threading.Event()
        
        self.max_absolute_cam_nos = -1
        self.RESET_FRAMENUMBER_DURATION=1.0 # seconds
        
        self.general_save_info = {}

        self._fake_sync_event = threading.Event()
        
        name = 'CoordReceiver thread'
        threading.Thread.__init__(self,name=name)

    def get_cam2mainbrain_data_port(self,cam_id):
        self.all_data_lock.acquire()
        try:
            i = self.cam_ids.index( cam_id )
            cam2mainbrain_data_port = self.cam2mainbrain_data_ports[i]
        finally:
            self.all_data_lock.release()

        return cam2mainbrain_data_port

    def get_general_cam_info(self):
        self.all_data_lock.acquire()
        try:
            result = self.general_save_info.copy()
        finally:
            self.all_data_lock.release()
        return result

    def set_reconstructor(self,r):
        self.all_data_lock.acquire()
        try:
            self.reconstructor = r
        finally:
            self.all_data_lock.release()

    def connect(self,cam_id):
        global hostname

        assert not self.main_brain.is_saving_data()
        
        self.all_data_lock.acquire()
        try:
            self.cam_ids.append(cam_id)
        
            # find cam2mainbrain_data_port
            if len(self.cam2mainbrain_data_ports)>0:
                cam2mainbrain_data_port = max(self.cam2mainbrain_data_ports)+1
            else:
                cam2mainbrain_data_port = 34813
            self.cam2mainbrain_data_ports.append( cam2mainbrain_data_port )

            # find absolute_cam_no
            self.max_absolute_cam_nos += 1        
            absolute_cam_no = self.max_absolute_cam_nos
            self.absolute_cam_nos.append( absolute_cam_no )

            self.cam_id2cam_no[cam_id] = absolute_cam_no

            # create and bind socket to listen to
            if REALTIME_UDP:
                sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sockobj.bind((hostname, cam2mainbrain_data_port))
                sockobj.setblocking(0)
                self.listen_sockets[ sockobj ] = cam_id
            else:
                sockobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockobj.bind((hostname, cam2mainbrain_data_port))
                sockobj.listen(1)
                sockobj.setblocking(0)
                self.server_sockets[ sockobj ] = cam_id
            self.last_timestamps.append(IMPOSSIBLE_TIMESTAMP) # arbitrary impossible number
            self.last_framenumbers_delay.append(-1) # arbitrary impossible number
            self.last_framenumbers_skip.append(-1) # arbitrary impossible number
            self.framenumber_offsets.append(0)
            self.general_save_info[cam_id] = {'absolute_cam_no':absolute_cam_no,
                                              'frame0':IMPOSSIBLE_TIMESTAMP}
            self.main_brain.queue_cam_info.put(  (cam_id, absolute_cam_no, IMPOSSIBLE_TIMESTAMP) )
        finally:
            self.all_data_lock.release()

        return cam2mainbrain_data_port

    def disconnect(self,cam_id):
        cam_idx = self.cam_ids.index( cam_id )
        self.all_data_lock.acquire()
        try:
            del self.cam_ids[cam_idx]
            del self.cam2mainbrain_data_ports[cam_idx]
            del self.absolute_cam_nos[cam_idx]
            for sockobj, test_cam_id in self.listen_sockets.iteritems():
                if cam_id == test_cam_id:
                    sockobj.close()
                    del self.listen_sockets[sockobj]
                    break # XXX naughty to delete item inside iteration
            for sockobj, test_cam_id in self.server_sockets.iteritems():
                if cam_id == test_cam_id:
                    sockobj.close()
                    del self.server_sockets[sockobj]
                    break # XXX naughty to delete item inside iteration
            del self.last_timestamps[cam_idx]
            del self.last_framenumbers_delay[cam_idx]
            del self.last_framenumbers_skip[cam_idx]
            del self.framenumber_offsets[cam_idx]
            del self.general_save_info[cam_id]
        finally:
            self.all_data_lock.release()
    
    def quit(self):
        # called from outside of thread to quit the thread
        self.quit_event.set()
        self.join() # wait until CoordReveiver thread quits

    def fake_synchronize(self):
        self._fake_sync_event.set()

    def OnSynchronize(self, cam_idx, cam_id, framenumber, timestamp,
                      realtime_coord_dict, new_data_framenumbers):
        self.framenumber_offsets[cam_idx] = framenumber
        if self.last_timestamps[cam_idx] != IMPOSSIBLE_TIMESTAMP:
            print cam_id,'(re)synchronized'
            # discard all previous data
            for k in realtime_coord_dict.keys():
                del realtime_coord_dict[k]
            new_data_framenumbers.clear()

        #else:
        #    print cam_id,'first 2D coordinates received'

        # make new absolute_cam_no to indicate new synchronization state
        self.max_absolute_cam_nos += 1        
        absolute_cam_no = self.max_absolute_cam_nos
        self.absolute_cam_nos[cam_idx] = absolute_cam_no

        self.cam_id2cam_no[cam_id] = absolute_cam_no

        self.general_save_info[cam_id]['absolute_cam_no']=absolute_cam_no
        self.general_save_info[cam_id]['frame0']=timestamp

        self.main_brain.queue_cam_info.put(  (cam_id, absolute_cam_no, timestamp) )

        
    def run(self):
        global downstream_hostnames, fastest_realtime_data, best_realtime_data
        global outgoing_UDP_socket, calib_data_lock, calib_IdMat, calib_points
        global calib_data_lock, XXX_framenumber
        
        header_fmt = '<dli'
        header_size = struct.calcsize(header_fmt)
        #pt_fmt = '<fffffffff'
        pt_fmt = 'ddddddddd'
        pt_size = struct.calcsize(pt_fmt)
        timeout = 0.1
        
        realtime_coord_dict = {}        
        new_data_framenumbers = sets.Set()

##        tstamp_fd = open('tstamps.bin',mode='wb')

        select_select = select.select
        time_time = time.time
        empty_list = []
        old_data = {}
        while not self.quit_event.isSet():
            if not REALTIME_UDP:
                try:
                    in_ready, out_ready, exc_ready = select_select( self.server_sockets.keys(),
                                                                    empty_list, empty_list, 0.0 )
                except select.error, exc:
                    print 'select.error on server socket, ignoring...'
                    continue
                except socket.error, exc:
                    print 'socket.error on server socket, ignoring...'
                    continue
                for sockobj in in_ready:
                    cam_id = self.server_sockets[sockobj]
                    client_sockobj, addr = sockobj.accept()
                    client_sockobj.setblocking(0)
                    print cam_id, 'connected from',addr
                    self.listen_sockets[client_sockobj]=cam_id
            try:
                in_ready, out_ready, exc_ready = select_select( self.listen_sockets.keys(),
                                                                empty_list, empty_list, timeout )
            except select.error, exc:
                print 'select.error on listen socket, ignoring...'
                continue
            except socket.error, exc:
                print 'socket.error on listen socket, ignoring...'
                continue
            except Exception, exc:
                raise
            except:
                print 'ERROR: CoordReceiver received an exception not derived from Exception'
                print '-='*10,'I should really quit now!','-='*10
                continue
##            now = time_time()
##            tstamp_fd.write( struct.pack('d',now ) )
            new_data_framenumbers.clear()
            if self._fake_sync_event.isSet():
                for cam_idx, cam_id in enumerate(self.cam_ids):
                    timestamp = self.last_timestamps[cam_idx]
                    framenumber = self.last_framenumbers_delay[cam_idx]
                    self.OnSynchronize( cam_idx, cam_id, framenumber, timestamp,
                                        realtime_coord_dict, new_data_framenumbers )
                self._fake_sync_event.clear()
            if not len(in_ready):
                continue
            
            self.all_data_lock.acquire()
            try:
                deferred_2d_data = []
                for sockobj in in_ready:
                    try:
                        cam_id = self.listen_sockets[sockobj]
                    except KeyError,ValueError:
                        # camera was dropped?
                        continue
                    cam_idx = self.cam_ids.index(cam_id)
                    #cam_id = self.cam_ids[cam_idx]
                    absolute_cam_no = self.absolute_cam_nos[cam_idx]

                    if REALTIME_UDP:
                        newdata, addr = sockobj.recvfrom(4096)
                    else:
                        newdata = sockobj.recv(4096)
                    data = old_data.get( sockobj, '')
                    data = data + newdata
                    while len(data):
                        header = data[:header_size]
                        if len(header) != header_size:
                            #print 'incomplete header buffer'
                            break
                        timestamp, framenumber, n_pts = struct.unpack(header_fmt,header)
                        points = []
                        if len(data) < header_size + n_pts*pt_size:
                            #print 'incomplete point info'
                            break
                        if framenumber-self.last_framenumbers_skip[cam_idx] > 1:
                            print '  WARNING: frame data loss %s'%(cam_id,) # (or UDP out-of-order)
                        self.last_framenumbers_skip[cam_idx]=framenumber
                        start=header_size
                        for i in range(n_pts):
                            end=start+pt_size
                            x,y,area,slope,eccentricity,p1,p2,p3,p4 = struct.unpack(pt_fmt,data[start:end])
                            points.append( (x,y,area,slope,eccentricity, p1,p2,p3,p4) )
                            start=end
                        data = data[end:]

                        # -----------------------------------------------

                        # XXX hack? make data available via cam_dict
                        cam_dict = self.main_brain.remote_api.cam_info[cam_id]
                        cam_dict['lock'].acquire()
                        cam_dict['points']=points
                        cam_dict['lock'].release()

                        #if timestamp-self.last_timestamps[cam_idx] > 0.02:
                        #    print 'WARNING frame skipped %s?'%cam_id

        ##                if framenumber-self.last_framenumbers_delay[cam_idx] > 1:
        ##                    print '  (frame delay/misorder %s)'%(cam_id,)

                        if timestamp-self.last_timestamps[cam_idx] > self.RESET_FRAMENUMBER_DURATION:
                            self.OnSynchronize( cam_idx, cam_id, framenumber, timestamp,
                                                realtime_coord_dict, new_data_framenumbers )

                        self.last_timestamps[cam_idx]=timestamp
                        self.last_framenumbers_delay[cam_idx]=framenumber
                        corrected_framenumber = framenumber-self.framenumber_offsets[cam_idx]
                        XXX_framenumber = corrected_framenumber

                        if self.main_brain.is_saving_data() or len(getnan(points[0][0])[0]):
                            # Save 2D data (even when no point found) to allow
                            # temporal correlation of movie frames to 2D data.
                            deferred_2d_data.append((absolute_cam_no, # defer saving to later
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
                                                     ))

                        # save new frame data
                        # XXX for now, only attempt 3D reconstruction of 1st point
                        realtime_coord_dict.setdefault(corrected_framenumber,{})[cam_id]=points[0]

                        new_data_framenumbers.add( corrected_framenumber ) # insert into set

                    # preserve unprocessed data
                    old_data[sockobj] = data

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
                        if PT[0] > -0.9: # only use found points
                            d2[cam_id] = PT
                    num_good_images = len(d2)
                    if num_good_images < 2:
                        # no point in dealing with this frame!
                        continue
                    #if num_good_images==2 or num_cams_arrived==len(self.cam_ids):
                    if num_cams_arrived==len(self.cam_ids):
                        #X, line3d = self.reconstructor.find3d(d2.items())
                        #cam_ids_used = d2.keys()
                        try:
                            X, line3d, cam_ids_used, min_mean_dist = ru.find_best_3d(self.reconstructor,d2)
                        except:
                            # this prevents us from bombing this thread...
                            print 'WARNING:'
                            traceback.print_last()
                            print 'SKIPPED 3d calculation for this frame.'
                            continue
                        cam_nos_used = [self.cam_id2cam_no[cam_id] for cam_id in cam_ids_used]

                        if line3d is None:
                            line3d = nan, nan, nan, nan, nan, nan
                        find3d_time = time.time()

                        x,y,z=X
                        outgoing_data = [x,y,z]
                        outgoing_data.extend( line3d ) # 6 component vector
                        outgoing_data.append( find3d_time )

                        if len(downstream_hostnames):
                            data_packet = struct.pack('ifffffffffd',corrected_framenumber,*outgoing_data)
                        if num_good_images==2:
                            # fastest 3d data
                            fastest_realtime_data = X, line3d, cam_ids_used, min_mean_dist
                            try:
                                for downstream_hostname in downstream_hostnames:
                                    outgoing_UDP_socket.sendto(data_packet,(downstream_hostname,FASTEST_DATA_PORT))
                            except:
                                print 'WARNING: could not send 3d point data to projector:'
                                print
                            if self.main_brain.is_saving_data():
                                self.main_brain.queue_data3d_fastest.put( (corrected_framenumber,
                                                                           outgoing_data,
                                                                           cam_nos_used,
                                                                           min_mean_dist) )
                        if num_cams_arrived==len(self.cam_ids):
                            # realtime 3d data
                            best_realtime_data = X, line3d, cam_ids_used, min_mean_dist
                            try:
                                for downstream_hostname in downstream_hostnames:
                                    outgoing_UDP_socket.sendto(data_packet,(downstream_hostname,BEST_DATA_PORT))
                            except:
                                print 'WARNING: could not send 3d point data to projector:'
                                print
                            if self.main_brain.is_saving_data():
                                self.main_brain.queue_data3d_best.put( (corrected_framenumber,
                                                                        outgoing_data,
                                                                        cam_nos_used,
                                                                        min_mean_dist) )

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
                                if len(getnan(pt[:2])[0]):
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
                # XXX does this drop unintended frames on re-sync?
                if len(realtime_coord_dict)>100:
                    k=realtime_coord_dict.keys()
                    k.sort()
                    for ki in k[:-50]:
                        del realtime_coord_dict[ki]  

                if len(deferred_2d_data):
                    self.main_brain.queue_data2d.put( deferred_2d_data )

            finally:
                self.all_data_lock.release()
    ##        tstamp_fd.close()
    
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
            try:
                self.new_cam_ids = []
                self.old_cam_ids = []
            finally:
                self.changed_cam_lock.release()
            self.main_brain = main_brain
            
            # threading control locks
            self.quit_now = threading.Event()
            self.thread_done = threading.Event()

        def external_get_and_clear_pending_cams(self):
            self.changed_cam_lock.acquire()
            try:
                new_cam_ids = self.new_cam_ids
                self.new_cam_ids = []
                old_cam_ids = self.old_cam_ids
                self.old_cam_ids = []
            finally:
                self.changed_cam_lock.release()
            return new_cam_ids, old_cam_ids

        def external_get_cam_ids(self):
            self.cam_info_lock.acquire()
            try:
                cam_ids = self.cam_info.keys()
            finally:
                self.cam_info_lock.release()
            cam_ids.sort()
            return cam_ids

        def external_get_info(self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    scalar_control_info = copy.deepcopy(cam['scalar_control_info'])
                    fqdn = cam['fqdn']
                    port = cam['port']
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()
            return scalar_control_info, fqdn, port

        def external_get_image_fps_points(self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    image = cam['image']
                    cam['image'] = None
                    fps = cam['fps']
                    cam['fps'] = None
                    points = cam['points'][:]
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()
            # NB: points are undistorted (and therefore do not align
            # with distorted image)
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

        def external_start_recording( self, cam_id, raw_filename, bg_filename):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['start_recording']=raw_filename, bg_filename
                finally:
                    cam_lock.release()
            finally:
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
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['use_arena']=value
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_find_r_center( self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['find_r_center']=None
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_set_collecting_background( self, cam_id, value):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['collecting_bg']=value
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_take_background( self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['take_bg']=None
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()        

        def external_clear_background( self, cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['clear_bg']=None
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()        

        def external_set_debug( self, cam_id, value):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['debug']=value
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        def external_set_cal( self, cam_id, pmat, intlin, intnonlin):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cam['commands']['cal']= pmat, intlin, intnonlin
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()

        # --- thread boundary -----------------------------------------

        def listen(self,daemon):
            """thread mainloop"""
            quit_now_isSet = self.quit_now.isSet
            hr = daemon.handleRequests
            while not quit_now_isSet():
                hr(0.1) # block on select for n seconds
                self.cam_info_lock.acquire()
                try:
                    cam_ids = self.cam_info.keys()
                finally:
                    self.cam_info_lock.release()
                for cam_id in cam_ids:
                    self.cam_info_lock.acquire()
                    try:
                        connected = self.cam_info[cam_id]['caller'].connected
                    finally:
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
            
            cam2mainbrain_data_port = self.main_brain.coord_receiver.connect(cam_id)
            self.cam_info_lock.acquire()
            try:
                self.cam_info[cam_id] = {'commands':{}, # command queue for cam
                                         'lock':threading.Lock(), # prevent concurrent access
                                         'image':None,  # most recent image from cam
                                         'fps':None,    # most recept fps from cam
                                         'points':[], # 2D image points
                                         'caller':caller,    # most recept fps from cam
                                         'scalar_control_info':scalar_control_info,
                                         'fqdn':fqdn,
                                         'port':port,
                                         'cam2mainbrain_data_port':cam2mainbrain_data_port,
                                         }
            finally:
                self.cam_info_lock.release()
            self.no_cams_connected.clear()
            self.changed_cam_lock.acquire()
            try:
                self.new_cam_ids.append(cam_id)
            finally:
                self.changed_cam_lock.release()
            
            return cam_id

        def set_image(self,cam_id,image):
            """set most recent image (caller: remote camera)"""
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    self.cam_info[cam_id]['image'] = image
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()            

        def set_fps(self,cam_id,fps):
            """set most recent fps (caller: remote camera)"""
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    self.cam_info[cam_id]['fps'] = fps
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()            

        def get_and_clear_commands(self,cam_id):
            self.cam_info_lock.acquire()
            try:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                cam_lock.acquire()
                try:
                    cmds = cam['commands']
                    cam['commands'] = {}
                finally:
                    cam_lock.release()
            finally:
                self.cam_info_lock.release()
            return cmds

        def get_cam2mainbrain_port(self,cam_id):
            """Send port number to which camera should send realtime data"""
            cam2mainbrain_data_port = self.main_brain.coord_receiver.get_cam2mainbrain_data_port(cam_id)
            return cam2mainbrain_data_port

        def close(self,cam_id):
            """gracefully say goodbye (caller: remote camera)"""
            self.cam_info_lock.acquire()
            try:
                self.main_brain.coord_receiver.disconnect(cam_id)
                #self.cam_info[cam_id]['coord_receiver'].quit()
                #del self.cam_info[cam_id]['coord_receiver']
                del self.cam_info[cam_id]
                if not len(self.cam_info):
                    self.no_cams_connected.set()
                self.changed_cam_lock.acquire()
                try:
                    self.old_cam_ids.append(cam_id)
                finally:
                    self.changed_cam_lock.release()
            finally:
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
        self.listen_thread.setDaemon(True) # don't let this thread keep app alive
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

        self._currently_recording_movies = {}
        
        self.reconstructor = None

        # Attributes which come in use when saving data occurs
        self.h5file = None
        self.h5data2d = None
        self.h5cam_info = None
        self.h5movie_info = None
        self.h5data3d_fastest = None
        self.h5data3d_best = None
        self.h5additional_info = None

        # Queues of information to save
        self.queue_data2d         = Queue.Queue()
        self.queue_cam_info       = Queue.Queue()
        self.queue_data3d_fastest = Queue.Queue()
        self.queue_data3d_best    = Queue.Queue()

        self.coord_receiver = CoordReceiver(self)
        self.coord_receiver.setDaemon(True)
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

    def start_listening(self):
        # start listen thread
        #self.listen_thread.setDaemon(True)
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
        try:
            IdMat = calib_IdMat
            calib_IdMat = []
        
            points = calib_points
            calib_points = []
        finally:
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
            if self.is_saving_data():
                raise RuntimeError("Cannot add new camera while saving data")
            scalar_control_info, fqdn, port = self.remote_api.external_get_info(cam_id)
            for new_cam_func in self._new_camera_functions:
                new_cam_func(cam_id,scalar_control_info,(fqdn,port))

        for cam_id in old_cam_ids:
            for old_cam_func in self._old_camera_functions:
                old_cam_func(cam_id)
                
        self._service_save_data()

    def get_last_image_fps(self, cam_id):
        
        # Points are originally undistorted (and therefore do not
        # align with distorted image). We must distort them.
        
        image, fps, points = self.remote_api.external_get_image_fps_points(cam_id)
        if self.reconstructor is None:
            distorted_points = points
        else:
            distorted_points = []
            for pt in points:
                xd,yd = self.reconstructor.distort(cam_id,pt)
                dp = list(pt)
                dp[0] = xd
                dp[1] = yd
                distorted_points.append( dp )
        return image, fps, distorted_points

    def fake_synchronize(self):
        self.coord_receiver.fake_synchronize()

    def close_camera(self,cam_id):
        sys.stdout.flush()
        self.remote_api.external_quit( cam_id )
        sys.stdout.flush()

    def set_use_arena(self, cam_id, value):
        self.remote_api.external_set_use_arena( cam_id, value)

    def set_debug_mode(self, cam_id, value):
        self.remote_api.external_set_debug( cam_id, value)

    def set_collecting_background(self, cam_id, value):
        self.remote_api.external_set_collecting_background( cam_id, value)

    def take_background(self,cam_id):
        self.remote_api.external_take_background(cam_id)

    def clear_background(self,cam_id):
        self.remote_api.external_clear_background(cam_id)

    def find_r_center(self,cam_id):
        self.remote_api.external_find_r_center(cam_id)

    def send_set_camera_property(self, cam_id, property_name, value):
        self.remote_api.external_send_set_camera_property( cam_id, property_name, value)

    def request_image_async(self, cam_id):
        self.remote_api.external_request_image_async(cam_id)

    def start_recording(self, cam_id, raw_filename, bg_filename):
        global XXX_framenumber

        self.remote_api.external_start_recording( cam_id, raw_filename, bg_filename)
        approx_start_frame = XXX_framenumber
        self._currently_recording_movies[ cam_id ] = (raw_filename, approx_start_frame)
        if self.is_saving_data():
            self.h5movie_info.row['cam_id'] = cam_id
            self.h5movie_info.row['filename'] = raw_filename
            self.h5movie_info.row['approx_start_frame'] = approx_start_frame
            self.h5movie_info.row.append()
            self.h5movie_info.flush()
        
    def stop_recording(self, cam_id):
        global XXX_framenumber
        self.remote_api.external_stop_recording(cam_id)
        approx_stop_frame = XXX_framenumber
        raw_filename, approx_start_frame = self._currently_recording_movies[ cam_id ]
        del self._currently_recording_movies[ cam_id ]
        # modify save file to include approximate movie stop time
        if self.is_saving_data():
            nrow = None
            for r in self.h5movie_info:
                # get row in table
                if (r['cam_id'] == cam_id and r['filename'] == raw_filename and
                    r['approx_start_frame']==approx_start_frame):
                    nrow =r.nrow()
                    break
            if nrow is not None:
                nrowi = int(nrow) # pytables bug workaround...
                assert nrowi == nrow # pytables bug workaround...
                self.h5movie_info.cols.approx_stop_frame[nrowi] = approx_stop_frame
            else:
                raise RuntimeError("could not find row to save movie stop frame.")
                    
    def quit(self):
        # XXX ----- non-isolated calls to remote_api being done ----
        # this may be called twice: once explicitly and once by __del__
        self.remote_api.cam_info_lock.acquire()
        try:
            cam_ids = self.remote_api.cam_info.keys()
        finally:
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
        if self.is_saving_data():
            self._service_save_data()
            self.h5file.close()
            self.h5file = None

    def load_calibration(self,dirname):
        if self.is_saving_data():
            raise RuntimeError("Cannot (re)load calibration while saving data")
        cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = flydra.reconstruct.Reconstructor(dirname)

        self.coord_receiver.set_reconstructor(self.reconstructor)
        
        for cam_id in cam_ids:
            pmat = self.reconstructor.get_pmat(cam_id)
            intlin = self.reconstructor.get_intrinsic_linear(cam_id)
            intnonlin = self.reconstructor.get_intrinsic_nonlinear(cam_id)
            self.remote_api.external_set_cal( cam_id, pmat, intlin, intnonlin )
    
    def __del__(self):
        self.quit()

    def set_all_cameras_debug_mode( self, value ):
        cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_set_debug( cam_id, value)

    def is_saving_data(self):
        return self.h5file is not None

    def start_saving_data(self, filename):
        if os.path.exists(filename):
            raise RuntimeError("will not overwrite data file")

        self.h5file = PT.openFile(filename, mode="w", title="Flydra data file")
        expected_rows = int(1e6)
        ct = self.h5file.createTable # shorthand
        root = self.h5file.root # shorthand
        self.h5data2d   = ct(root,'data2d', Info2D, "2d data",
                             expectedrows=expected_rows*5)
        self.h5cam_info = ct(root,'cam_info', CamSyncInfo, "Cam Sync Info",
                             expectedrows=500)
        self.h5movie_info = ct(root,'movie_info', MovieInfo, "Movie Info",
                               expectedrows=500)
        if self.reconstructor is not None:
            cal_group = self.h5file.createGroup(root,'calibration')
            
            pmat_group = self.h5file.createGroup(cal_group,'pmat')
            for cam_id in self.remote_api.external_get_cam_ids():
                self.h5file.createArray(pmat_group, cam_id,
                                        self.reconstructor.get_pmat(cam_id))
            res_group = self.h5file.createGroup(cal_group,'resolution')
            for cam_id in self.remote_api.external_get_cam_ids():
                res = self.reconstructor.get_resolution(cam_id)
                self.h5file.createArray(res_group, cam_id, res)
                                        
            intlin_group = self.h5file.createGroup(cal_group,'intrinsic_linear')
            for cam_id in self.remote_api.external_get_cam_ids():
                intlin = self.reconstructor.get_intrinsic_linear(cam_id)
                self.h5file.createArray(intlin_group, cam_id, intlin)
                                        
            intnonlin_group = self.h5file.createGroup(cal_group,'intrinsic_nonlinear')
            for cam_id in self.remote_api.external_get_cam_ids():
                self.h5file.createArray(intnonlin_group, cam_id,
                                        self.reconstructor.get_intrinsic_nonlinear(cam_id))

            self.h5additional_info = ct(cal_group,'additional_info', AdditionalInfo,
                                        '')
            row = self.h5additional_info.row
            row['cal_source_type'] = self.reconstructor.cal_source_type
            row['cal_source'] = self.reconstructor.cal_source
            row['minimum_eccentricity'] = flydra.reconstruct.MINIMUM_ECCENTRICITY
            row.append()
            self.h5additional_info.flush()
                
            self.h5data3d_fastest = ct(root,'data3d_fastest', Info3D,
                                       "3d data (fastest)",
                                       expectedrows=expected_rows)
            self.h5data3d_best = ct(root,'data3d_best', Info3D,
                                    "3d data (best)",
                                    expectedrows=expected_rows)

        general_save_info=self.coord_receiver.get_general_cam_info()
        for cam_id,dd in general_save_info.iteritems():
            self.h5cam_info.row['cam_id'] = cam_id
            self.h5cam_info.row['camn']   = dd['absolute_cam_no']
            self.h5cam_info.row['frame0'] = dd['frame0']
            self.h5cam_info.row.append()
        self.h5cam_info.flush()

    def stop_saving_data(self):
        self._service_save_data()
        self.h5file.close()
        self.h5file = None
        self.h5data2d = None
        self.h5cam_info = None
        self.h5movie_info = None
        self.h5data3d_fastest = None
        self.h5data3d_best = None
        self.h5additional_info = None

    def _service_save_data(self):
        # ** 2d data **
        #   clear queue
        list_of_rows_of_data2d = []
        try:
            while True:
                tmp = self.queue_data2d.get(0)
                list_of_rows_of_data2d.append( tmp )
        except Queue.Empty:
            pass
        #   save
        if self.h5data2d is not None:
            #nrows = 0
            for rows_of_data2d in list_of_rows_of_data2d:
                #nrows += len(rows_of_data2d)
                try:
                    self.h5data2d.append( rows_of_data2d )
                except ValueError:
                    print 'ERROR following'
                    print
                    print 'rows_of_data2d='
                    print rows_of_data2d
                    print
                    raise
            self.h5data2d.flush()
            #print 'saved %d rows of 2D data'%nrows

        # ** camera info **
        #   clear queue
        list_of_cam_info = []
        try:
            while True:
                list_of_cam_info.append( self.queue_cam_info.get(0) )
        except Queue.Empty:
            pass
        #   save
        if self.h5cam_info is not None:
            cam_info_row = self.h5cam_info.row
            for cam_info in list_of_cam_info:
                cam_id, absolute_cam_no, frame0 = cam_info
                cam_info_row['cam_id'] = cam_id
                cam_info_row['camn']   = absolute_cam_no
                cam_info_row['frame0'] = frame0
                cam_info_row.append()
                
            self.h5cam_info.flush()

        # ** 3d data **
        for q, h5table in zip([self.queue_data3d_fastest, self.queue_data3d_best],
                              [self.h5data3d_fastest, self.h5data3d_best]):
            #   clear queue
            list_of_3d_data = []
            try:
                while True:
                    list_of_3d_data.append( q.get(0) )
            except Queue.Empty:
                pass
            #   save
            if h5table is not None:
                row = h5table.row
                for data3d in list_of_3d_data:
                    corrected_framenumber, outgoing_data, cam_nos_used, mean_dist = data3d
                    cam_nos_used.sort()
                    cam_nos_used_str = ' '.join( map(str, cam_nos_used) )
                    
                    row['frame']=corrected_framenumber
                    
                    row['x']=outgoing_data[0]
                    row['y']=outgoing_data[1]
                    row['z']=outgoing_data[2]

                    row['p0']=outgoing_data[3]
                    row['p1']=outgoing_data[4]
                    row['p2']=outgoing_data[5]
                    row['p3']=outgoing_data[6]
                    row['p4']=outgoing_data[7]
                    row['p5']=outgoing_data[8]

                    row['timestamp']=outgoing_data[9]
                    
                    row['camns_used']=cam_nos_used_str
                    row['mean_dist']=mean_dist
                    row.append()

                h5table.flush()

