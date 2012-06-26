"""core runtime code for online, realtime tracking"""
# TODO:
# 1. make variable eccentricity threshold dependent on area (bigger area = lower threshold)
from __future__ import with_statement, division
import threading, time, socket, select, sys, os, copy, struct, math
import warnings, errno
import collections
import traceback
import Pyro.core
import flydra.reconstruct
import flydra.reconstruct_utils as ru
import numpy
import numpy as nx
from numpy import nan, inf
near_inf = 9.999999e20
import Queue

pytables_filt = numpy.asarray
import atexit
import pickle, copy
import pkg_resources

import motmot.utils.config
import motmot.fview_ext_trig.ttrigger
import motmot.fview_ext_trig.live_timestamp_modeler

import flydra.version
import flydra.kalman.flydra_kalman_utils as flydra_kalman_utils
import flydra.kalman.flydra_tracker
import flydra.fastgeom as geom
#import flydra.geom as geom

import flydra.data_descriptions

# ensure that pytables uses numpy:
import tables
import tables as PT
assert PT.__version__ >= '1.3.1' # bug was fixed in pytables 1.3.1 where HDF5 file kept in inconsistent state
import tables.flavor
tables.flavor.restrict_flavors(keep=['numpy'])
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

import roslib
roslib.load_manifest('rospy')
import rospy
roslib.load_manifest('ros_flydra')
from ros_flydra.msg import flydra_mainbrain_super_packet
from ros_flydra.msg import flydra_mainbrain_packet, flydra_object
from geometry_msgs.msg import Point, Vector3

import flydra.debuglock
DebugLock = flydra.debuglock.DebugLock

DO_KALMAN= True # Enables/disables Kalman filter based tracking
MIN_KALMAN_OBSERVATIONS_TO_SAVE = 10 # how many data points are required before saving trajectory?

SHOW_3D_PROCESSING_LATENCY = False

import flydra.common_variables
NETWORK_PROTOCOL = flydra.common_variables.NETWORK_PROTOCOL
ATTEMPT_DATA_RECOVERY = True
#ATTEMPT_DATA_RECOVERY = False

if os.name == 'posix':
    try:
        import posix_sched
    except ImportError, err:
        warnings.warn('Could not open posix_sched module')

Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here...

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

IMPOSSIBLE_TIMESTAMP = -10.0

PT_TUPLE_IDX_X = flydra.data_descriptions.PT_TUPLE_IDX_X
PT_TUPLE_IDX_Y = flydra.data_descriptions.PT_TUPLE_IDX_Y
PT_TUPLE_IDX_FRAME_PT_IDX = flydra.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX
PT_TUPLE_IDX_CUR_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_CUR_VAL_IDX
PT_TUPLE_IDX_MEAN_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_MEAN_VAL_IDX
PT_TUPLE_IDX_SUMSQF_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_SUMSQF_VAL_IDX

########
# persistent configuration data ( implementation in motmot.utils.config )
def get_rc_params():
    defaultParams = {
        'frames_per_second'  : 100.0,
        'hypothesis_test_max_acceptable_error' : 50.0,
        'kalman_model' :'EKF mamarama, units: mm',
        'max_reconstruction_latency_sec':0.06, # 60 msec
        'max_N_hypothesis_test':3,
        }
    fviewrc_fname = motmot.utils.config.rc_fname(filename='mainbrainrc',
                                                 dirname='.flydra')
    rc_params = motmot.utils.config.get_rc_params(fviewrc_fname,
                                                  defaultParams)
    return rc_params
def save_rc_params():
    save_fname = motmot.utils.config.rc_fname(must_already_exist=False,
                                              filename='mainbrainrc',
                                              dirname='.flydra')
    motmot.utils.config.save_rc_params(save_fname,rc_params)
rc_params = get_rc_params()
max_reconstruction_latency_sec = rc_params['max_reconstruction_latency_sec']
max_N_hypothesis_test =  rc_params['max_N_hypothesis_test']
########

XXX_framenumber = 0

class MainBrainKeeper:
    def __init__(self):
        self.kept = []
        atexit.register(self.atexit)
    def register(self, mainbrain_instance ):
        self.kept.append( mainbrain_instance )
    def atexit(self):
        for k in self.kept:
            k.quit() # closes hdf5 file and closes cameras

main_brain_keeper = MainBrainKeeper() # global to close MainBrain instances upon exit

class LockedValue:
    def __init__(self,initial_value=None):
        self.lock = threading.Lock()
        self._val = initial_value
        self._q = Queue.Queue()
    def set(self,value):
        self._q.put( value )
    def get(self):
        try:
            while 1:
                self._val = self._q.get_nowait()
        except Queue.Empty:
            pass
        return self._val

best_realtime_data=None

try:
    hostname = socket.gethostbyname('mainbrain')
except:
    hostname = socket.gethostbyname(socket.gethostname())

downstream_hosts = []

if 0:
    downstream_hosts.append( ('192.168.1.199',28931) ) # projector
if 0:
    downstream_hosts.append( ('127.0.0.1',28931) ) # self
if 0:
    downstream_hosts.append( ('192.168.1.151',28931) ) # brain1

downstream_kalman_hosts = []
if 1:
    downstream_kalman_hosts.append( ('127.0.0.1',28931) ) # localhost
    downstream_kalman_hosts.append( ('192.168.10.41',28931) ) # wtstim
#    downstream_kalman_hosts.append( ('255.255.255.255',28931) ) # broadcast to every device on subnet
if 0:
    downstream_kalman_hosts.append( ('astraw-office.kicks-ass.net',28931) ) # send off subnet

outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2D data format for PyTables:
Info2D = flydra.data_descriptions.Info2D

TextLogDescription = flydra.data_descriptions.TextLogDescription

class CamSyncInfo(PT.IsDescription):
    cam_id = PT.StringCol(256,pos=0)
    camn   = PT.UInt16Col(pos=1)
    frame0 = PT.FloatCol(pos=2)
    hostname = PT.StringCol(2048,pos=0)

class HostClockInfo(PT.IsDescription):
    remote_hostname  = PT.StringCol(255,pos=0)
    start_timestamp  = PT.FloatCol(pos=1)
    remote_timestamp = PT.FloatCol(pos=2)
    stop_timestamp   = PT.FloatCol(pos=3)

class TriggerClockInfo(PT.IsDescription):
    start_timestamp  = PT.FloatCol(pos=0)
    framecount       = PT.Int64Col(pos=1)
    tcnt             = PT.UInt16Col(pos=2)
    stop_timestamp   = PT.FloatCol(pos=3)

class MovieInfo(PT.IsDescription):
    cam_id             = PT.StringCol(16,pos=0)
    filename           = PT.StringCol(255,pos=1)
    approx_start_frame = PT.Int64Col(pos=2)
    approx_stop_frame  = PT.Int64Col(pos=3)

class Info3D(PT.IsDescription):
    # This is for the non-Kalman version.
    frame      = PT.Int64Col(pos=0)

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

FilteredObservations = flydra_kalman_utils.FilteredObservations
kalman_observations_2d_idxs_type = flydra_kalman_utils.kalman_observations_2d_idxs_type

h5_obs_names = PT.Description(FilteredObservations().columns)._v_names

# allow rapid building of numpy.rec.array:
Info2DCol_description = PT.Description(Info2D().columns)._v_nestedDescr

def encode_data_packet( corrected_framenumber,
                        line3d_valid,
                        outgoing_data,
                        min_mean_dist):

    # This is for the non-Kalman data. See
    # kalman.flydra_tracker.Tracker.encode_data_packet and
    # kalman.data_packets.encode_data_packet() for Kalman version.

    fmt = '<iBfffffffffdf' # XXX I guess this no longer works -- what's the B for? 20071011
    packable_data = list(outgoing_data)
    if not line3d_valid:
        packable_data[3:9] = 0,0,0,0,0,0
    packable_data.append( min_mean_dist )
    try:
        data_packet = struct.pack(fmt,
                                  corrected_framenumber,
                                  line3d_valid,
                                  *packable_data)
    except SystemError, x:
        print 'fmt',fmt
        print 'corrected_framenumber',corrected_framenumber
        print 'line3d_valid',line3d_valid
        print 'packable_data',packable_data
        raise
    return data_packet

def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def get_best_realtime_data():
    global best_realtime_data
    data = best_realtime_data
    best_realtime_data = None
    return data

##def DEBUG(msg=''):
##    print msg,'line',sys._getframe().f_back.f_lineno,', thread', threading.currentThread()
##    #for t in threading.enumerate():
##    #    print '   ',t

def DEBUG(msg=''):
    return



class TimestampEchoReceiver(threading.Thread):
    def __init__(self,main_brain):
        self.main_brain = main_brain

        name = 'TimestampEchoReceiver thread'
        threading.Thread.__init__(self,name=name)

    def run(self):
        ip2hostname = {}

        timestamp_echo_fmt2 = flydra.common_variables.timestamp_echo_fmt2

        timestamp_echo_gatherer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = flydra.common_variables.timestamp_echo_gatherer_port
        timestamp_echo_gatherer.bind((hostname, port))

        last_clock_diff_measurements = collections.defaultdict(list)

        while 1:
            try:
                timestamp_echo_buf, (timestamp_echo_remote_ip,cam_port) = timestamp_echo_gatherer.recvfrom(4096)
            except Exception, err:
                print 'WARNING: unknown Exception receiving timestamp echo data:',str(err)
                continue
            except:
                print 'WARNING: unknown error (non-Exception!) receiving timestamp echo data'
                continue

            stop_timestamp = time.time()

            start_timestamp,remote_timestamp = struct.unpack(timestamp_echo_fmt2,timestamp_echo_buf)

            tlist = last_clock_diff_measurements[timestamp_echo_remote_ip]
            tlist.append( (start_timestamp,remote_timestamp,stop_timestamp) )
            if len(tlist)==100:
                if timestamp_echo_remote_ip not in ip2hostname:
                    ip2hostname[timestamp_echo_remote_ip]=socket.getfqdn(timestamp_echo_remote_ip)
                remote_hostname = ip2hostname[timestamp_echo_remote_ip]
                tarray = numpy.array(tlist)

                del tlist[:] # clear list
                start_timestamps = tarray[:,0]
                stop_timestamps = tarray[:,2]
                roundtrip_duration = stop_timestamps-start_timestamps
                # find best measurement (that with shortest roundtrip_duration)
                rowidx = numpy.argmin(roundtrip_duration)
                srs = tarray[rowidx,:]
                start_timestamp, remote_timestamp, stop_timestamp = srs
                clock_diff_msec = abs(remote_timestamp-start_timestamp)*1e3
                if clock_diff_msec > 1:
                    print '%s : clock diff: %.3f msec(measurement err: %.3f msec)'%(
                        remote_hostname,
                        clock_diff_msec,
                        roundtrip_duration[rowidx]*1e3,
                        )

                self.main_brain.queue_host_clock_info.put(  (remote_hostname,
                                                             start_timestamp,
                                                             remote_timestamp,
                                                             stop_timestamp) )
                if 0:
                    measurement_duration = roundtrip_duration[rowidx]
                    clock_diff = stop_timestamp-remote_timestamp

                    print '%s: the remote diff is %.1f msec (within 0-%.1f msec accuracy)'%(
                        remote_hostname, clock_diff*1000, measurement_duration*1000)

class TrigReceiver(threading.Thread):
    def __init__(self,main_brain):
        self.main_brain = main_brain

        name = 'TrigReceiver thread'
        threading.Thread.__init__(self,name=name)

    def run(self):
        global hostname

        trigger_network_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = flydra.common_variables.trigger_network_socket_port
        trigger_network_socket.bind((hostname, port))

        while 1: # XXX enable quit
            try:
                trig_buf, (remote_ip,cam_port) = trigger_network_socket.recvfrom(4096)
            except Exception, err:
                print 'WARNING: unknown Exception receiving trigger data:',str(err)
                continue
            except:
                print 'WARNING: unknown error (non-Exception!) receiving trigger data'
                continue

            if trig_buf=='1':
                with self.main_brain.trigger_device_lock:
                    pre_timestamp = time.time()
                    self.main_brain.trigger_device.ext_trig1 = True
                    # hmm, calling log_message is normally what the cameras do..
                    self.main_brain.remote_api.log_message('<mainbrain>',pre_timestamp,'EXTTRIG1')

            elif trig_buf=='2':
                with self.main_brain.trigger_device_lock:
                    pre_timestamp = time.time()
                    self.main_brain.trigger_device.ext_trig2 = True
                    # hmm, calling log_message is normally what the cameras do..
                    self.main_brain.remote_api.log_message('<mainbrain>',pre_timestamp,'EXTTRIG2')

            elif trig_buf=='3':
                with self.main_brain.trigger_device_lock:
                    pre_timestamp = time.time()
                    self.main_brain.trigger_device.ext_trig3 = True
                    # hmm, calling log_message is normally what the cameras do..
                    self.main_brain.remote_api.log_message('<mainbrain>',pre_timestamp,'EXTTRIG3')

class CoordRealReceiver(threading.Thread):
    # called from CoordinateProcessor thread
    def __init__(self,quit_event):
        global hostname

        self.quit_event = quit_event
        self.socket_lock = threading.Lock()

        self.out_queue = Queue.Queue()
        with self.socket_lock:
            self.listen_sockets = {}
            self.server_sockets = {}

        name = 'CoordRealReceiver thread'
        threading.Thread.__init__(self,name=name)

    def add_socket(self,cam2mainbrain_data_port,cam_id):
        global hostname

        if NETWORK_PROTOCOL == 'udp':
            sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockobj.bind((hostname, cam2mainbrain_data_port))
            sockobj.setblocking(0)
            with self.socket_lock:
                self.listen_sockets[sockobj]=cam_id
        elif NETWORK_PROTOCOL == 'tcp':
            sockobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockobj.bind((hostname, cam2mainbrain_data_port))
            sockobj.listen(1)
            sockobj.setblocking(0)
            with self.socket_lock:
                self.server_sockets[ sockobj ] = cam_id
        else:
            raise ValueError('unknown NETWORK_PROTOCOL')
    def remove_socket(self,cam_id):
        with self.socket_lock:
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

    def get_data(self):
        Q = self.out_queue
        L = []

        try:
            L.append( Q.get(1,.1) ) # block for 0.1 second timeout for the first item
            while 1:
                # don't wait for next items, but collect them if they're there
                L.append( Q.get_nowait() )
        except Queue.Empty:
            pass
        return L

    # called from CoordRealReceiver thread
    def run(self):
        timeout=.1
        empty_list = []
        BENCHMARK_2D_GATHER = False
        if BENCHMARK_2D_GATHER:
            header_fmt = '<ddliI'
            header_size = struct.calcsize(header_fmt)
        while not self.quit_event.isSet():
            if NETWORK_PROTOCOL == 'tcp':
                with self.socket_lock:
                    listen_sockets = self.server_sockets.keys()
                try:
                    in_ready, out_ready, exc_ready = select.select( listen_sockets,
                                                                    empty_list, empty_list, 0.0)
                except select.error, exc:
                    print 'select.error on server socket, ignoring...'
                except socket.error, exc:
                    print 'socket.error on server socket, ignoring...'
                else:
                    for sockobj in in_ready:
                        with self.socket_lock:
                            cam_id = self.server_sockets[sockobj]
                        client_sockobj, addr = sockobj.accept()
                        client_sockobj.setblocking(0)
                        print cam_id, 'connected from',addr
                        with self.socket_lock:
                            self.listen_sockets[client_sockobj]=cam_id
            DEBUG('1')
            with self.socket_lock:
                listen_sockets = self.listen_sockets.keys()
            try:
                in_ready, out_ready, exc_ready = select.select( listen_sockets,
                                                                empty_list, empty_list, timeout )
            except select.error, exc:
                print 'select.error on listen socket, ignoring...'
            except socket.error, exc:
                print 'socket.error on listen socket, ignoring...'
            else:
                if not len(in_ready):
                    continue

                # now gather all data waiting on the sockets

                for sockobj in in_ready:
                    try:
                        with self.socket_lock:
                            cam_id = self.listen_sockets[sockobj]
                    except KeyError,ValueError:
                        print 'strange - what is in my listen sockets list?',sockobj
                        # XXX camera was dropped?
                        continue

                    if NETWORK_PROTOCOL == 'udp':
                        try:
                            data, addr = sockobj.recvfrom(4096)
                        except Exception, err:
                            print 'WARNING: unknown Exception receiving UDP data:',str(err)
                            continue
                        except:
                            print 'WARNING: unknown error (non-Exception!) receiving UDP data:'
                            continue
                    elif NETWORK_PROTOCOL == 'tcp':
                        data = sockobj.recv(4096)
                    else:
                        raise ValueError('unknown NETWORK_PROTOCOL')

                    if BENCHMARK_2D_GATHER:
                        header = data[:header_size]
                        if len(header) != header_size:
                            # incomplete header buffer
                            break
                        # this timestamp is the remote camera's timestamp
                        (timestamp, camn_received_time, framenumber,
                         n_pts,n_frames_skipped) = struct.unpack(header_fmt,header)
                        recv_latency_msec = (time.time()-camn_received_time)*1e3
                        print 'recv_latency_msec % 3.1f'%recv_latency_msec
                    self.out_queue.put((cam_id, data ))

class CoordinateSender(threading.Thread):
      """a class to send realtime coordinate data from a separate thread"""
      def __init__(self,my_queue,my_ros_queue,quit_event, publish_ros):
          self.my_queue = my_queue
          self.my_ros_queue = my_ros_queue
          self.quit_event = quit_event
          self.publish_ros = publish_ros
          self.pub = None
          name = 'CoordinateSender thread'
          threading.Thread.__init__(self,name=name)
      def run(self):
          global downstream_kalman_hosts
          out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
          block = 1
          timeout = .1
          encode_super_packet = flydra.kalman.data_packets.encode_super_packet

          if self.publish_ros:
              self.pub = rospy.Publisher(
                                    'flydra_mainbrain_super_packets',
                                    flydra_mainbrain_super_packet)

          while not self.quit_event.isSet():
              packets = []
              packets.append( self.my_queue.get() )
              while 1:
                  try:
                      packets.append( self.my_queue.get_nowait() )
                  except Queue.Empty:
                      break
              # now packets is a list of all recent data
              super_packet = encode_super_packet( packets )
              for downstream_host in downstream_kalman_hosts:
                  outgoing_UDP_socket.sendto(super_packet,downstream_host)

              if self.publish_ros:
                  ros_packets = []
                  ros_packets.append( self.my_ros_queue.get() )
                  while 1:
                      try:
                          ros_packets.append( self.my_ros_queue.get_nowait() )
                      except Queue.Empty:
                          break
                  ros_super_packet = flydra_mainbrain_super_packet(packets=ros_packets)
                  self.pub.publish(ros_super_packet)


class CoordinateProcessor(threading.Thread):
    def __init__(self,main_brain,save_profiling_data=False,
                 debug_level=None,
                 show_sync_errors=True,
                 show_overall_latency=None,
                 publish_ros=False):
        global hostname
        self.main_brain = main_brain
        self.debug_level = debug_level
        self.show_overall_latency = show_overall_latency
        self.publish_ros = publish_ros

        self.save_profiling_data = save_profiling_data
        if self.save_profiling_data:
            self.data_dict_queue = []

        self.realtime_kalman_data_queue = Queue.Queue()
        self.realtime_ros_packets = Queue.Queue()

        self.cam_ids = []
        self.cam2mainbrain_data_ports = []
        self.absolute_cam_nos = [] # a.k.a. "camn"
        self.last_timestamps = []
        self.last_framenumbers_delay = []
        self.last_framenumbers_skip = []
        if ATTEMPT_DATA_RECOVERY:
            #self.request_data_lock = DebugLock('request_data_lock',True) # protect request_data
            self.request_data_lock = threading.Lock() # protect request_data
            self.request_data = {}
        self.cam_id2cam_no = {}
        self.camn2cam_id = {}
        self.reconstructor = None
        self.reconstructor_meters = None
        self.tracker = None
        self.show_sync_errors=show_sync_errors

        self.ip2hostname = {}

        self.tracker_lock = threading.Lock()
        #self.tracker_lock = DebugLock('tracker_lock',verbose=True)

        self.all_data_lock = threading.Lock()
        #self.all_data_lock = DebugLock('all_data_lock',verbose=False)
        self.quit_event = threading.Event()

        self.max_absolute_cam_nos = -1

        self.general_save_info = {}

        self.realreceiver = CoordRealReceiver(self.quit_event)
        self.realreceiver.setDaemon(True)
        self.realreceiver.start()

        cs = CoordinateSender(
                self.realtime_kalman_data_queue,
                self.realtime_ros_packets,
                self.quit_event,
                self.publish_ros)
        cs.setDaemon(True)
        cs.start()

        name = 'CoordinateProcessor thread'
        threading.Thread.__init__(self,name=name)

    def get_cam2mainbrain_data_port(self,cam_id):
        with self.all_data_lock:
            i = self.cam_ids.index( cam_id )
            cam2mainbrain_data_port = self.cam2mainbrain_data_ports[i]
        return cam2mainbrain_data_port

    def get_general_cam_info(self):
        with self.all_data_lock:
            result = self.general_save_info.copy()
        return result

    def get_missing_data_dict(self):
        # called from main thread, must lock data in realtime coord thread
        result_by_camn = {}
        with self.request_data_lock:
            for absolute_cam_no,tmp_queue in self.request_data.iteritems():
                list_of_missing_framenumbers = []
                cam_id = None
                framenumber_offset = None
                try:
                    while 1:
                        value = tmp_queue.get_nowait()
                        this_cam_id, this_framenumber_offset, this_list = value
                        if cam_id is None:
                            cam_id = this_cam_id
                        if framenumber_offset is None:
                            framenumber_offset = this_framenumber_offset

                        assert cam_id == this_cam_id # make sure given camn comes from single cam_id
                        assert framenumber_offset == this_framenumber_offset

                        list_of_missing_framenumbers.extend( this_list )
                except Queue.Empty:
                    pass
                if len(list_of_missing_framenumbers):
                    result_by_camn[absolute_cam_no] = cam_id, framenumber_offset, list_of_missing_framenumbers
        return result_by_camn

    def set_reconstructor(self,r):
        # called from main thread, must lock to send to realtime coord thread
        with self.all_data_lock:
            self.reconstructor = r

        if r is None:
            self.reconstructor_meters = None
            return

        # get version that operates in meters
        scale_factor = self.reconstructor.get_scale_factor()
        self.reconstructor_meters = self.reconstructor.get_scaled(scale_factor)

    def set_new_tracker(self,kalman_model=None):
        # called from main thread, must lock to send to realtime coord thread
        scale_factor = self.reconstructor.get_scale_factor()
        tracker = flydra.kalman.flydra_tracker.Tracker(self.reconstructor_meters,
                                                       scale_factor=scale_factor,
                                                       kalman_model=kalman_model)
        tracker.set_killed_tracker_callback( self.enqueue_finished_tracked_object )
        with self.tracker_lock:
            if self.tracker is not None:
                self.tracker.kill_all_trackers() # save (if necessary) all old data
            self.tracker = tracker # bind to name, replacing old tracker
            if self.save_profiling_data:
                tracker = copy.copy(self.tracker)
                tracker.kill_tracker_callbacks = []
                if 1:
                    raise NotImplementedError('')
                else:
                    # this is the old call, it needs to be fixed...
                    tracker.live_tracked_objects = []
                tracker.dead_tracked_objects = []
                self.data_dict_queue.append( ('tracker',tracker))

    def enqueue_finished_tracked_object(self, tracked_object ):
        # this is from called within the realtime coords thread
        if self.main_brain.is_saving_data():
            if self.debug_level.isSet():
                print 'saving finished kalman object with %d frames'%(len(tracked_object.frames),)
            self.main_brain.queue_data3d_kalman_estimates.put(
                (tracked_object.obj_id,
                 tracked_object.frames, tracked_object.xhats, tracked_object.Ps,
                 tracked_object.timestamps,
                 tracked_object.observations_frames, tracked_object.observations_data,
                 tracked_object.observations_2d, tracked_object.observations_Lcoords,
                 ) )

    def connect(self,cam_id,cam_hostname):

        # called from Remote-API thread on camera connect

        global hostname

        assert not self.main_brain.is_saving_data()

        with self.all_data_lock:
            self.cam_ids.append(cam_id)

            # find cam2mainbrain_data_port
            if len(self.cam2mainbrain_data_ports)>0:
                cam2mainbrain_data_port = max(self.cam2mainbrain_data_ports)+1
            else:
                cam2mainbrain_data_port = flydra.common_variables.min_cam2mainbrain_data_port # arbitrary number
            self.cam2mainbrain_data_ports.append( cam2mainbrain_data_port )

            # find absolute_cam_no
            self.max_absolute_cam_nos += 1
            absolute_cam_no = self.max_absolute_cam_nos
            self.absolute_cam_nos.append( absolute_cam_no )

            self.camn2cam_id[absolute_cam_no] = cam_id
            self.cam_id2cam_no[cam_id] = absolute_cam_no

            # create and bind socket to listen to
            self.realreceiver.add_socket(cam2mainbrain_data_port,cam_id)
            self.last_timestamps.append(IMPOSSIBLE_TIMESTAMP) # arbitrary impossible number
            self.last_framenumbers_delay.append(-1) # arbitrary impossible number
            self.last_framenumbers_skip.append(-1) # arbitrary impossible number
            self.general_save_info[cam_id] = {'absolute_cam_no':absolute_cam_no,
                                              'frame0':IMPOSSIBLE_TIMESTAMP}
            self.main_brain.queue_cam_info.put(  (cam_id, absolute_cam_no, IMPOSSIBLE_TIMESTAMP, cam_hostname) )
        return cam2mainbrain_data_port

    def disconnect(self,cam_id):
        cam_idx = self.cam_ids.index( cam_id )
        with self.all_data_lock:
            del self.cam_ids[cam_idx]
            del self.cam2mainbrain_data_ports[cam_idx]
            del self.absolute_cam_nos[cam_idx]
            self.realreceiver.remove_socket(cam_id)
            del self.last_timestamps[cam_idx]
            del self.last_framenumbers_delay[cam_idx]
            del self.last_framenumbers_skip[cam_idx]
            del self.general_save_info[cam_id]

    def quit(self):
        # called from outside of thread to quit the thread
        if self.save_profiling_data:
            fname = "data_for_kalman_profiling.pkl"
            fullpath = os.path.abspath(fname)
            print "saving data for profiling to %s"%fullpath
            to_save = self.data_dict_queue
            save_fd = open(fullpath,mode="wb")
            pickle.dump( to_save, save_fd )
            save_fd.close()
            print "done saving"
        self.quit_event.set()
        self.join() # wait until CoordReveiver thread quits

    def OnSynchronize(self, cam_idx, cam_id, framenumber, timestamp,
                      realtime_coord_dict, timestamp_check_dict,
                      realtime_kalman_coord_dict,
                      oldest_timestamp_by_corrected_framenumber,
                      new_data_framenumbers):

        if self.main_brain.is_saving_data():
            print 'ERROR: re-synchronized while saving data!'
            return

        if self.last_timestamps[cam_idx] != IMPOSSIBLE_TIMESTAMP:
            print cam_id,'(re)synchronized'
            # discard all previous data
            for k in realtime_coord_dict.keys():
                del realtime_coord_dict[k]
                del timestamp_check_dict[k]
            for k in realtime_kalman_coord_dict.keys():
                del realtime_kalman_coord_dict[k]
            for k in oldest_timestamp_by_corrected_framenumber.keys():
                del oldest_timestamp_by_corrected_framenumber[k]
            new_data_framenumbers.clear()

        #else:
        #    print cam_id,'first 2D coordinates received'

        # make new absolute_cam_no to indicate new synchronization state
        self.max_absolute_cam_nos += 1
        absolute_cam_no = self.max_absolute_cam_nos
        self.absolute_cam_nos[cam_idx] = absolute_cam_no

        self.camn2cam_id[absolute_cam_no] = cam_id
        self.cam_id2cam_no[cam_id] = absolute_cam_no

        self.general_save_info[cam_id]['absolute_cam_no']=absolute_cam_no
        self.general_save_info[cam_id]['frame0']=timestamp

        self.main_brain.queue_cam_info.put(  (cam_id, absolute_cam_no, timestamp) )

    def run(self):
        """main loop of CoordinateProcessor"""
        global downstream_hosts, best_realtime_data
        global outgoing_UDP_socket
        global XXX_framenumber

        if os.name == 'posix':
            try:
                max_priority = posix_sched.get_priority_max( posix_sched.FIFO )
                sched_params = posix_sched.SchedParam(max_priority)
                posix_sched.setscheduler(0, posix_sched.FIFO, sched_params)
                print 'excellent, 3D reconstruction thread running in maximum prioity mode'
            except Exception, x:
                print 'WARNING: could not run in maximum priority mode (PID %d): %s'%(os.getpid(),str(x))

        header_fmt = '<ddliI'
        header_size = struct.calcsize(header_fmt)
        pt_fmt = '<dddddddddBBddBddddddBdd' # keep in sync with camnode.py
        pt_size = struct.calcsize(pt_fmt)

        realtime_coord_dict = {}
        timestamp_check_dict = {}
        realtime_kalman_coord_dict = collections.defaultdict(dict)
        oldest_timestamp_by_corrected_framenumber = {}

        new_data_framenumbers = set()

        no_point_tuple = (nan,nan,nan,nan,nan,nan,nan,nan,nan,False,0,0,0,0)

        convert_format = flydra_kalman_utils.convert_format # shorthand

        max_error = self.main_brain.get_hypothesis_test_max_error()

        if NETWORK_PROTOCOL == 'tcp':
            old_data = {}

        debug_drop_fd = None

        while not self.quit_event.isSet():
            incoming_2d_data = self.realreceiver.get_data() # blocks
            if not len(incoming_2d_data):
                continue

            new_data_framenumbers.clear()

            BENCHMARK_GATHER=False
            if BENCHMARK_GATHER:
                incoming_remote_received_timestamps = []

            with self.all_data_lock:
            #self.all_data_lock.acquire(latency_warn_msec=1.0)

                deferred_2d_data = []
                for cam_id, newdata in incoming_2d_data:

                    try:
                        cam_idx = self.cam_ids.index(cam_id)
                    except ValueError, err:
                        print 'ERROR: ignoring lost cam_id %s'%(cam_id,)
                        continue
                    absolute_cam_no = self.absolute_cam_nos[cam_idx]

                    if NETWORK_PROTOCOL == 'tcp':
                        data = old_data.get( sockobj, '')
                        data += newdata
                    else:
                        data = newdata

                    while len(data):
                        header = data[:header_size]
                        if len(header) != header_size:
                            # incomplete header buffer
                            break
                        # this raw_timestamp is the remote camera's timestamp (?? from the driver, not the host clock??)
                        (raw_timestamp, camn_received_time, raw_framenumber,
                         n_pts,n_frames_skipped) = struct.unpack(header_fmt,header)
                        if BENCHMARK_GATHER:
                            incoming_remote_received_timestamps.append( camn_received_time )

                        DEBUG_DROP = self.main_brain.remote_api.cam_info[cam_id]['scalar_control_info']['debug_drop']
                        if DEBUG_DROP:
                            if debug_drop_fd is None:
                                debug_drop_fd = open('debug_framedrop.txt',mode='w')
                            debug_drop_fd.write('%d,%d\n'%(raw_framenumber,n_pts))
                        points_in_pluecker_coords_meters = []
                        points_undistorted = []
                        points_distorted = []
                        if len(data) < header_size + n_pts*pt_size:
                            # incomplete point info
                            break
                        predicted_framenumber = n_frames_skipped + self.last_framenumbers_skip[cam_idx] + 1
                        if raw_framenumber<predicted_framenumber:
                            print 'cam_id',cam_id
                            print 'raw_framenumber',raw_framenumber
                            print 'n_frames_skipped',n_frames_skipped
                            print 'predicted_framenumber',predicted_framenumber
                            print 'self.last_framenumbers_skip[cam_idx]',self.last_framenumbers_skip[cam_idx]
                            raise RuntimeError('got framenumber already received or skipped!')
                        elif raw_framenumber>predicted_framenumber:
                            if not self.last_framenumbers_skip[cam_idx]==-1:
                                # this is not the first frame

                                # probably because network buffer filled up before we emptied it
                                print '  WARNING: frame data loss %s'%(cam_id,)

                            if ATTEMPT_DATA_RECOVERY:
                                if not self.last_framenumbers_skip[cam_idx]==-1:
                                    # this is not the first frame
                                    missing_frame_numbers = range(
                                        self.last_framenumbers_skip[cam_idx]+1,
                                        raw_framenumber)

                                    with self.request_data_lock:
                                        tmp_queue = self.request_data.setdefault(absolute_cam_no,Queue.Queue())

                                    tmp_framenumber_offset = self.main_brain.timestamp_modeler.get_frame_offset(cam_id)
                                    tmp_queue.put( (cam_id,  tmp_framenumber_offset, missing_frame_numbers) )
                                    del tmp_framenumber_offset
                                    del tmp_queue # drop reference to queue
                                    del missing_frame_numbers

                        self.last_framenumbers_skip[cam_idx]=raw_framenumber
                        start=header_size
                        if n_pts:
                            # valid points
                            for frame_pt_idx in range(n_pts):
                                end=start+pt_size
                                (x_distorted,y_distorted,area,slope,eccentricity,
                                 p1,p2,p3,p4,line_found,slope_found,
                                 x_undistorted,y_undistorted,
                                 ray_valid,
                                 ray0, ray1, ray2, ray3, ray4, ray5, # pluecker coords from cam center to detected point
                                 cur_val, mean_val, sumsqf_val,
                                 )= struct.unpack(pt_fmt,data[start:end])
                                # nan cannot get sent across network in platform-independent way
                                if not line_found:
                                    p1,p2,p3,p4 = nan,nan,nan,nan
                                if slope == near_inf:
                                    slope = inf
                                if eccentricity == near_inf:
                                    eccentricity = inf
                                if not slope_found:
                                    slope = nan

                                # Keep in sync with kalmanize.py and data_descriptions.py
                                pt_undistorted = (x_undistorted,y_undistorted,
                                                  area,slope,eccentricity,
                                                  p1,p2,p3,p4, line_found, frame_pt_idx,
                                                  cur_val, mean_val, sumsqf_val)
                                pt_distorted = (x_distorted,y_distorted,
                                                area,slope,eccentricity,
                                                p1,p2,p3,p4, line_found, frame_pt_idx,
                                                cur_val, mean_val, sumsqf_val)
                                if ray_valid:
                                    points_in_pluecker_coords_meters.append( (pt_undistorted,
                                                                              geom.line_from_HZline((ray0,ray1,
                                                                                                            ray2,ray3,
                                                                                                            ray4,ray5))
                                                                              ))
                                points_undistorted.append( pt_undistorted )
                                points_distorted.append( pt_distorted )
                                start=end
                        else:
                            # no points found
                            end = start
                            # append non-point to allow correlation of
                            # timestamps with frame number
                            points_distorted.append( no_point_tuple )
                            points_undistorted.append( no_point_tuple )
                        data = data[end:]

                        # ===================================================

                        # XXX hack? make data available via cam_dict
                        cam_dict = self.main_brain.remote_api.cam_info[cam_id]
                        with cam_dict['lock']:
                            cam_dict['points_distorted']=points_distorted

                        # Use camn_received_time to determine sync
                        # info. This avoids 2 potential problems:
                        #  * using raw_timestamp can fail if the
                        #    camera drivers don't provide useful data
                        #  * using time.time() can fail if the network
                        #    latency jitter is on the order of the
                        #    inter frame interval.
                        tmp = self.main_brain.timestamp_modeler.register_frame(
                            cam_id,raw_framenumber,camn_received_time,full_output=True)
                        trigger_timestamp, corrected_framenumber, did_frame_offset_change = tmp
                        if did_frame_offset_change:
                            self.OnSynchronize( cam_idx, cam_id, raw_framenumber, trigger_timestamp,
                                                realtime_coord_dict,
                                                timestamp_check_dict,
                                                realtime_kalman_coord_dict,
                                                oldest_timestamp_by_corrected_framenumber,
                                                new_data_framenumbers )

                        self.last_timestamps[cam_idx]=trigger_timestamp
                        self.last_framenumbers_delay[cam_idx]=raw_framenumber
                        XXX_framenumber = corrected_framenumber

                        if self.main_brain.is_saving_data():
                            for point_tuple in points_distorted:
                                # Save 2D data (even when no point found) to allow
                                # temporal correlation of movie frames to 2D data.
                                frame_pt_idx = point_tuple[PT_TUPLE_IDX_FRAME_PT_IDX]
                                cur_val = point_tuple[PT_TUPLE_IDX_CUR_VAL_IDX]
                                mean_val = point_tuple[PT_TUPLE_IDX_MEAN_VAL_IDX]
                                sumsqf_val = point_tuple[PT_TUPLE_IDX_SUMSQF_VAL_IDX]
                                if corrected_framenumber is None:
                                    # don't bother saving if we don't know when it was from
                                    continue
                                deferred_2d_data.append((absolute_cam_no, # defer saving to later
                                                         corrected_framenumber,
                                                         trigger_timestamp,camn_received_time)
                                                        +point_tuple[:5]
                                                        +(frame_pt_idx,cur_val,mean_val,sumsqf_val))
                        # save new frame data

                        if corrected_framenumber not in realtime_coord_dict:
                            realtime_coord_dict[corrected_framenumber] = {}
                            timestamp_check_dict[corrected_framenumber] = {}

                        # For hypothesis testing: attempt 3D reconstruction of 1st point from each 2D view
                        realtime_coord_dict[corrected_framenumber][cam_id]= points_undistorted[0]
                        #timestamp_check_dict[corrected_framenumber][cam_id]= camn_received_time
                        timestamp_check_dict[corrected_framenumber][cam_id]= trigger_timestamp

                        if len( points_in_pluecker_coords_meters):
                            # save all 3D Pluecker coordinates for Kalman filtering
                            realtime_kalman_coord_dict[corrected_framenumber][absolute_cam_no]=(
                                points_in_pluecker_coords_meters)

                        if n_pts:
                            inc_val = 1
                        else:
                            inc_val = 0

                        if corrected_framenumber in oldest_timestamp_by_corrected_framenumber:
                            orig_timestamp,n = oldest_timestamp_by_corrected_framenumber[ corrected_framenumber ]
                            if orig_timestamp is None:
                                oldest = trigger_timestamp # this may also be None, but eventually won't be
                            else:
                                oldest = min(trigger_timestamp, orig_timestamp)
                            oldest_timestamp_by_corrected_framenumber[ corrected_framenumber ] = (oldest,n+inc_val)
                            del oldest, n, orig_timestamp
                        else:
                            oldest_timestamp_by_corrected_framenumber[ corrected_framenumber ] = trigger_timestamp, inc_val

                        new_data_framenumbers.add( corrected_framenumber ) # insert into set

                    # preserve unprocessed data
                    if NETWORK_PROTOCOL == 'tcp':
                        old_data[sockobj] = data

                if BENCHMARK_GATHER:
                    incoming_remote_received_timestamps = numpy.array(incoming_remote_received_timestamps)
                    min_incoming_remote_timestamp = incoming_remote_received_timestamps.min()
                    max_incoming_remote_timestamp = incoming_remote_received_timestamps.max()
                    finish_packet_sorting_time = time.time()
                    min_packet_gather_dur = finish_packet_sorting_time-max_incoming_remote_timestamp
                    max_packet_gather_dur = finish_packet_sorting_time-min_incoming_remote_timestamp
                    print 'proc dur: % 3.1f % 3.1f'%(min_packet_gather_dur*1e3,
                                                     max_packet_gather_dur*1e3)

                finished_corrected_framenumbers = [] # for quick deletion

                ########################################################################

                # Now we've grabbed all data waiting on network. Now it's
                # time to calculate 3D info.

                # XXX could go for latest data first to minimize latency
                # on that data.

                ########################################################################

                for corrected_framenumber in new_data_framenumbers:
                    oldest_camera_timestamp, n = oldest_timestamp_by_corrected_framenumber[ corrected_framenumber ]
                    if oldest_camera_timestamp is None:
                        ## print 'no latency estimate available -- skipping 3D reconstruction'
                        continue
                    if (time.time() - oldest_camera_timestamp) > max_reconstruction_latency_sec:
                        #print 'maximum reconstruction latency exceeded -- skipping 3D reconstruction'
                        continue

                    data_dict = realtime_coord_dict[corrected_framenumber]
                    if len(data_dict)==len(self.cam_ids): # all camera data arrived

                        if self.debug_level.isSet():
                            print 'frame %d'%(corrected_framenumber,)

                        if SHOW_3D_PROCESSING_LATENCY:
                            start_3d_proc = time.time()

                        # mark for deletion out of data queue
                        finished_corrected_framenumbers.append( corrected_framenumber )

                        if self.reconstructor is None:
                            # can't do any 3D math without calibration information
                            best_realtime_data = None
                            continue

                        if DO_KALMAN:
                            with self.tracker_lock:
                                if self.tracker is None: # tracker isn't instantiated yet...
                                    best_realtime_data = None
                                    continue

                                pluecker_coords_by_camn = realtime_kalman_coord_dict[corrected_framenumber]

                                if self.save_profiling_data:
                                    dumps = pickle.dumps(pluecker_coords_by_camn)
                                    self.data_dict_queue.append(('gob',(corrected_framenumber,
                                                                        dumps,
                                                                        self.camn2cam_id)))
                                pluecker_coords_by_camn = self.tracker.calculate_a_posteriori_estimates(
                                    corrected_framenumber,
                                    pluecker_coords_by_camn,
                                    self.camn2cam_id)

                                if self.debug_level.isSet():
                                    print '%d live objects:'%(self.tracker.live_tracked_objects.how_many_are_living(),),
                                    results = self.tracker.live_tracked_objects.rmap( 'get_most_recent_data' ) # reverse map
                                    Xs = []
                                    for result in results:
                                        if result is None:
                                            continue
                                        obj_id,last_xhat,P = result
                                        print last_xhat[:3]
                                    print

                                if self.save_profiling_data:
                                    self.data_dict_queue.append(('ntrack',self.tracker.live_tracked_objects.how_many_are_living()))

                                now = time.time()
                                if SHOW_3D_PROCESSING_LATENCY:
                                    start_3d_proc_a = now
                                if self.show_overall_latency.isSet():
                                    oldest_camera_timestamp, n = oldest_timestamp_by_corrected_framenumber[ corrected_framenumber ]
                                    if n>0:
                                        if 0:
                                            print 'overall latency %d: %.1f msec (oldest: %s now: %s)'%(
                                                n,
                                                (now-oldest_camera_timestamp)*1e3,
                                                repr(oldest_camera_timestamp),
                                                repr(now),
                                                )
                                        else:

                                            print 'overall latency (%d camera detected 2d points): %.1f msec (note: may exclude camera->camera computer latency)'%(
                                                n,
                                                (now-oldest_camera_timestamp)*1e3,
                                                )

                                if 1:
                                    # The above calls
                                    # self.enqueue_finished_tracked_object()
                                    # when a tracked object is no longer
                                    # tracked.

                                    # Now, tracked objects have been updated (and their 2D data points
                                    # removed from consideration), so we can use old flydra
                                    # "hypothesis testing" algorithm on remaining data to see if there
                                    # are new objects.

                                    scale_factor = self.tracker.scale_factor
                                    results = self.tracker.live_tracked_objects.rmap( 'get_most_recent_data' ) # reverse map
                                    Xs = []
                                    for result in results:
                                        if result is None:
                                            continue
                                        obj_id,last_xhat,P = result
                                        X = last_xhat[0]/scale_factor, last_xhat[1]/scale_factor, last_xhat[2]/scale_factor
                                        Xs.append(X)
                                    if len(Xs):
                                        best_realtime_data = Xs, 0.0
                                    else:
                                        best_realtime_data = None

                                if SHOW_3D_PROCESSING_LATENCY:
                                    start_3d_proc_b = time.time()

                                # Convert to format accepted by find_best_3d()
                                found_data_dict,first_idx_by_camn = convert_format(
                                    pluecker_coords_by_camn,
                                    self.camn2cam_id,
                                    area_threshold=0.0,
                                    only_likely=True)

                                if SHOW_3D_PROCESSING_LATENCY:
                                    if len(found_data_dict) < 2:
                                        print ' ',
                                    else:
                                        print '*',

                                if len(found_data_dict) >= 2:
                                    # Can't do any 3D math without at least 2 cameras giving good
                                    # data.
                                    try:
                                        (this_observation_orig_units, this_observation_Lcoords_orig_units, cam_ids_used,
                                         min_mean_dist) = ru.hypothesis_testing_algorithm__find_best_3d(
                                            self.reconstructor,
                                            found_data_dict,
                                            max_error,
                                            max_n_cams=max_N_hypothesis_test,
                                            )
                                    except ru.NoAcceptablePointFound, err:
                                        pass
                                    else:
                                        this_observation_camns = [self.cam_id2cam_no[cam_id] for cam_id in cam_ids_used]
                                        this_observation_idxs = [first_idx_by_camn[camn] for camn in this_observation_camns] # zero idx
                                        ####################################
                                        #  Now join found point into Tracker
                                        if self.save_profiling_data:
                                            self.data_dict_queue.append(('join',(corrected_framenumber,
                                                                                 this_observation_orig_units,
                                                                                 this_observation_Lcoords_orig_units,
                                                                                 this_observation_camns,
                                                                                 this_observation_idxs
                                                                                 )))
                                        # test for novelty
                                        believably_new = self.tracker.is_believably_new( this_observation_orig_units)
                                        if believably_new:
                                            self.tracker.join_new_obj( corrected_framenumber,
                                                                       this_observation_orig_units,
                                                                       this_observation_Lcoords_orig_units,
                                                                       this_observation_camns,
                                                                       this_observation_idxs
                                                                       )
                                if 1:
                                    if self.tracker.live_tracked_objects.how_many_are_living():
                                        data_packet = self.tracker.encode_data_packet(
                                            corrected_framenumber,
                                            oldest_camera_timestamp,now)
                                        if data_packet is not None:
                                            self.realtime_kalman_data_queue.put(data_packet)
                                        if self.publish_ros:
                                            results = self.tracker.live_tracked_objects.rmap( 'get_most_recent_data' )
                                            ros_objects = []
                                            for result in results:
                                                if result is None:
                                                    continue
                                                obj_id,xhat,P = result
                                                this_ros_object = flydra_object(obj_id=obj_id,
                                                                                position=Point(*xhat[:3]),
                                                                                velocity=Vector3(*xhat[3:6]),
                                                                                posvel_covariance_diagonal=numpy.diag(P)[:6].tolist())
                                                ros_objects.append( this_ros_object )
                                            ros_packet = flydra_mainbrain_packet(
                                                framenumber=corrected_framenumber,
                                                reconstruction_stamp=rospy.Time.from_sec(now),
                                                acquire_stamp=rospy.Time.from_sec(oldest_camera_timestamp),
                                                objects = ros_objects)
                                            self.realtime_ros_packets.put( ros_packet )

                                if SHOW_3D_PROCESSING_LATENCY:
                                    start_3d_proc_c = time.time()

                        else: # closes "if DO_KALMAN:"

                            found_data_dict = {} # old "good" points will go in here
                            for cam_id, this_point in data_dict.iteritems():
                                if not numpy.isnan(this_point[0]): # only use if point was found
                                    found_data_dict[cam_id] = this_point[:9]

                            if len(found_data_dict) < 2:
                                # Can't do any 3D math without at least 2
                                # cameras giving good data.
                                continue

                            try:
                                # hypothesis testing algorithm
                                (X, line3d, cam_ids_used,min_mean_dist
                                 ) = ru.hypothesis_testing_algorithm__find_best_3d(
                                    self.reconstructor,
                                    found_data_dict,
                                    max_n_cams=max_N_hypothesis_test,
                                    )
                            except:
                                # this prevents us from bombing this thread...
                                print 'WARNING:'
                                traceback.print_exc()
                                print 'SKIPPED 3d calculation for this frame.'
                                continue
                            cam_nos_used = [self.cam_id2cam_no[cam_id] for cam_id in cam_ids_used]

                            if line3d is None:
                                line3d = nan, nan, nan, nan, nan, nan
                                line3d_valid = False
                            else:
                                line3d_valid = True

                            find3d_time = time.time()

                            x,y,z=X
                            outgoing_data = [x,y,z]
                            outgoing_data.extend( line3d ) # 6 component vector
                            outgoing_data.append( find3d_time )

                            if len(downstream_hosts):
                                # This is for the non-Kalman data. See
                                # tracker.encode_data_packet() for
                                # Kalman version.
                                data_packet = encode_data_packet(
                                    corrected_framenumber,
                                    line3d_valid,
                                    outgoing_data,
                                    min_mean_dist,
                                    )

                            # realtime 3d data
                            best_realtime_data = [X], min_mean_dist
                            try:
                                for downstream_host in downstream_hosts:
                                    outgoing_UDP_socket.sendto(data_packet,downstream_host)
                            except:
                                print 'WARNING: could not send 3d point data over UDP'
                                print
                            if self.main_brain.is_saving_data():
                                self.main_brain.queue_data3d_best.put( (corrected_framenumber,
                                                                        outgoing_data,
                                                                        cam_nos_used,
                                                                        min_mean_dist) )
                        if SHOW_3D_PROCESSING_LATENCY:
                            stop_3d_proc = time.time()
                            dur_3d_proc_msec = (stop_3d_proc - start_3d_proc)*1e3
                            dur_3d_proc_msec_a = (start_3d_proc_a - start_3d_proc)*1e3
                            dur_3d_proc_msec_b = (start_3d_proc_b - start_3d_proc)*1e3
                            dur_3d_proc_msec_c = (start_3d_proc_c - start_3d_proc)*1e3

                            print 'dur_3d_proc_msec % 3.1f % 3.1f % 3.1f % 3.1f'%(
                                dur_3d_proc_msec,
                                dur_3d_proc_msec_a,
                                dur_3d_proc_msec_b,
                                dur_3d_proc_msec_c)

                for finished in finished_corrected_framenumbers:
                    if 1:
                        #check that timestamps are in reasonable agreement (low priority)
                        if 0:
                            timestamps_by_cam_id = numpy.array(timestamp_check_dict[finished].values())
                            for xy in timestamp_check_dict[finished].iteritems():
                                print repr(xy)

                        if 1:
                            diff_from_start = []
                            for cam_id, tmp_trigger_timestamp in timestamp_check_dict[finished].iteritems():
                                diff_from_start.append( tmp_trigger_timestamp )
                            timestamps_by_cam_id = numpy.array( diff_from_start )

                        if self.show_sync_errors:
                            if len(timestamps_by_cam_id):
                                if numpy.max(abs(timestamps_by_cam_id - timestamps_by_cam_id[0])) > 0.005:
                                    print 'timestamps off by more than 5 msec -- synchronization error'

                    del realtime_coord_dict[finished]
                    del timestamp_check_dict[finished]
                    try:
                        del realtime_kalman_coord_dict[finished]
                    except KeyError:
                        pass

                # Clean up old frame records to save RAM.

                # This is only needed when multiple cameras are not
                # synchronized, (When camera-camera frame
                # correspondences are unknown.)

                # XXX This probably drops unintended frames on
                # re-sync, but who cares?

                if len(realtime_coord_dict)>100:
                    print 'Cameras not synchronized or network dropping packets -- unmatched 2D data accumulating'
                    k=realtime_coord_dict.keys()
                    k.sort()

                    if 1:
                        # get one sample
                        corrected_framenumber = k[0]
                        data_dict = realtime_coord_dict[corrected_framenumber]
                        this_cam_ids = data_dict.keys()
                        missing_cam_id_guess = list(set(self.cam_ids) - set( this_cam_ids ))
                        if len(missing_cam_id_guess):
                            print ' a guess at missing cam_id(s):',list(set(self.cam_ids) - set( this_cam_ids ))

                    for ki in k[:-50]:
                        del realtime_coord_dict[ki]
                        del timestamp_check_dict[ki]

                if len(realtime_kalman_coord_dict)>100:
                    print 'deleting unused 3D data (this should be a rare occurrance)'
                    k=realtime_kalman_coord_dict.keys()
                    k.sort()
                    for ki in k[:-50]:
                        del realtime_kalman_coord_dict[ki]

                if len(oldest_timestamp_by_corrected_framenumber)>100:
                    k=oldest_timestamp_by_corrected_framenumber.keys()
                    k.sort()
                    for ki in k[:-50]:
                        del oldest_timestamp_by_corrected_framenumber[ki]

                if len(deferred_2d_data):
                    self.main_brain.queue_data2d.put( deferred_2d_data )

        if DO_KALMAN:
            with self.tracker_lock:
                if self.tracker is not None:
                    self.tracker.kill_all_trackers() # save (if necessary) all old data

class MainBrain(object):
    """Handle all camera network stuff and interact with application"""

    class RemoteAPI(Pyro.core.ObjBase):

        # ================================================================
        #
        # Methods called locally
        #
        # ================================================================

        def get_version(self):
            return flydra.version.__version__

        def post_init(self, main_brain):
            """call after __init__"""
            # let Pyro handle __init__
            self.cam_info = {}
            self.cam_info_lock = threading.Lock()
            self.changed_cam_lock = threading.Lock()
            self.no_cams_connected = threading.Event()
            self.no_cams_connected.set()
            with self.changed_cam_lock:
                self.new_cam_ids = []
                self.old_cam_ids = []
            self.main_brain = main_brain

            # threading control locks
            self.quit_now = threading.Event()
            self.thread_done = threading.Event()
            self.message_queue = Queue.Queue()

        def external_get_and_clear_pending_cams(self):
            with self.changed_cam_lock:
                new_cam_ids = self.new_cam_ids
                self.new_cam_ids = []
                old_cam_ids = self.old_cam_ids
                self.old_cam_ids = []
            return new_cam_ids, old_cam_ids

        def external_get_cam_ids(self):
            with self.cam_info_lock:
                cam_ids = self.cam_info.keys()
            cam_ids.sort()
            return cam_ids

        def external_get_info(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    scalar_control_info = copy.deepcopy(cam['scalar_control_info'])
                    fqdn = cam['fqdn']
                    port = cam['port']
            return scalar_control_info, fqdn, port

        def external_get_image_fps_points(self, cam_id):
            ### XXX should extend to include lines
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    coord_and_image = cam['image']
                    fps = cam['fps']
                    points_distorted = cam['points_distorted'][:]
            # NB: points are distorted (and therefore align
            # with distorted image)
            if coord_and_image is not None:
                image_coords, image = coord_and_image
            else:
                image_coords, image = None, None
            return image, fps, points_distorted, image_coords

        def external_send_set_camera_property( self, cam_id, property_name, value):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands'].setdefault('set',{})[property_name]=value
                    old_value = cam['scalar_control_info'][property_name]
                    if type(old_value) == tuple and type(value) == int:
                        # brightness, gain, shutter
                        cam['scalar_control_info'][property_name] = (value, old_value[1], old_value[2])
                    else:
                        cam['scalar_control_info'][property_name] = value

        def external_request_image_async(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['get_im']=None

        def external_start_recording( self, cam_id, raw_file_basename):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['start_recording']=raw_file_basename

        def external_stop_recording( self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['stop_recording']=None

        def external_start_small_recording( self, cam_id,
                                            small_filebasename):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['start_small_recording']=small_filebasename

        def external_stop_small_recording( self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['stop_small_recording']=None

        def external_quit( self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['quit']=True

        def external_take_background( self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['take_bg']=None

        def external_request_missing_data(self, cam_id, camn, framenumber_offset, list_of_missing_framenumbers):
            with self.cam_info_lock:
                if cam_id not in self.cam_info:
                    # the camera was dropped, ignore this request
                    return
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']

                camn_and_list = [camn, framenumber_offset]
                camn_and_list.extend( list_of_missing_framenumbers )
                cmd_str = ' '.join(map(repr,camn_and_list))
                with cam_lock:
                    cam['commands']['request_missing']=cmd_str

        def external_clear_background( self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['clear_bg']=None

        def external_set_cal( self, cam_id, pmat, intlin, intnonlin, scale_factor):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cam['commands']['cal']= pmat, intlin, intnonlin, scale_factor
                    cam['is_calibrated'] = True
        # === thread boundary =========================================

        def listen(self,daemon):
            """thread mainloop"""
            hr = daemon.handleRequests
            while not self.quit_now.isSet():
                try:
                    hr(0.1) # block on select for n seconds
                except select.error, err:
                    print 'select.error on RemoteAPI.listen(), ignoring...'
                    continue
                DEBUG('2')
                with self.cam_info_lock:
                    cam_ids = self.cam_info.keys()
                for cam_id in cam_ids:
                    with self.cam_info_lock:
                        connected = self.cam_info[cam_id]['caller'].connected
                    if not connected:
                        print 'main_brain WARNING: lost %s at %s'%(cam_id,time.asctime())
                        self.close(cam_id)
            self.thread_done.set()

        # ================================================================
        #
        # Methods called remotely from listeners
        #
        # These all get called in their own thread.  Don't call across
        # the thread boundary without using locks, especially to GUI
        # or OpenGL.
        #
        # ================================================================

        def register_downstream_kalman_host(self,host,port):
            global downstream_kalman_hosts
            addr = (host,port)
            if addr not in downstream_kalman_hosts:
                print 'appending to kalman host list:',addr
                downstream_kalman_hosts.append( (host,port) )
            else:
                print 'already in kalman host list:',addr

        def remove_downstream_kalman_host(self,host,port):
            global downstream_kalman_hosts
            host_tuple = (host,port)
            try:
                i = downstream_kalman_hosts.index( host_tuple )
            except ValueError:
                return # could not find entry
            del downstream_kalman_hosts[i]

        # ================================================================
        #
        # Methods called remotely from cameras
        #
        # These all get called in their own thread.  Don't call across
        # the thread boundary without using locks, especially to GUI
        # or OpenGL.
        #
        # ================================================================

        def register_new_camera(self,cam_no,scalar_control_info,port,force_cam_id=None):
            """register new camera, return cam_id (caller: remote camera)"""

            caller= self.daemon.getLocalStorage().caller # XXX Pyro hack??
            caller_addr= caller.addr
            caller_ip, caller_port = caller_addr
            fqdn = socket.getfqdn(caller_ip)

            if force_cam_id is None:
                cam_id = '%s_%d'%(fqdn,cam_no)
            else:
                cam_id = force_cam_id

            print "REGISTER NEW CAMERA %s from node %s"%(cam_id,fqdn)

            cam2mainbrain_data_port = self.main_brain.coord_processor.connect(cam_id,fqdn)
            with self.cam_info_lock:
                self.cam_info[cam_id] = {'commands':{}, # command queue for cam
                                         'lock':threading.Lock(), # prevent concurrent access
                                         'image':None,  # most recent image from cam
                                         'fps':None,    # most recept fps from cam
                                         'points_distorted':[], # 2D image points
                                         'caller':caller,
                                         'scalar_control_info':scalar_control_info,
                                         'fqdn':fqdn,
                                         'port':port,
                                         'cam2mainbrain_data_port':cam2mainbrain_data_port,
                                         'is_calibrated':False, # has 3D calibration been sent yet?
                                         }
            self.no_cams_connected.clear()
            with self.changed_cam_lock:
                self.new_cam_ids.append(cam_id)
            return cam_id

        def set_image(self,cam_id,coord_and_image):
            """set most recent image (caller: remote camera)"""
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    self.cam_info[cam_id]['image'] = coord_and_image

        def receive_missing_data(self, cam_id, framenumber_offset, missing_data ):
            #print 'received missing data from camera %s (offset %d):'%(cam_id, framenumber_offset)
            if len(missing_data)==0:
                # no missing data
                return

            deferred_2d_data = []
            for (absolute_cam_no, framenumber, remote_timestamp, camn_received_time,
                 points_distorted) in missing_data:

                corrected_framenumber = framenumber-framenumber_offset
                if len(points_distorted)==0:
                    # no point was tracked that frame
                    points_distorted = [(nan,nan,nan,nan,nan,nan,nan,nan,nan,False,0,0,0,0)] # same as no_point_tuple
                for point_tuple in points_distorted:
                    # Save 2D data (even when no point found) to allow
                    # temporal correlation of movie frames to 2D data.
                    try:
                        frame_pt_idx = point_tuple[PT_TUPLE_IDX_FRAME_PT_IDX]
                        cur_val = point_tuple[PT_TUPLE_IDX_CUR_VAL_IDX]
                        mean_val = point_tuple[PT_TUPLE_IDX_MEAN_VAL_IDX]
                        sumsqf_val = point_tuple[PT_TUPLE_IDX_SUMSQF_VAL_IDX]
                    except:
                        print >> sys.stderr, 'error while appending point_tuple',point_tuple
                        raise
                    if corrected_framenumber is None:
                        # don't bother saving if we don't know when it was from
                        continue
                    deferred_2d_data.append((absolute_cam_no, # defer saving to later
                                             corrected_framenumber,
                                             remote_timestamp, camn_received_time)
                                            +point_tuple[:5]
                                            +(frame_pt_idx,cur_val,mean_val,sumsqf_val))
            self.main_brain.queue_data2d.put(deferred_2d_data)

        def set_fps(self,cam_id,fps):
            """set most recent fps (caller: remote camera)"""
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    self.cam_info[cam_id]['fps'] = fps

        def get_and_clear_commands(self,cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam['lock']
                with cam_lock:
                    cmds = cam['commands']
                    cam['commands'] = {}
            return cmds

        def get_cam2mainbrain_port(self,cam_id):
            """Send port number to which camera should send realtime data"""
            cam2mainbrain_data_port = self.main_brain.coord_processor.get_cam2mainbrain_data_port(cam_id)
            return cam2mainbrain_data_port

        def log_message(self,cam_id,host_timestamp,message):
            mainbrain_timestamp = time.time()
            print 'received log message from %s: %s'%(cam_id,message)
            self.message_queue.put( (mainbrain_timestamp,cam_id,host_timestamp,message) )

        def close(self,cam_id):
            """gracefully say goodbye (caller: remote camera)"""
            with self.cam_info_lock:
                self.main_brain.coord_processor.disconnect(cam_id)
                del self.cam_info[cam_id]
                if not len(self.cam_info):
                    self.no_cams_connected.set()
                with self.changed_cam_lock:
                    self.old_cam_ids.append(cam_id)

    ######## end of RemoteAPI class

    # main MainBrain class

    def __init__(self,server=None,save_profiling_data=False, show_sync_errors=True, publish_ros=False):
        global main_brain_keeper, hostname

        self.publish_ros = publish_ros

        if server is not None:
            hostname = server
        print 'running mainbrain at hostname "%s"'%hostname

        self.debug_level = threading.Event()
        self.show_overall_latency = threading.Event()

        self.trigger_device_lock = threading.Lock()
        with self.trigger_device_lock:
            self.trigger_device = motmot.fview_ext_trig.ttrigger.DeviceModel()
            self.trigger_device.frames_per_second = rc_params['frames_per_second']
            self.timestamp_modeler = motmot.fview_ext_trig.live_timestamp_modeler.LiveTimestampModeler()
            self.timestamp_modeler.set_trigger_device( self.trigger_device )

        Pyro.core.initServer(banner=0)

        port = flydra.common_variables.mainbrain_port

        # start Pyro server
        daemon = Pyro.core.Daemon(host=hostname,port=port)
        remote_api = MainBrain.RemoteAPI(); remote_api.post_init(self)
        URI=daemon.connect(remote_api,'main_brain')

        # create (but don't start) listen thread
        self.listen_thread=threading.Thread(target=remote_api.listen,
                                            name='RemoteAPI-Thread',
                                            args=(daemon,))
        #self.listen_thread.setDaemon(True) # don't let this thread keep app alive
        self.remote_api = remote_api

        self._new_camera_functions = []
        self._old_camera_functions = []

        self.last_requested_image = {}
        self.pending_requests = {}
        self.last_set_param_time = {}

        self.outgoing_latency_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.num_cams = 0
        self.MainBrain_cam_ids_copy = [] # keep a copy of all cam_ids connected
        self._fqdns_by_cam_id = {}
        self.set_new_camera_callback(self.IncreaseCamCounter)
        self.set_new_camera_callback(self.SendExpectedFPS)
        self.set_new_camera_callback(self.SendCalibration)
        self.set_old_camera_callback(self.DecreaseCamCounter)

        self.last_saved_data_time = 0.0
        self.last_trigger_framecount_check_time = 0.0

        self._currently_recording_movies = {}

        self.reconstructor = None

        # Attributes which come in use when saving data occurs
        self.h5file = None
        self.h5data2d = None
        self.h5cam_info = None
        self.h5host_clock_info = None
        self.h5trigger_clock_info = None
        self.h5movie_info = None
        self.h5textlog = None
        if DO_KALMAN:
            self.h5data3d_kalman_estimates = None
            self.h5data3d_kalman_observations = None
            self.h5_2d_obs = None
        else:
            self.h5data3d_best = None

        # Queues of information to save
        self.queue_data2d          = Queue.Queue()
        self.queue_cam_info        = Queue.Queue()
        self.queue_host_clock_info = Queue.Queue()
        self.queue_trigger_clock_info = Queue.Queue()
        self.queue_data3d_best     = Queue.Queue()

        self.queue_data3d_kalman_estimates = Queue.Queue()

        self.hypothesis_test_max_error = LockedValue(
            rc_params['hypothesis_test_max_acceptable_error']) # maximum reprojection error

        self.coord_processor = CoordinateProcessor(self,
                                                   save_profiling_data=save_profiling_data,
                                                   debug_level=self.debug_level,
                                                   show_overall_latency=self.show_overall_latency,
                                                   show_sync_errors=show_sync_errors,
                                                   publish_ros=self.publish_ros)
        #self.coord_processor.setDaemon(True)
        self.coord_processor.start()

        self.trig_receiver = TrigReceiver(self)
        self.trig_receiver.setDaemon(True)
        self.trig_receiver.start()

        self.timestamp_echo_receiver = TimestampEchoReceiver(self)
        self.timestamp_echo_receiver.setDaemon(True)
        self.timestamp_echo_receiver.start()

        main_brain_keeper.register( self )

    def get_fps(self):
        return self.trigger_device.frames_per_second_actual

    def set_fps(self,fps):
        self.do_synchronization(new_fps=fps)

    def do_synchronization(self,new_fps=None):
        if self.is_saving_data():
            raise RuntimeError('will not (re)synchronize while saving data')

        if new_fps is not None:
            self.trigger_device.frames_per_second = new_fps
            actual_new_fps = self.trigger_device.frames_per_second_actual

        self.timestamp_modeler.synchronize = True # fire event handler
        if new_fps is not None:
            cam_ids = self.remote_api.external_get_cam_ids()
            for cam_id in cam_ids:
                try:
                    self.send_set_camera_property(
                        cam_id, 'expected_trigger_framerate', actual_new_fps )
                except Exception,err:
                    print 'ERROR:',err
            rc_params['frames_per_second'] = actual_new_fps
            save_rc_params()

    def get_hypothesis_test_max_error(self):
        return self.hypothesis_test_max_error.get()

    def set_hypothesis_test_max_error(self,val):
        self.hypothesis_test_max_error.set(val)
        rc_params['hypothesis_test_max_acceptable_error'] = val
        save_rc_params()

    def IncreaseCamCounter(self,cam_id,scalar_control_info,fqdn_and_port):
        self.num_cams += 1
        self.MainBrain_cam_ids_copy.append( cam_id )

    def SendExpectedFPS(self,cam_id,scalar_control_info,fqdn_and_port):
        self.send_set_camera_property( cam_id, 'expected_trigger_framerate', self.trigger_device.frames_per_second_actual )

    def SendCalibration(self,cam_id,scalar_control_info,fqdn_and_port):
        if self.reconstructor is not None and cam_id in self.reconstructor.get_cam_ids():
            pmat = self.reconstructor.get_pmat(cam_id)
            intlin = self.reconstructor.get_intrinsic_linear(cam_id)
            intnonlin = self.reconstructor.get_intrinsic_nonlinear(cam_id)
            scale_factor = self.reconstructor.get_scale_factor()
            self.remote_api.external_set_cal( cam_id, pmat, intlin, intnonlin, scale_factor)

    def DecreaseCamCounter(self,cam_id):
        try:
            idx = self.MainBrain_cam_ids_copy.index( cam_id )
        except ValueError, err:
            print 'IGNORING ERROR: DecreaseCamCounter() called with non-existant cam_id'
            return
        self.num_cams -= 1
        del self.MainBrain_cam_ids_copy[idx]

    def get_num_cams(self):
        return self.num_cams

    def get_scalarcontrolinfo(self, cam_id):
        sci, fqdn, port = self.remote_api.external_get_info(cam_id)
        return sci

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

    def service_pending(self):
        """the MainBrain application calls this fairly frequently (e.g. every 100 msec)"""
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

        now = time.time()
        diff = now - self.last_saved_data_time
        if diff >= 5.0: # request missing data and save data every 5 seconds
            self._request_missing_data()
            self._service_save_data()
            self.last_saved_data_time = now

        diff = now - self.last_trigger_framecount_check_time
        if diff >= 5.0:
            self._trigger_framecount_check()
            self.last_trigger_framecount_check_time = now

        self._check_latencies()

    def _trigger_framecount_check(self):
        try:
            tmp = self.timestamp_modeler.update(return_last_measurement_info=True)
            start_timestamp, stop_timestamp, framecount, tcnt = tmp
            self.queue_trigger_clock_info.put((start_timestamp, framecount, tcnt, stop_timestamp))
        except motmot.fview_ext_trig.live_timestamp_modeler.ImpreciseMeasurementError, err:
            pass

    def _check_latencies(self):
        timestamp_echo_fmt1 = flydra.common_variables.timestamp_echo_fmt1
        timestamp_echo_listener_port = flydra.common_variables.timestamp_echo_listener_port

        for cam_id in self.MainBrain_cam_ids_copy:
            if cam_id not in self._fqdns_by_cam_id:
                sci, fqdn, cam2mainbrain_port = self.remote_api.external_get_info(cam_id)
                self._fqdns_by_cam_id[cam_id] = fqdn
            else:
                fqdn = self._fqdns_by_cam_id[cam_id]
            buf = struct.pack( timestamp_echo_fmt1, time.time() )
            self.outgoing_latency_UDP_socket.sendto(buf,(fqdn,timestamp_echo_listener_port))

    def get_last_image_fps(self, cam_id):
        # XXX should extend to include lines

        # Points are originally distorted (and align with distorted
        # image).
        (image, fps, points_distorted,
         image_coords) = self.remote_api.external_get_image_fps_points(cam_id)

        return image, fps, points_distorted, image_coords

    def close_camera(self,cam_id):
        sys.stdout.flush()
        self.remote_api.external_quit( cam_id )
        sys.stdout.flush()

    def set_collecting_background(self, cam_id, value):
        self.remote_api.external_send_set_camera_property( cam_id, 'collecting_background', value)

    def set_color_filter(self, cam_id, value):
        self.remote_api.external_send_set_camera_property( cam_id, 'color_filter', value)

    def take_background(self,cam_id):
        self.remote_api.external_take_background(cam_id)

    def clear_background(self,cam_id):
        self.remote_api.external_clear_background(cam_id)

    def send_set_camera_property(self, cam_id, property_name, value):
        self.remote_api.external_send_set_camera_property( cam_id, property_name, value)

    def request_image_async(self, cam_id):
        self.remote_api.external_request_image_async(cam_id)

    def get_debug_level(self):
        return self.debug_level.isSet()

    def set_debug_level(self,value):
        if value:
            self.debug_level.set()
        else:
            self.debug_level.clear()

    def get_show_overall_latency(self):
        return self.show_overall_latency.isSet()

    def set_show_overall_latency(self,value):
        if value:
            self.show_overall_latency.set()
        else:
            self.show_overall_latency.clear()

    def start_recording(self, cam_id, raw_file_basename):
        global XXX_framenumber

        self.remote_api.external_start_recording( cam_id, raw_file_basename)
        approx_start_frame = XXX_framenumber
        self._currently_recording_movies[ cam_id ] = (raw_file_basename, approx_start_frame)
        if self.is_saving_data():
            self.h5movie_info.row['cam_id'] = cam_id
            self.h5movie_info.row['filename'] = raw_file_basename+'.fmf'
            self.h5movie_info.row['approx_start_frame'] = approx_start_frame
            self.h5movie_info.row.append()
            self.h5movie_info.flush()

    def stop_recording(self, cam_id):
        global XXX_framenumber
        self.remote_api.external_stop_recording(cam_id)
        approx_stop_frame = XXX_framenumber
        raw_file_basename, approx_start_frame = self._currently_recording_movies[ cam_id ]
        del self._currently_recording_movies[ cam_id ]
        # modify save file to include approximate movie stop time
        if self.is_saving_data():
            nrow = None
            for r in self.h5movie_info:
                # get row in table
                if (r['cam_id'] == cam_id and r['filename'] == raw_file_basename+'.fmf' and
                    r['approx_start_frame']==approx_start_frame):
                    nrow =r.nrow
                    break
            if nrow is not None:
                nrowi = int(nrow) # pytables bug workaround...
                assert nrowi == nrow # pytables bug workaround...
                approx_stop_framei = int(approx_stop_frame)
                assert approx_stop_framei == approx_stop_frame

                new_columns = numpy.rec.fromarrays([[approx_stop_framei]], formats='i8')
                self.h5movie_info.modifyColumns(start=nrowi, columns=new_columns, names=['approx_stop_frame'])
            else:
                raise RuntimeError("could not find row to save movie stop frame.")

    def start_small_recording(self, cam_id, small_filename):
        self.remote_api.external_start_small_recording( cam_id,
                                                        small_filename)

    def stop_small_recording(self, cam_id):
        self.remote_api.external_stop_small_recording(cam_id)

    def quit(self):
        """closes any files being saved and closes camera connections"""
        # XXX ====== non-isolated calls to remote_api being done ======
        # this may be called twice: once explicitly and once by __del__
        with self.remote_api.cam_info_lock:
            cam_ids = self.remote_api.cam_info.keys()

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

        self.stop_saving_data()
        self.coord_processor.quit()

    def load_calibration(self,dirname):
        if self.is_saving_data():
            raise RuntimeError("Cannot (re)load calibration while saving data")
        connected_cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = flydra.reconstruct.Reconstructor(dirname)
        calib_cam_ids = self.reconstructor.get_cam_ids()

        calib_cam_ids = calib_cam_ids

        self.coord_processor.set_reconstructor(self.reconstructor)

        for cam_id in calib_cam_ids:
            pmat = self.reconstructor.get_pmat(cam_id)
            intlin = self.reconstructor.get_intrinsic_linear(cam_id)
            intnonlin = self.reconstructor.get_intrinsic_nonlinear(cam_id)
            scale_factor = self.reconstructor.get_scale_factor()
            if cam_id in connected_cam_ids:
                self.remote_api.external_set_cal( cam_id, pmat, intlin, intnonlin, scale_factor )

    def clear_calibration(self):
        if self.is_saving_data():
            raise RuntimeError("Cannot unload calibration while saving data")
        cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = None

        self.coord_processor.set_reconstructor(self.reconstructor)

        for cam_id in cam_ids:
            self.remote_api.external_set_cal( cam_id, None, None, None, None )

    def set_new_tracker(self,kalman_model_name=None):
        if self.is_saving_data():
            raise RuntimeError('will not set Kalman parameters while saving data')

        fps = self.get_fps()
        dt = 1.0/fps
        dynamic_model = flydra.kalman.dynamic_models.get_kalman_model(name=kalman_model_name,dt=dt)

        self.kalman_saver_info_instance = flydra_kalman_utils.KalmanSaveInfo(name=kalman_model_name)
        self.KalmanEstimatesDescription = self.kalman_saver_info_instance.get_description()
        self.dynamic_model=dynamic_model
        self.dynamic_model_name=kalman_model_name

        self.h5_xhat_names = PT.Description(self.KalmanEstimatesDescription().columns)._v_names

        # send params over to realtime coords thread
        self.coord_processor.set_new_tracker(kalman_model=dynamic_model)

    def __del__(self):
        self.quit()

    def is_saving_data(self):
        return self.h5file is not None

    def start_saving_data(self, filename):
        if os.path.exists(filename):
            raise RuntimeError("will not overwrite data file")

        self.timestamp_modeler.block_activity = True
        self.h5file = PT.openFile(filename, mode="w", title="Flydra data file")
        expected_rows = int(1e6)
        ct = self.h5file.createTable # shorthand
        root = self.h5file.root # shorthand
        self.h5data2d   = ct(root,'data2d_distorted', Info2D, "2d data",
                             expectedrows=expected_rows*5)
        self.h5cam_info = ct(root,'cam_info', CamSyncInfo, "Cam Sync Info",
                             expectedrows=500)
        self.h5host_clock_info = ct(root,'host_clock_info', HostClockInfo, "Host Clock Info",
                                    expectedrows=6*60*24) # 24 hours at 10 sec sample intervals
        self.h5trigger_clock_info = ct(root,'trigger_clock_info', TriggerClockInfo, "Trigger Clock Info",
                                       expectedrows=6*60*24) # 24 hours at 10 sec sample intervals
        self.h5movie_info = ct(root,'movie_info', MovieInfo, "Movie Info",
                               expectedrows=500)
        self.h5textlog = ct(root,'textlog', TextLogDescription,
                            "text log")
        self._startup_message()
        if self.reconstructor is not None:
            self.reconstructor.save_to_h5file(self.h5file)
            if DO_KALMAN:
                self.h5data3d_kalman_estimates = ct(root,'kalman_estimates', self.KalmanEstimatesDescription,
                                                    "3d data (from Kalman filter)",
                                                    expectedrows=expected_rows)
                self.h5data3d_kalman_estimates.attrs.dynamic_model_name = self.dynamic_model_name
                self.h5data3d_kalman_estimates.attrs.dynamic_model = self.dynamic_model

                self.h5data3d_kalman_observations = ct(root,'kalman_observations', FilteredObservations,
                                                       "3d data (input to Kalman filter)",
                                                       expectedrows=expected_rows)
                self.h5_2d_obs = self.h5file.createVLArray(self.h5file.root,
                                                           'kalman_observations_2d_idxs',
                                                           kalman_observations_2d_idxs_type(), # dtype should match with tro.observations_2d
                                                           "camns and idxs")
                self.h5_2d_obs_next_idx = 0
            else:
                self.h5data3d_best = ct(root,'data3d_best', Info3D,
                                        "3d data (best)",
                                        expectedrows=expected_rows)

        general_save_info=self.coord_processor.get_general_cam_info()
        for cam_id,dd in general_save_info.iteritems():
            self.h5cam_info.row['cam_id'] = cam_id
            self.h5cam_info.row['camn']   = dd['absolute_cam_no']
            self.h5cam_info.row['frame0'] = dd['frame0']
            with self.remote_api.cam_info_lock:
                self.h5cam_info.row['hostname'] = self.remote_api.cam_info[cam_id]['fqdn']
            self.h5cam_info.row.append()
        self.h5cam_info.flush()

        # save raw image from each camera
        img = self.h5file.createGroup(root,'images','sample images')
        cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            image, fps, points_distorted, image_coords = self.get_last_image_fps(cam_id)
            if image is None:
                raise ValueError('image cannot be None')
            self.h5file.createArray( img, cam_id, image, 'sample image from %s'%cam_id )

    def stop_saving_data(self):
        self._service_save_data()
        if self.is_saving_data():
            self.h5file.close()
            self.h5file = None
            self.timestamp_modeler.block_activity = False
        else:
            DEBUG('saving already stopped, cannot stop again')
        self.h5data2d = None
        self.h5cam_info = None
        self.h5host_clock_info = None
        self.h5trigger_clock_info = None
        self.h5movie_info = None
        self.h5textlog = None
        if DO_KALMAN:
            self.h5data3d_kalman_estimates = None
            self.h5data3d_kalman_observations = None
            self.h5_2d_obs = None
        else:
            self.h5data3d_best = None

    def _startup_message(self):
        textlog_row = self.h5textlog.row
        cam_id = 'mainbrain'
        timestamp = time.time()

        # This line is important (including the formatting). It is
        # read by flydra.a2.check_atmel_clock.

        list_of_textlog_data = [
            (timestamp,cam_id,timestamp,
             ('MainBrain running at %s fps, (top %s, '
              'hypothesis_test_max_error %s, trigger_CS3 %s, FOSC %s, flydra_version %s, '
              'time_tzname0 %s)'%(
            str(self.trigger_device.frames_per_second_actual),
            str(self.trigger_device._t3_state.timer3_top),
            str(self.get_hypothesis_test_max_error()),
            str(self.trigger_device._t3_state.timer3_CS),
            str(self.trigger_device.FOSC),
            flydra.version.__version__,
            time.tzname[0],
            ))),
            (timestamp,cam_id,timestamp, 'using flydra version %s'%(
             flydra.version.__version__,)),
            ]
        for textlog_data in list_of_textlog_data:
            (mainbrain_timestamp,cam_id,host_timestamp,message) = textlog_data
            textlog_row['mainbrain_timestamp'] = mainbrain_timestamp
            textlog_row['cam_id'] = cam_id
            textlog_row['host_timestamp'] = host_timestamp
            textlog_row['message'] = message
            textlog_row.append()

        self.h5textlog.flush()

    def _request_missing_data(self):
        if ATTEMPT_DATA_RECOVERY:
            # request from camera computers any data that we're missing
            missing_data_dict = self.coord_processor.get_missing_data_dict()
            for camn, (cam_id, framenumber_offset, list_of_missing_framenumbers) in missing_data_dict.iteritems():
                #print 'requesting from camn %d: %d frames %s'%(camn,len(list_of_missing_framenumbers), numpy.array(list_of_missing_framenumbers) )
                self.remote_api.external_request_missing_data(cam_id,camn,framenumber_offset,list_of_missing_framenumbers)

    def _service_save_data(self):
        # ** 2d data **
        #   clear queue
        list_of_rows_of_data2d = []
        try:
            while True:
                tmp = self.queue_data2d.get(0)
                list_of_rows_of_data2d.extend( tmp )
        except Queue.Empty:
            pass
        #   save
        if self.h5data2d is not None and len(list_of_rows_of_data2d):
            # it's much faster to convert to numpy first:
            recarray = numpy.rec.array(
                list_of_rows_of_data2d,
                dtype=Info2DCol_description)
            self.h5data2d.append( recarray )
            self.h5data2d.flush()

        # ** textlog **
        # clear queue
        list_of_textlog_data = []
        try:
            while True:
                tmp = self.remote_api.message_queue.get(0)
                list_of_textlog_data.append( tmp )
        except Queue.Empty:
            pass
        if 1:
            for textlog_data in list_of_textlog_data:
                (mainbrain_timestamp,cam_id,host_timestamp,message) = textlog_data
                print 'MESSAGE: %s %s "%s"'%(cam_id, time.asctime(time.localtime(host_timestamp)), message)
        #   save
        if self.h5textlog is not None and len(list_of_textlog_data):
            textlog_row = self.h5textlog.row
            for textlog_data in list_of_textlog_data:
                (mainbrain_timestamp,cam_id,host_timestamp,message) = textlog_data
                textlog_row['mainbrain_timestamp'] = mainbrain_timestamp
                textlog_row['cam_id'] = cam_id
                textlog_row['host_timestamp'] = host_timestamp
                textlog_row['message'] = message
                textlog_row.append()

            self.h5textlog.flush()

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
                cam_id, absolute_cam_no, frame0, camhost = cam_info
                cam_info_row['cam_id'] = cam_id
                cam_info_row['camn']   = absolute_cam_no
                cam_info_row['frame0'] = frame0
                cam_info_row['hostname'] = camhost
                cam_info_row.append()

            self.h5cam_info.flush()

        if DO_KALMAN:
            # ** 3d data - kalman **
            q = self.queue_data3d_kalman_estimates

            #   clear queue
            list_of_3d_data = []
            try:
                while True:
                    list_of_3d_data.append( q.get(0) )
            except Queue.Empty:
                pass
            if self.h5data3d_kalman_estimates is not None:
##                print 'saving kalman data (%d objects)'%(
##                    len(list_of_3d_data),)
                for (obj_id, tro_frames, tro_xhats, tro_Ps, tro_timestamps,
                     obs_frames, obs_data,
                     observations_2d, obs_Lcoords) in list_of_3d_data:


                    if len(obs_frames)<MIN_KALMAN_OBSERVATIONS_TO_SAVE:
                        # only save data with at least 10 observations
                        if self.debug_level.isSet():
                            print 'not saving kalman object -- too few observations to save'
                        continue

                    if self.debug_level.isSet():
                        print 'saving kalman object %d'%(obj_id,)

                    # save observation 2d data indexes
                    this_idxs = []
                    for camns_and_idxs in observations_2d:
                        this_idxs.append( self.h5_2d_obs_next_idx )
                        self.h5_2d_obs.append( camns_and_idxs )
                        self.h5_2d_obs_next_idx += 1
                    self.h5_2d_obs.flush()

                    this_idxs = numpy.array( this_idxs, dtype=numpy.uint64 ) # becomes obs_2d_idx (index into 'kalman_observations_2d_idxs')

                    # save observations
                    observations_frames = numpy.array(obs_frames, dtype=numpy.uint64)
                    obj_id_array = numpy.empty(observations_frames.shape, dtype=numpy.uint32)
                    obj_id_array.fill(obj_id)
                    observations_data = numpy.array(obs_data, dtype=numpy.float32)
                    observations_Lcoords = numpy.array(obs_Lcoords, dtype=numpy.float32)
                    list_of_obs = [observations_data[:,i] for i in range(observations_data.shape[1])]
                    list_of_lines = [observations_Lcoords[:,i] for i in range(observations_Lcoords.shape[1])]
                    array_list = [obj_id_array,observations_frames]+list_of_obs+[this_idxs]+list_of_lines
                    obs_recarray = numpy.rec.fromarrays(array_list, names = h5_obs_names)

                    self.h5data3d_kalman_observations.append(obs_recarray)
                    self.h5data3d_kalman_observations.flush()

                    # save xhat info (kalman estimates)
                    frames = numpy.array(tro_frames, dtype=numpy.uint64)
                    timestamps = numpy.array(tro_timestamps, dtype=numpy.float64)
                    xhat_data = numpy.array(tro_xhats, dtype=numpy.float32)
                    P_data_full = numpy.array(tro_Ps, dtype=numpy.float32)
                    obj_id_array = numpy.empty(frames.shape, dtype=numpy.uint32)
                    obj_id_array.fill(obj_id)
                    list_of_xhats = [xhat_data[:,i] for i in range(xhat_data.shape[1])]
                    ksii = self.kalman_saver_info_instance
                    list_of_Ps = ksii.covar_mats_to_covar_entries(P_data_full)
                    xhats_recarray = numpy.rec.fromarrays(
                        [obj_id_array,frames,timestamps]+list_of_xhats+list_of_Ps,
                        names = self.h5_xhat_names)

                    self.h5data3d_kalman_estimates.append( xhats_recarray )
                    self.h5data3d_kalman_estimates.flush()

        else:
            # ** 3d data - hypothesis testing **
            q = self.queue_data3d_best
            h5table = self.h5data3d_best

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

        # ** camera info **
        #   clear queue
        list_of_host_clock_info = []
        try:
            while True:
                list_of_host_clock_info.append( self.queue_host_clock_info.get(0) )
        except Queue.Empty:
            pass
        #   save
        if self.h5host_clock_info is not None:
            host_clock_info_row = self.h5host_clock_info.row
            for host_clock_info in list_of_host_clock_info:
                remote_hostname, start_timestamp, remote_timestamp, stop_timestamp = host_clock_info
                host_clock_info_row['remote_hostname'] = remote_hostname
                host_clock_info_row['start_timestamp'] = start_timestamp
                host_clock_info_row['remote_timestamp'] = remote_timestamp
                host_clock_info_row['stop_timestamp'] = stop_timestamp
                host_clock_info_row.append()

            self.h5host_clock_info.flush()

        #   clear queue
        list_of_trigger_clock_info = []
        try:
            while True:
                list_of_trigger_clock_info.append( self.queue_trigger_clock_info.get(0) )
        except Queue.Empty:
            pass
        #   save
        if self.h5trigger_clock_info is not None:
            row = self.h5trigger_clock_info.row
            for trigger_clock_info in list_of_trigger_clock_info:
                start_timestamp, framecount, tcnt, stop_timestamp = trigger_clock_info
                row['start_timestamp'] = start_timestamp
                row['framecount'] = framecount
                row['tcnt'] = tcnt
                row['stop_timestamp'] = stop_timestamp
                row.append()

            self.h5trigger_clock_info.flush()
