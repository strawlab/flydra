"""core runtime code for online, realtime tracking"""
from __future__ import with_statement, division
import threading, time, socket, sys, os, copy, struct
import warnings
import json
import collections

import tzlocal
import flydra_core.reconstruct
import numpy
import numpy as np
from numpy import nan
import Queue
from distutils.version import LooseVersion

pytables_filt = numpy.asarray
import atexit

import flydra_core.version
import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils
import flydra_core.kalman.flydra_tracker

import flydra_core.data_descriptions
from flydra_core.coordinate_receiver import CoordinateProcessor, ATTEMPT_DATA_RECOVERY

# ensure that pytables uses numpy:
import tables

# bug was fixed in pytables 1.3.1 where HDF5 file kept in inconsistent state
assert LooseVersion(tables.__version__) >= LooseVersion("1.3.1")

import tables.flavor

tables.flavor.restrict_flavors(keep=["numpy"])
warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

import roslib

roslib.load_manifest("rospy")
roslib.load_manifest("std_srvs")
roslib.load_manifest("ros_flydra")
roslib.load_manifest("triggerbox")
import rospy
import std_srvs.srv
import std_msgs.msg
from ros_flydra.msg import FlydraError, CameraList
import ros_flydra.srv
import ros_flydra.cv2_bridge

from triggerbox.triggerbox_client import TriggerboxClient
from triggerbox.triggerbox_host import TriggerboxHost

import flydra_core.rosutils

LOG = flydra_core.rosutils.Log(to_ros=True)

MIN_KALMAN_OBSERVATIONS_TO_SAVE = (
    0  # how many data points are required before saving trajectory?
)

import flydra_core.common_variables
import flydra_core.flydra_socket as flydra_socket

WIRE_ORDER_CUR_VAL_IDX = flydra_core.data_descriptions.WIRE_ORDER_CUR_VAL_IDX
WIRE_ORDER_MEAN_VAL_IDX = flydra_core.data_descriptions.WIRE_ORDER_MEAN_VAL_IDX
WIRE_ORDER_SUMSQF_VAL_IDX = flydra_core.data_descriptions.WIRE_ORDER_SUMSQF_VAL_IDX


class MainBrainKeeper:
    def __init__(self):
        self.kept = []
        atexit.register(self.atexit)

    def register(self, mainbrain_instance):
        self.kept.append(mainbrain_instance)

    def atexit(self):
        for k in self.kept:
            k.quit()  # closes hdf5 file and closes cameras


main_brain_keeper = MainBrainKeeper()  # global to close MainBrain instances upon exit


class LockedValue:
    def __init__(self, initial_value=None):
        self.lock = threading.Lock()
        self._val = initial_value
        self._q = Queue.Queue()

    def set(self, value):
        self._q.put(value)

    def get(self):
        try:
            while 1:
                self._val = self._q.get_nowait()
        except Queue.Empty:
            pass
        return self._val


# 2D data format for PyTables:
Info2D = flydra_core.data_descriptions.Info2D
TextLogDescription = flydra_core.data_descriptions.TextLogDescription
CamSyncInfo = flydra_core.data_descriptions.CamSyncInfo
HostClockInfo = flydra_core.data_descriptions.HostClockInfo
TriggerClockInfo = flydra_core.data_descriptions.TriggerClockInfo
MovieInfo = flydra_core.data_descriptions.MovieInfo
ExperimentInfo = flydra_core.data_descriptions.ExperimentInfo

FilteredObservations = flydra_kalman_utils.FilteredObservations
ML_estimates_2d_idxs_type = flydra_kalman_utils.ML_estimates_2d_idxs_type

h5_obs_names = tables.Description(FilteredObservations().columns)._v_names

# allow rapid building of numpy.rec.array:
Info2DCol_description = tables.Description(Info2D().columns)._v_nested_descr


def save_ascii_matrix(filename, m):
    fd = open(filename, mode="wb")
    for row in m:
        fd.write(" ".join(map(str, row)))
        fd.write("\n")


class TimestampEchoReceiver(threading.Thread):
    def __init__(self, main_brain):
        self.main_brain = main_brain
        threading.Thread.__init__(self, name="TimestampEchoReceiver thread")

    def run(self):
        ip2hostname = {}

        timestamp_echo_fmt2 = flydra_core.common_variables.timestamp_echo_fmt2

        port = flydra_core.common_variables.timestamp_echo_gatherer_port  # my port
        addrinfo = flydra_socket.make_addrinfo(
            host=flydra_socket.get_bind_address(), port=port
        )
        timestamp_echo_gatherer = flydra_socket.FlydraTransportReceiver(addrinfo)
        addrinfo = timestamp_echo_gatherer.get_listen_addrinfo()
        LOG.info("MainBrain TimestampEchoReceiver binding %s" % (addrinfo.to_dict(),))

        last_clock_diff_measurements = collections.defaultdict(list)

        while 1:
            try:
                timestamp_echo_buf, sender_sockaddr = timestamp_echo_gatherer.recv(
                    return_sender_sockaddr=True
                )
            except Exception as err:
                LOG.warn("unknown Exception receiving timestamp echo data: %s" % err)
                continue
            except:
                LOG.warn("unknown error (non-Exception!) receiving timestamp echo data")
                continue
            (timestamp_echo_remote_ip, cam_port) = sender_sockaddr

            stop_timestamp = time.time()

            start_timestamp, remote_timestamp = struct.unpack(
                timestamp_echo_fmt2, timestamp_echo_buf
            )

            tlist = last_clock_diff_measurements[timestamp_echo_remote_ip]
            tlist.append((start_timestamp, remote_timestamp, stop_timestamp))
            if len(tlist) == 100:
                if timestamp_echo_remote_ip not in ip2hostname:
                    ip2hostname[timestamp_echo_remote_ip] = socket.getfqdn(
                        timestamp_echo_remote_ip
                    )
                remote_hostname = ip2hostname[timestamp_echo_remote_ip]
                tarray = numpy.array(tlist)

                del tlist[:]  # clear list
                start_timestamps = tarray[:, 0]
                stop_timestamps = tarray[:, 2]
                roundtrip_duration = stop_timestamps - start_timestamps
                # find best measurement (that with shortest roundtrip_duration)
                rowidx = numpy.argmin(roundtrip_duration)
                srs = tarray[rowidx, :]
                start_timestamp, remote_timestamp, stop_timestamp = srs
                clock_diff_msec = abs(remote_timestamp - start_timestamp) * 1e3
                if clock_diff_msec > 1:
                    self.main_brain.error_ros_msgs_pub.publish(
                        FlydraError(
                            FlydraError.CLOCK_DIFF,
                            "%s/%f" % (remote_hostname, clock_diff_msec),
                        )
                    )
                    LOG.warn(
                        "%s : clock diff: %.3f msec(measurement err: %.3f msec)"
                        % (
                            remote_hostname,
                            clock_diff_msec,
                            roundtrip_duration[rowidx] * 1e3,
                        )
                    )

                self.main_brain.queue_host_clock_info.put(
                    (remote_hostname, start_timestamp, remote_timestamp, stop_timestamp)
                )
                if 0:
                    measurement_duration = roundtrip_duration[rowidx]
                    clock_diff = stop_timestamp - remote_timestamp

                    LOG.debug(
                        "%s: the remote diff is %.1f msec (within 0-%.1f msec accuracy)"
                        % (
                            remote_hostname,
                            clock_diff * 1000,
                            measurement_duration * 1000,
                        )
                    )


class MainBrain(object):
    """Handle all camera network stuff and interact with application"""

    # See commits explaining socket starvation on why these are not all enabled
    ROS_CONTROL_API = dict(
        start_collecting_background=(std_srvs.srv.Empty),
        stop_collecting_background=(std_srvs.srv.Empty),
        take_background=(std_srvs.srv.Empty),
        #        clear_background=(std_srvs.srv.Empty),
        start_saving_data=(std_srvs.srv.Empty),
        stop_saving_data=(std_srvs.srv.Empty),
        start_recording=(std_srvs.srv.Empty),
        stop_recording=(std_srvs.srv.Empty),
        start_small_recording=(std_srvs.srv.Empty),
        stop_small_recording=(std_srvs.srv.Empty),
        do_synchronization=(std_srvs.srv.Empty),
        log_message=(ros_flydra.srv.MainBrainLogMessage),
        get_version=(ros_flydra.srv.MainBrainGetVersion),
        register_new_camera=(ros_flydra.srv.MainBrainRegisterNewCamera),
        get_listen_address=(ros_flydra.srv.MainBrainGetListenAddress),
        get_and_clear_commands=(ros_flydra.srv.MainBrainGetAndClearCommands),
        set_image=(ros_flydra.srv.MainBrainSetImage),
        receive_missing_data=(ros_flydra.srv.MainBrainReceiveMissingData),
        close_camera=(ros_flydra.srv.MainBrainCloseCamera),
    )

    ROS_CONFIGURATION = dict(
        frames_per_second=100.0,
        triggerbox_namespace="/trig1",
        triggerbox_hardware_device="",
        kalman_model="EKF mamarama, units: mm",
        max_reconstruction_latency_sec=0.06,  # 60 msec
        max_N_hypothesis_test=3,
        save_data_dir="~/FLYDRA",
        save_movie_dir="~/FLYDRA_MOVIES",
        camera_calibration="",
        use_unix_domain_sockets=False,
        posix_scheduler="",  # '' means OS default, set to e.g. ['FIFO', 99] for max
    )

    class RemoteAPI:

        # ================================================================
        #
        # Methods called locally
        #
        # ================================================================

        def post_init(self, main_brain):
            """call after __init__"""
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
                cam_lock = cam["lock"]
                with cam_lock:
                    scalar_control_info = copy.deepcopy(cam["scalar_control_info"])
                    fqdn = cam["fqdn"]
                    camnode_ros_name = cam["camnode_ros_name"]
            return scalar_control_info, fqdn, camnode_ros_name

        def external_get_image_fps_points(self, cam_id):
            ### XXX should extend to include lines
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    coord_and_image = cam["image"]
                    points_distorted = cam["points_distorted"][:]
            # NB: points are distorted (and therefore align
            # with distorted image)
            if coord_and_image is not None:
                image_coords, image = coord_and_image
            else:
                image_coords, image = None, None
            fps = np.nan
            return image, fps, points_distorted, image_coords

        def external_send_set_camera_property(self, cam_id, property_name, value):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"].setdefault("set", {})[property_name] = value
                    old_value = cam["scalar_control_info"][property_name]
                    if type(old_value) == tuple and type(value) == int:
                        # brightness, gain, shutter
                        cam["scalar_control_info"][property_name] = (
                            value,
                            old_value[1],
                            old_value[2],
                        )
                    else:
                        cam["scalar_control_info"][property_name] = value

        def external_request_image_async(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["get_im"] = None

        def external_start_recording(self, cam_id, raw_file_basename):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["start_recording"] = raw_file_basename

        def external_stop_recording(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["stop_recording"] = None

        def external_start_small_recording(self, cam_id, small_filebasename):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["start_small_recording"] = small_filebasename

        def external_stop_small_recording(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["stop_small_recording"] = None

        def external_quit(self, cam_id):
            with self.cam_info_lock:
                if cam_id in self.cam_info:
                    cam = self.cam_info[cam_id]
                    cam_lock = cam["lock"]
                    with cam_lock:
                        cam["commands"]["quit"] = True

        def external_take_background(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["take_bg"] = None

        def external_request_missing_data(
            self, cam_id, camn, framenumber_offset, list_of_missing_framenumbers
        ):
            with self.cam_info_lock:
                if cam_id not in self.cam_info:
                    # the camera was dropped, ignore this request
                    return
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]

                camn_and_list = [camn, framenumber_offset]
                camn_and_list.extend(list_of_missing_framenumbers)
                cmd_str = " ".join(map(repr, camn_and_list))
                with cam_lock:
                    cam["commands"]["request_missing"] = cmd_str
            LOG.info(
                "requested missing data from %s. offset %d, frames %s"
                % (cam_id, framenumber_offset, list_of_missing_framenumbers)
            )

        def external_clear_background(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cam["commands"]["clear_bg"] = None

        # ================================================================
        #
        # Methods called remotely from cameras
        #
        # These all get called in their own thread.  Don't call across
        # the thread boundary without using locks, especially to GUI
        # or OpenGL.
        #
        # ================================================================

        def register_new_cam(
            self, cam_guid, scalar_control_info, camnode_ros_name, cam_hostname
        ):
            """register new camera (caller: remote camera)"""

            assert camnode_ros_name is not None
            fqdn = cam_hostname

            do_close = False
            with self.cam_info_lock:
                if cam_guid in self.cam_info:
                    do_close = True

            if do_close:
                LOG.warn("camera %s already exists, clearing existing data" % cam_guid)
                self.close(cam_guid)
                self.main_brain.service_pending()

            LOG.info(
                "REGISTER NEW CAMERA %s on %s @ ros node %s"
                % (cam_guid, fqdn, camnode_ros_name)
            )
            self.main_brain.coord_processor.connect(cam_guid)

            with self.cam_info_lock:
                self.cam_info[cam_guid] = {
                    "commands": {},  # command queue for cam
                    "lock": threading.Lock(),  # prevent concurrent access
                    "image": None,  # most recent image from cam
                    "points_distorted": [],  # 2D image points
                    "scalar_control_info": scalar_control_info,
                    "fqdn": fqdn,
                    "camnode_ros_name": camnode_ros_name,
                }
            self.no_cams_connected.clear()
            with self.changed_cam_lock:
                self.new_cam_ids.append(cam_guid)

        def get_listen_addr(self):
            return self.main_brain.coord_processor.get_listen_address()

        def set_image(self, cam_id, coord_and_image):
            """set most recent image (caller: remote camera)"""
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    self.cam_info[cam_id]["image"] = coord_and_image

        def receive_missing_data(self, cam_id, framenumber_offset, missing_data):
            rospy.loginfo(
                "received requested stale data for frame %d" % framenumber_offset
            )

            if len(missing_data) == 0:
                # no missing data
                return

            deferred_2d_data = []
            for (
                absolute_cam_no,
                framenumber,
                remote_timestamp,
                camn_received_time,
                points_distorted,
            ) in missing_data:

                corrected_framenumber = framenumber - framenumber_offset
                if len(points_distorted) == 0:
                    # No point was tracked that frame, send nan values.
                    points_distorted = [(nan, nan, nan, nan, nan, False, 0, 0, 0)]
                for frame_pt_idx, point_tuple in enumerate(points_distorted):
                    # Save 2D data (even when no point found) to allow
                    # temporal correlation of movie frames to 2D data.
                    try:
                        cur_val = point_tuple[WIRE_ORDER_CUR_VAL_IDX]
                        mean_val = point_tuple[WIRE_ORDER_MEAN_VAL_IDX]
                        sumsqf_val = point_tuple[WIRE_ORDER_SUMSQF_VAL_IDX]
                    except:
                        LOG.warn("error while appending point_tuple %r" % point_tuple)
                        raise
                    if corrected_framenumber is None:
                        # don't bother saving if we don't know when it was from
                        continue

                    point_tuple5 = tuple(point_tuple[:5])
                    deferred_2d_data.append(
                        (
                            absolute_cam_no,  # defer saving to later
                            corrected_framenumber,
                            remote_timestamp,
                            camn_received_time,
                        )
                        + point_tuple5
                        + (frame_pt_idx, cur_val, mean_val, sumsqf_val)
                    )
            self.main_brain.queue_data2d.put(deferred_2d_data)

        def get_and_clear_commands(self, cam_id):
            with self.cam_info_lock:
                cam = self.cam_info[cam_id]
                cam_lock = cam["lock"]
                with cam_lock:
                    cmds = cam["commands"]
                    cam["commands"] = {}
            return cmds

        def log_message(self, cam_id, timestamp, message):
            mainbrain_timestamp = time.time()
            LOG.info("received log message from %s: %s" % (cam_id, message))
            self.message_queue.put((mainbrain_timestamp, cam_id, timestamp, message))

        def close(self, cam_id):
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

    def __init__(self, server=None, save_profiling_data=False, show_sync_errors=True):
        global main_brain_keeper

        if server is not None:
            LOG.warn("deprecated 'server' argument given.")

        LOG.info('ros node name "%s"' % rospy.get_name())

        self.load_config()

        self.debug_level = threading.Event()
        self.show_overall_latency = threading.Event()

        self._is_synchronizing = False

        # we support in or out of process trigger boxes
        if self.config["triggerbox_hardware_device"]:
            # in process
            self.trigger_device = TriggerboxHost(
                device=self.config["triggerbox_hardware_device"],
                ros_topic_base=self.config["triggerbox_namespace"],
            )
        else:
            # out of process
            self.trigger_device = TriggerboxClient(
                host_node=self.config["triggerbox_namespace"]
            )

        self.trigger_device.clock_measurement_callback = (
            self._on_trigger_clock_measurement
        )
        self.trigger_device.set_frames_per_second_blocking(
            self.config["frames_per_second"]
        )

        self.block_triggerbox_activity = False

        remote_api = MainBrain.RemoteAPI()
        remote_api.post_init(self)

        self.remote_api = remote_api

        self._config_change_functions = []
        self._new_camera_functions = []
        self._old_camera_functions = []

        self.last_requested_image = {}
        self.pending_requests = {}
        self.last_set_param_time = {}

        self.cam_host_sockets = {}

        self.num_cams = 0
        self.MainBrain_cam_ids_copy = []  # keep a copy of all cam_ids connected
        self._ip_addrs_by_cam_id = {}
        self.set_new_camera_callback(self.IncreaseCamCounter)
        self.set_new_camera_callback(self.AddTimestampEchoer)
        self.set_new_camera_callback(self.SendExpectedFPS)
        self.set_old_camera_callback(self.DecreaseCamCounter)

        self.last_saved_data_time = 0.0

        self._currently_recording_movies = {}

        # Attributes accessed by other threads (see the corresponding @property
        # get/set-ters of the attribute for locking (if any)
        self._best_realtime_data = None
        self._framenumber = 0

        self.reconstructor = None

        # Attributes which come in use when saving data occurs
        self.close_pending = False
        self._service_save_data_lock = threading.Lock()
        self.h5file = None
        self.h5filename = ""
        self.h5data2d = None
        self.h5cam_info = None
        self.h5host_clock_info = None
        self.h5trigger_clock_info = None
        self.h5movie_info = None
        self.h5exp_info = None
        self.h5textlog = None
        if 1:
            self.h5data3d_kalman_estimates = None
            self.h5data3d_ML_estimates = None
            self.h5_2d_obs = None

        # Queues of information to save
        self.queue_data2d = Queue.Queue()
        self.queue_host_clock_info = Queue.Queue()
        self.queue_trigger_clock_info = Queue.Queue()
        self.queue_data3d_best = Queue.Queue()

        self.queue_data3d_kalman_estimates = Queue.Queue()
        self.error_ros_msgs_pub = rospy.Publisher("~error", FlydraError, queue_size=100)

        self.coord_processor = CoordinateProcessor(
            self,
            save_profiling_data=save_profiling_data,
            debug_level=self.debug_level,
            show_overall_latency=self.show_overall_latency,
            show_sync_errors=show_sync_errors,
            max_reconstruction_latency_sec=self.config[
                "max_reconstruction_latency_sec"
            ],
            max_N_hypothesis_test=self.config["max_N_hypothesis_test"],
            use_unix_domain_sockets=self.config["use_unix_domain_sockets"],
            posix_scheduler=self.config["posix_scheduler"],
        )
        # self.coord_processor.setDaemon(True)
        self.coord_processor.start()

        self.timestamp_echo_receiver = TimestampEchoReceiver(self)
        self.timestamp_echo_receiver.setDaemon(True)
        self.timestamp_echo_receiver.start()

        # setup ROS
        self.pub_data_file = rospy.Publisher(
            "~data_file", std_msgs.msg.String, queue_size=0, latch=True
        )
        self.pub_data_file.publish("")
        self.pub_calib_file = rospy.Publisher(
            "~calibration", std_msgs.msg.String, queue_size=0, latch=True
        )
        self.pub_calib_file.publish("")
        self.pub_num_cams = rospy.Publisher(
            "~num_cameras", std_msgs.msg.UInt32, queue_size=0, latch=True
        )
        self.pub_num_cams.publish(0)

        self.experiment_uuid = None
        self.sub_exp_uuid = rospy.Subscriber(
            "experiment_uuid", std_msgs.msg.String, self._on_experiment_uuid
        )

        self.services = {}
        for name, srv in self.ROS_CONTROL_API.iteritems():
            self.services[name] = rospy.Service(
                "~%s" % name, srv, self._ros_generic_service_dispatch
            )

        # final config processing
        self.load_calibration(self.config["camera_calibration"])
        self.set_new_tracker(self.config["kalman_model"])
        self.set_save_data_dir(self.config["save_data_dir"])

        main_brain_keeper.register(self)

    def _on_experiment_uuid(self, msg):
        self.experiment_uuid = msg.data
        if self.is_saving_data():
            self.h5exp_info.row["uuid"] = self.experiment_uuid
            self.h5exp_info.row.append()
            self.h5exp_info.flush()

    def _on_trigger_clock_measurement(
        self, start_timestamp, pulsenumber, fraction_n_of_255, stop_timestamp
    ):
        self.queue_trigger_clock_info.put(
            (start_timestamp, pulsenumber, fraction_n_of_255, stop_timestamp)
        )

    def _ros_generic_service_dispatch(self, req):
        calledservice = req._connection_header["service"]
        calledfunction = calledservice.split("/")[-1]
        if calledfunction in self.ROS_CONTROL_API:
            srvclass = self.ROS_CONTROL_API[calledfunction]

            # dynamically build the request and response argument lists for the mainbrain api
            # call. This requires the mainbrain api have the same calling signature as the
            # service definitions, and, if you want to return something over ros, the return
            # type signature must again match the service return type signature

            respclass = srvclass._response_class

            # determine the args to pass to the function based on the srv description (which
            # is embodied in __slots__, a variable created by the ros build system to list
            # the attributes (i.e. parameters only) of the request
            kwargs = {}
            for attr in req.__slots__:
                kwargs[attr] = getattr(req, attr)

            result = getattr(self, calledfunction)(**kwargs)

            kwargs = {}
            for i, attr in enumerate(respclass.__slots__):
                kwargs[attr] = result[i]

            return respclass(**kwargs)

    @property
    def framenumber(self):
        return self._framenumber

    @framenumber.setter
    def framenumber(self, value):
        self._framenumber = value

    @property
    def best_realtime_data(self):
        data = self._best_realtime_data
        self._best_realtime_data = None
        return data

    @best_realtime_data.setter
    def best_realtime_data(self, value):
        self._best_realtime_data = value

    def load_config(self):
        self.config = {}
        for k, v in self.ROS_CONFIGURATION.iteritems():
            self.config[k] = rospy.get_param("~%s" % k, v)

    def save_config(self):
        for k, v in self.config.iteritems():
            if k in self.ROS_CONFIGURATION:
                rospy.set_param("~%s" % k, v)
        for func in self._config_change_functions:
            func()

    def get_fps(self):
        return self.trigger_device.get_frames_per_second()

    def set_fps(self, fps):
        self.do_synchronization(new_fps=fps)

    def get_version(self):
        return (std_msgs.msg.String(flydra_core.version.__version__),)

    def log_message(self, cam_id, timestamp, message):
        self.remote_api.log_message(cam_id.data, timestamp.data, message.data)

    def register_new_camera(
        self, cam_guid, scalar_control_info_json, camnode_ros_name, cam_hostname, cam_ip
    ):
        if len(cam_ip.data) > 0:
            LOG.warn("'cam_ip' parameter set, even though it is deprecated")
        scalar_control_info = json.loads(scalar_control_info_json.data)
        self.remote_api.register_new_cam(
            cam_guid=cam_guid.data,
            scalar_control_info=scalar_control_info,
            camnode_ros_name=camnode_ros_name.data,
            cam_hostname=cam_hostname.data,
        )
        return [std_msgs.msg.Int32(-1)]

    def get_listen_address(self):
        listen_addr = self.remote_api.get_listen_addr()
        listen_addr_json = json.dumps(listen_addr)
        return (std_msgs.msg.String(listen_addr_json),)

    def get_and_clear_commands(self, cam_id):
        cmds = self.remote_api.get_and_clear_commands(cam_id.data)
        cmds_json = json.dumps(cmds)
        return [std_msgs.msg.String(cmds_json)]

    def set_image(self, cam_id, left, bottom, image):
        cam_id = cam_id.data
        lb = left.data, bottom.data
        image = ros_flydra.cv2_bridge.imgmsg_to_numpy(image)
        self.remote_api.set_image(cam_id, (lb, image))

    def receive_missing_data(self, cam_id, framenumber_offset, missing_data_json_buf):
        missing_data = json.loads(missing_data_json_buf.data)
        self.remote_api.receive_missing_data(
            cam_id.data, framenumber_offset.data, missing_data
        )

    def close_xcamera(self, cam_id):
        cam_id = cam_id.data
        self.remote_api.close(cam_id)

    def do_synchronization(self, new_fps=None):
        if self.is_saving_data():
            raise RuntimeError("will not (re)synchronize while saving data")

        self.coord_processor.delete_list_of_synced_cameras()
        self._is_synchronizing = True

        assert self.block_triggerbox_activity == False

        if new_fps is not None:
            self.trigger_device.set_frames_per_second(new_fps)
            actual_new_fps = self.trigger_device.get_frames_per_second()

            # the tracker depends on the framerate
            self.update_tracker_fps(actual_new_fps)

        self.coord_processor.mainbrain_is_attempting_synchronizing()
        self.trigger_device.synchronize(
            flydra_core.common_variables.sync_duration + 1.0
        )
        if new_fps is not None:
            cam_ids = self.remote_api.external_get_cam_ids()
            for cam_id in cam_ids:
                try:
                    self.send_set_camera_property(
                        cam_id, "expected_trigger_framerate", actual_new_fps
                    )
                except Exception as err:
                    LOG.warn("set_camera_property_error %s" % err)

            self.config["frames_per_second"] = float(actual_new_fps)
            self.save_config()

    def IncreaseCamCounter(self, cam_id, scalar_control_info, fqdn):
        self.num_cams += 1
        self.MainBrain_cam_ids_copy.append(cam_id)
        self.pub_num_cams.publish(self.num_cams)

    def AddTimestampEchoer(self, cam_id, scalar_control_info, fqdn):
        if fqdn not in self.cam_host_sockets:
            port = flydra_core.common_variables.timestamp_echo_listener_port
            addrinfo = flydra_socket.make_addrinfo(host=fqdn, port=port)
            self.cam_host_sockets[fqdn] = flydra_socket.FlydraTransportSender(addrinfo)

    def SendExpectedFPS(self, cam_id, scalar_control_info, fqdn):
        self.send_set_camera_property(
            cam_id,
            "expected_trigger_framerate",
            self.trigger_device.get_frames_per_second(),
        )

    def DecreaseCamCounter(self, cam_id):
        try:
            idx = self.MainBrain_cam_ids_copy.index(cam_id)
        except ValueError:
            LOG.warn(
                "IGNORING ERROR: DecreaseCamCounter() called with non-existant cam_id"
            )
            return
        self.num_cams -= 1
        del self.MainBrain_cam_ids_copy[idx]
        self.pub_num_cams.publish(self.num_cams)

    def get_num_cams(self):
        return self.num_cams

    def get_scalarcontrolinfo(self, cam_id):
        sci, fqdn, camnode_ros_name = self.remote_api.external_get_info(cam_id)
        return sci

    def get_widthheight(self, cam_id):
        sci, fqdn, camnode_ros_name = self.remote_api.external_get_info(cam_id)
        w = sci["width"]
        h = sci["height"]
        return w, h

    def get_roi(self, cam_id):
        sci, fqdn, camnode_ros_name = self.remote_api.external_get_info(cam_id)
        lbrt = sci["roi"]
        return lbrt

    def get_all_params(self):
        cam_ids = self.remote_api.external_get_cam_ids()
        all = {}
        for cam_id in cam_ids:
            sci, fqdn, camnode_ros_name = self.remote_api.external_get_info(cam_id)
            all[cam_id] = sci
        return all

    def start_listening(self):
        """ the last thing called before we work - give the config callback watchers a callback
        to check on the state of the mainbrain post __init__ """
        self.save_config()

    def set_config_change_callback(self, handler):
        self._config_change_functions.append(handler)

    def set_new_camera_callback(self, handler):
        self._new_camera_functions.append(handler)

    def set_old_camera_callback(self, handler):
        self._old_camera_functions.append(handler)

    def service_pending(self):
        """the MainBrain application calls this fairly frequently (e.g. every 100 msec)"""
        new_cam_ids, old_cam_ids = self.remote_api.external_get_and_clear_pending_cams()
        for cam_id in new_cam_ids:
            if cam_id in old_cam_ids:
                continue  # inserted and then removed
            if self.is_saving_data():
                raise RuntimeError("Cannot add new camera while saving data")
            sci, fqdn, camnode_ros_name = self.remote_api.external_get_info(cam_id)
            for new_cam_func in self._new_camera_functions:
                new_cam_func(cam_id, sci, fqdn)

        for cam_id in old_cam_ids:
            for old_cam_func in self._old_camera_functions:
                old_cam_func(cam_id)

        now = time.time()
        diff = now - self.last_saved_data_time
        if diff >= 5.0:  # request missing data and save data every 5 seconds
            self._request_missing_data()
            self._locked_service_save_data()
            self.last_saved_data_time = now

        self._check_latencies()

    def _check_latencies(self):
        timestamp_echo_fmt1 = flydra_core.common_variables.timestamp_echo_fmt1
        for sock in self.cam_host_sockets.itervalues():
            buf = struct.pack(timestamp_echo_fmt1, time.time())
            sock.send(buf)

    def get_last_image_fps(self, cam_id):
        # XXX should extend to include lines

        # Points are originally distorted (and align with distorted
        # image).
        (
            image,
            fps,
            points_distorted,
            image_coords,
        ) = self.remote_api.external_get_image_fps_points(cam_id)

        return image, fps, points_distorted, image_coords

    def close_camera(self, cam_id):
        sys.stdout.flush()
        self.remote_api.external_quit(cam_id)
        sys.stdout.flush()

    def start_collecting_background(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.set_collecting_background(cam_id, True)

    def stop_collecting_background(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.set_collecting_background(cam_id, False)

    def set_collecting_background(self, cam_id, value):
        self.remote_api.external_send_set_camera_property(
            cam_id, "collecting_background", value
        )

    def set_color_filter(self, cam_id, value):
        self.remote_api.external_send_set_camera_property(cam_id, "color_filter", value)

    def take_background(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_take_background(cam_id)

    def clear_background(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_clear_background(cam_id)

    def send_set_camera_property(self, cam_id, property_name, value):
        self.remote_api.external_send_set_camera_property(cam_id, property_name, value)

    def request_image_async(self, cam_id):
        self.remote_api.external_request_image_async(cam_id)

    def get_debug_level(self):
        return self.debug_level.isSet()

    def set_debug_level(self, value):
        if value:
            self.debug_level.set()
        else:
            self.debug_level.clear()

    def get_show_overall_latency(self):
        return self.show_overall_latency.isSet()

    def set_show_overall_latency(self, value):
        if value:
            self.show_overall_latency.set()
        else:
            self.show_overall_latency.clear()

    def start_recording(self, raw_file_basename=None, *cam_ids):
        nowstr = time.strftime("%Y%m%d_%H%M%S")
        if not raw_file_basename:
            if self.experiment_uuid is not None:
                raw_file_basename = os.path.join(
                    self.config["save_movie_dir"], self.experiment_uuid,
                )
            else:
                raw_file_basename = os.path.join(self.config["save_movie_dir"], nowstr,)

        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            raw_file_name = os.path.join(raw_file_basename, cam_id, nowstr)
            self.remote_api.external_start_recording(cam_id, raw_file_name)
            approx_start_frame = self.framenumber
            self._currently_recording_movies[cam_id] = (
                raw_file_name,
                approx_start_frame,
            )
            if self.is_saving_data():
                self.h5movie_info.row["cam_id"] = cam_id
                self.h5movie_info.row["filename"] = raw_file_name + ".fmf"
                self.h5movie_info.row["approx_start_frame"] = approx_start_frame
                self.h5movie_info.row.append()
                self.h5movie_info.flush()

    def stop_recording(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_stop_recording(cam_id)
            if cam_id not in self._currently_recording_movies:
                # we're not actually saving...
                continue
            approx_stop_frame = self.framenumber
            raw_file_basename, approx_start_frame = self._currently_recording_movies[
                cam_id
            ]
            del self._currently_recording_movies[cam_id]
            # modify save file to include approximate movie stop time
            if self.is_saving_data():
                nrow = None
                for r in self.h5movie_info:
                    # get row in table
                    if (
                        r["cam_id"] == cam_id
                        and r["filename"] == raw_file_basename + ".fmf"
                        and r["approx_start_frame"] == approx_start_frame
                    ):
                        nrow = r.nrow
                        break
                if nrow is not None:
                    nrowi = int(nrow)  # pytables bug workaround...
                    assert nrowi == nrow  # pytables bug workaround...
                    approx_stop_framei = int(approx_stop_frame)
                    assert approx_stop_framei == approx_stop_frame

                    new_columns = numpy.rec.fromarrays(
                        [[approx_stop_framei]], formats="i8"
                    )
                    self.h5movie_info.modify_columns(
                        start=nrowi, columns=new_columns, names=["approx_stop_frame"]
                    )
                else:
                    raise RuntimeError("could not find row to save movie stop frame.")

    def start_small_recording(self, raw_file_basename=None, *cam_ids):
        nowstr = time.strftime("%Y%m%d_%H%M%S")
        if not raw_file_basename:
            if self.experiment_uuid is not None:
                raw_file_basename = os.path.join(
                    self.config["save_movie_dir"], self.experiment_uuid,
                )
            else:
                raw_file_basename = os.path.join(self.config["save_movie_dir"], nowstr,)

        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            raw_file_name = os.path.join(raw_file_basename, cam_id, nowstr)
            self.remote_api.external_start_small_recording(cam_id, raw_file_name)

    def stop_small_recording(self, *cam_ids):
        if len(cam_ids) == 0:
            cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            self.remote_api.external_stop_small_recording(cam_id)

    def quit(self):
        """closes any files being saved and closes camera connections"""
        # XXX ====== non-isolated calls to remote_api being done ======
        # this may be called twice: once explicitly and once by __del__
        with self.remote_api.cam_info_lock:
            cam_ids = self.remote_api.cam_info.keys()

        for cam_id in cam_ids:
            self.close_camera(cam_id)
        self.remote_api.no_cams_connected.wait(2.0)
        self.remote_api.quit_now.set()  # tell thread to finish
        self.remote_api.thread_done.wait(0.5)  # wait for thread to finish
        if not self.remote_api.no_cams_connected.isSet():
            cam_ids = self.remote_api.cam_info.keys()
            LOG.warn("cameras failed to quit cleanly: %s" % cam_ids)
            # raise RuntimeError('cameras failed to quit cleanly: %s'%str(cam_ids))

        self.stop_saving_data()
        self.coord_processor.quit()

    def load_calibration(self, dirname):
        if self.is_saving_data():
            raise RuntimeError("Cannot (re)load calibration while saving data")

        if not dirname:
            return

        dirname = flydra_core.rosutils.decode_url(dirname)
        if os.path.exists(dirname):
            connected_cam_ids = self.remote_api.external_get_cam_ids()
            self.reconstructor = flydra_core.reconstruct.Reconstructor(dirname)
            calib_cam_ids = self.reconstructor.get_cam_ids()

            calib_cam_ids = calib_cam_ids

            self.coord_processor.set_reconstructor(self.reconstructor)

            self.pub_calib_file.publish(dirname)
            self.config["camera_calibration"] = dirname
            self.save_config()
        else:
            raise ValueError(
                "you specified loading calibration from %r, but that path does not exist"
                % dirname
            )

    def clear_calibration(self):
        if self.is_saving_data():
            raise RuntimeError("Cannot unload calibration while saving data")
        cam_ids = self.remote_api.external_get_cam_ids()
        self.reconstructor = None

        self.coord_processor.set_reconstructor(self.reconstructor)

        self.save_config()

    def update_tracker_fps(self, fps):
        self.set_new_tracker(self.dynamic_model_name, fps)

    def set_new_tracker(self, kalman_model_name, new_fps=None):
        if self.is_saving_data():
            raise RuntimeError("will not set Kalman parameters while saving data")

        self.dynamic_model_name = kalman_model_name

        if self.reconstructor is None:
            return

        fps = self.get_fps() if new_fps is None else new_fps
        dt = 1.0 / fps

        LOG.info("setting model to %s (fps: %s)" % (kalman_model_name, fps))

        dynamic_model = flydra_core.kalman.dynamic_models.get_kalman_model(
            name=kalman_model_name, dt=dt
        )

        self.kalman_saver_info_instance = flydra_kalman_utils.KalmanSaveInfo(
            name=kalman_model_name
        )
        self.KalmanEstimatesDescription = (
            self.kalman_saver_info_instance.get_description()
        )
        self.dynamic_model = dynamic_model

        self.h5_xhat_names = tables.Description(
            self.KalmanEstimatesDescription().columns
        )._v_names

        # send params over to realtime coords thread
        self.coord_processor.set_new_tracker(kalman_model=dynamic_model)
        self.coord_processor.tracker.clear_flushed_callbacks()
        self.coord_processor.tracker.set_flushed_callback(self.finally_close_save_files)

        self.config["kalman_model"] = kalman_model_name
        self.save_config()

    def __del__(self):
        self.quit()

    def _safe_makedir(self, path):
        """ raises OSError if path cannot be made """
        if not os.path.exists(path):
            os.makedirs(path)
            return path

    def set_save_data_dir(self, path):
        path = flydra_core.rosutils.decode_url(path)
        if os.path.isdir(path):
            save_data_dir = path
        else:
            try:
                save_data_dir = self._safe_makedir(path)
            except OSError:
                return None
        self.config["save_data_dir"] = save_data_dir
        self.save_config()
        LOG.info("saving data to %s" % save_data_dir)
        return save_data_dir

    def is_saving_data(self):
        return self.h5file is not None

    def start_saving_data(self, filename=None):
        if self.is_saving_data():
            return

        if not filename:
            filename = time.strftime("%Y%m%d_%H%M%S.mainbrain.h5")
        filename = os.path.join(self.config["save_data_dir"], filename)

        if os.path.exists(filename):
            raise RuntimeError("will not overwrite data file")

        self.h5filename = filename

        LOG.info("saving data to %s" % self.h5filename)
        self.pub_data_file.publish(self.h5filename)

        self.block_triggerbox_activity = True
        self.h5file = tables.open_file(
            os.path.expanduser(self.h5filename), mode="w", title="Flydra data file"
        )
        expected_rows = int(1e6)
        ct = self.h5file.create_table  # shorthand
        root = self.h5file.root  # shorthand
        self.h5data2d = ct(
            root, "data2d_distorted", Info2D, "2d data", expectedrows=expected_rows * 5
        )
        self.h5cam_info = ct(
            root, "cam_info", CamSyncInfo, "Cam Sync Info", expectedrows=500
        )
        self.h5host_clock_info = ct(
            root,
            "host_clock_info",
            HostClockInfo,
            "Host Clock Info",
            expectedrows=6 * 60 * 24,
        )  # 24 hours at 10 sec sample intervals
        self.h5trigger_clock_info = ct(
            root,
            "trigger_clock_info",
            TriggerClockInfo,
            "Trigger Clock Info",
            expectedrows=6 * 60 * 24,
        )  # 24 hours at 10 sec sample intervals
        self.h5movie_info = ct(
            root, "movie_info", MovieInfo, "Movie Info", expectedrows=500
        )
        self.h5textlog = ct(root, "textlog", TextLogDescription, "text log")
        self.h5exp_info = ct(
            root, "experiment_info", ExperimentInfo, "ExperimentInfo", expectedrows=100
        )

        self._startup_message()
        if self.reconstructor is not None:
            self.reconstructor.save_to_h5file(self.h5file)
            if 1:
                self.h5data3d_kalman_estimates = ct(
                    root,
                    "kalman_estimates",
                    self.KalmanEstimatesDescription,
                    "3d data (from Kalman filter)",
                    expectedrows=expected_rows,
                )
                self.h5data3d_kalman_estimates.attrs.dynamic_model_name = (
                    self.dynamic_model_name
                )
                self.h5data3d_kalman_estimates.attrs.dynamic_model = self.dynamic_model

                self.h5data3d_ML_estimates = ct(
                    root,
                    "ML_estimates",
                    FilteredObservations,
                    "dynamics-free maximum liklihood estimates",
                    expectedrows=expected_rows,
                )
                self.h5_2d_obs = self.h5file.create_vlarray(
                    self.h5file.root,
                    "ML_estimates_2d_idxs",
                    ML_estimates_2d_idxs_type(),  # dtype should match with tro.observations_2d
                    "camns and idxs",
                )
                self.h5_2d_obs_next_idx = 0

        general_save_info = self.coord_processor.get_general_cam_info()
        for cam_id, dd in general_save_info.iteritems():
            self.h5cam_info.row["cam_id"] = cam_id
            self.h5cam_info.row["camn"] = dd["absolute_cam_no"]
            with self.remote_api.cam_info_lock:
                self.h5cam_info.row["hostname"] = self.remote_api.cam_info[cam_id][
                    "fqdn"
                ]
            self.h5cam_info.row.append()
        self.h5cam_info.flush()

        # save raw image from each camera
        img = self.h5file.create_group(root, "images", "sample images")
        cam_ids = self.remote_api.external_get_cam_ids()
        for cam_id in cam_ids:
            image, fps, points_distorted, image_coords = self.get_last_image_fps(cam_id)
            if image is None:
                raise ValueError("image cannot be None")
            self.h5file.create_array(
                img, cam_id, image, "sample image from %s" % cam_id
            )

        self.save_config()

        if self.coord_processor.tracker is not None:
            # force all new tracked objects by killing existing tracks
            self.coord_processor.tracker.kill_all_trackers()

    def stop_saving_data(self):
        LOG.info("received request to stop saving file")
        self.close_pending = True
        if self.coord_processor.tracker is not None:
            # eventually this will trigger a call to self.finally_close_save_files()
            self.coord_processor.tracker.kill_all_trackers()
        else:
            self.finally_close_save_files()

    def finally_close_save_files(self):
        if not self.close_pending:
            return
        self.close_pending = False  # after the following, we will already be closed...

        with self._service_save_data_lock:
            LOG.info("entering final save data service call")
            self._service_save_data()  # we absolutely want to save
            LOG.info("entering done with final save data service call")
            if self.is_saving_data():
                self.h5file.close()
                self.h5file = None
                self.h5filename = ""
                self.pub_data_file.publish(self.h5filename)
                self.block_triggerbox_activity = False
                LOG.info("closed h5 file")
            else:
                LOG.info("saving already stopped, cannot stop again")
            self.h5data2d = None
            self.h5cam_info = None
            self.h5host_clock_info = None
            self.h5trigger_clock_info = None
            self.h5movie_info = None
            self.h5exp_info = None
            self.h5textlog = None
            self.h5data3d_kalman_estimates = None
            self.h5data3d_ML_estimates = None
            self.h5_2d_obs = None

            self.save_config()

    def _startup_message(self):
        textlog_row = self.h5textlog.row
        cam_id = "mainbrain"
        timestamp = time.time()

        # Get local timezone name. See https://stackoverflow.com/a/17365806/1633026
        local_tz_name = tzlocal.get_localzone()

        # This line is important (including the formatting). It is
        # read by flydra_analysis.a2.check_atmel_clock.

        list_of_textlog_data = [
            (
                timestamp,
                cam_id,
                timestamp,
                "MainBrain running at %s fps, (flydra_version %s, time_tzname0 %s)"
                % (
                    self.trigger_device.get_frames_per_second(),
                    flydra_core.version.__version__,
                    local_tz_name,
                ),
            ),
            (
                timestamp,
                cam_id,
                timestamp,
                "using flydra version %s" % (flydra_core.version.__version__,),
            ),
        ]

        list_of_textlog_data.append(
            (timestamp, cam_id, timestamp, "using numpy version %s" % numpy.__version__)
        )

        list_of_textlog_data.append(
            (
                timestamp,
                cam_id,
                timestamp,
                "using pytables version %s" % tables.__version__,
            )
        )

        for lib in ("hdf5", "zlib", "lzo", "bzip2", "blosc"):
            try:
                _, ver, _ = tables.which_lib_version(lib)
                list_of_textlog_data.append(
                    (
                        timestamp,
                        cam_id,
                        timestamp,
                        "using pytables:%s version %s" % (lib, ver),
                    )
                )
            except ValueError:
                # unknown lib
                pass

        for textlog_data in list_of_textlog_data:
            (mainbrain_timestamp, cam_id, host_timestamp, message) = textlog_data
            textlog_row["mainbrain_timestamp"] = mainbrain_timestamp
            textlog_row["cam_id"] = cam_id
            textlog_row["host_timestamp"] = host_timestamp
            textlog_row["message"] = message
            textlog_row.append()

        self.h5textlog.flush()

    def _request_missing_data(self):
        if ATTEMPT_DATA_RECOVERY:
            # request from camera computers any data that we're missing
            missing_data_dict = self.coord_processor.get_missing_data_dict()
            for (
                camn,
                (cam_id, framenumber_offset, list_of_missing_framenumbers),
            ) in missing_data_dict.iteritems():
                self.remote_api.external_request_missing_data(
                    cam_id, camn, framenumber_offset, list_of_missing_framenumbers
                )

    def _locked_service_save_data(self):
        with self._service_save_data_lock:
            self._service_save_data()

    def _service_save_data(self):
        # ** 2d data **
        #   clear queue
        list_of_rows_of_data2d = []
        try:
            while True:
                tmp = self.queue_data2d.get(0)
                list_of_rows_of_data2d.extend(tmp)
        except Queue.Empty:
            pass
        #   save
        if self.h5data2d is not None and len(list_of_rows_of_data2d):
            # it's much faster to convert to numpy first:
            recarray = numpy.rec.array(
                list_of_rows_of_data2d, dtype=Info2DCol_description
            )
            self.h5data2d.append(recarray)
            self.h5data2d.flush()

        # ** textlog **
        if self.h5textlog is not None:
            # we don't want to miss messages, so wait until we are saving
            list_of_textlog_data = []
            try:
                while True:
                    tmp = self.remote_api.message_queue.get(0)
                    list_of_textlog_data.append(tmp)
            except Queue.Empty:
                pass

            if list_of_textlog_data:
                textlog_row = self.h5textlog.row
                for textlog_data in list_of_textlog_data:
                    (
                        mainbrain_timestamp,
                        cam_id,
                        host_timestamp,
                        message,
                    ) = textlog_data
                    textlog_row["mainbrain_timestamp"] = mainbrain_timestamp
                    textlog_row["cam_id"] = cam_id
                    textlog_row["host_timestamp"] = host_timestamp
                    textlog_row["message"] = message
                    textlog_row.append()

                self.h5textlog.flush()

        if 1:
            # ** 3d data - kalman **
            q = self.queue_data3d_kalman_estimates

            #   clear queue
            list_of_3d_data = []
            try:
                while True:
                    list_of_3d_data.append(q.get(0))
            except Queue.Empty:
                pass
            if self.h5data3d_kalman_estimates is not None:
                for (
                    obj_id,
                    tro_frames,
                    tro_xhats,
                    tro_Ps,
                    tro_timestamps,
                    obs_frames,
                    obs_data,
                    observations_2d,
                    obs_Lcoords,
                ) in list_of_3d_data:

                    if len(obs_frames) < MIN_KALMAN_OBSERVATIONS_TO_SAVE:
                        # only save data with at least N observations
                        continue

                    # save observation 2d data indexes
                    this_idxs = []
                    for camns_and_idxs in observations_2d:
                        this_idxs.append(self.h5_2d_obs_next_idx)
                        self.h5_2d_obs.append(camns_and_idxs)
                        self.h5_2d_obs_next_idx += 1
                    self.h5_2d_obs.flush()

                    this_idxs = numpy.array(
                        this_idxs, dtype=numpy.uint64
                    )  # becomes obs_2d_idx (index into 'ML_estimates_2d_idxs')

                    # save observations
                    observations_frames = numpy.array(obs_frames, dtype=numpy.uint64)
                    obj_id_array = numpy.empty(
                        observations_frames.shape, dtype=numpy.uint32
                    )
                    obj_id_array.fill(obj_id)
                    observations_data = numpy.array(obs_data, dtype=numpy.float32)
                    observations_Lcoords = numpy.array(obs_Lcoords, dtype=numpy.float32)
                    list_of_obs = [
                        observations_data[:, i]
                        for i in range(observations_data.shape[1])
                    ]
                    list_of_lines = [
                        observations_Lcoords[:, i]
                        for i in range(observations_Lcoords.shape[1])
                    ]
                    array_list = (
                        [obj_id_array, observations_frames]
                        + list_of_obs
                        + [this_idxs]
                        + list_of_lines
                    )
                    obs_recarray = numpy.rec.fromarrays(array_list, names=h5_obs_names)

                    self.h5data3d_ML_estimates.append(obs_recarray)
                    self.h5data3d_ML_estimates.flush()

                    # save xhat info (kalman estimates)
                    frames = numpy.array(tro_frames, dtype=numpy.uint64)
                    timestamps = numpy.array(tro_timestamps, dtype=numpy.float64)
                    xhat_data = numpy.array(tro_xhats, dtype=numpy.float32)
                    P_data_full = numpy.array(tro_Ps, dtype=numpy.float32)
                    obj_id_array = numpy.empty(frames.shape, dtype=numpy.uint32)
                    obj_id_array.fill(obj_id)
                    list_of_xhats = [xhat_data[:, i] for i in range(xhat_data.shape[1])]
                    ksii = self.kalman_saver_info_instance
                    list_of_Ps = ksii.covar_mats_to_covar_entries(P_data_full)
                    xhats_recarray = numpy.rec.fromarrays(
                        [obj_id_array, frames, timestamps] + list_of_xhats + list_of_Ps,
                        names=self.h5_xhat_names,
                    )

                    self.h5data3d_kalman_estimates.append(xhats_recarray)
                    self.h5data3d_kalman_estimates.flush()

        # ** camera info **
        #   clear queue
        list_of_host_clock_info = []
        try:
            while True:
                list_of_host_clock_info.append(self.queue_host_clock_info.get(0))
        except Queue.Empty:
            pass
        #   save
        if self.h5host_clock_info is not None:
            host_clock_info_row = self.h5host_clock_info.row
            for host_clock_info in list_of_host_clock_info:
                (
                    remote_hostname,
                    start_timestamp,
                    remote_timestamp,
                    stop_timestamp,
                ) = host_clock_info
                host_clock_info_row["remote_hostname"] = remote_hostname
                host_clock_info_row["start_timestamp"] = start_timestamp
                host_clock_info_row["remote_timestamp"] = remote_timestamp
                host_clock_info_row["stop_timestamp"] = stop_timestamp
                host_clock_info_row.append()

            self.h5host_clock_info.flush()

        #   clear queue
        list_of_trigger_clock_info = []
        try:
            while True:
                list_of_trigger_clock_info.append(self.queue_trigger_clock_info.get(0))
        except Queue.Empty:
            pass
        #   save
        if self.h5trigger_clock_info is not None:
            row = self.h5trigger_clock_info.row
            for trigger_clock_info in list_of_trigger_clock_info:
                start_timestamp, framecount, tcnt, stop_timestamp = trigger_clock_info
                row["start_timestamp"] = start_timestamp
                row["framecount"] = framecount
                row["tcnt"] = tcnt
                row["stop_timestamp"] = stop_timestamp
                row.append()

            self.h5trigger_clock_info.flush()
