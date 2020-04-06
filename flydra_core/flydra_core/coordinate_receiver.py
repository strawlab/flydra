from __future__ import with_statement, division
import threading, time, socket, select, os, copy, struct
import warnings
import collections
import errno

import flydra_core.reconstruct
import flydra_core._reconstruct_utils as ru
import numpy
import numpy as np
from numpy import nan, inf
from flydra_core.common_variables import near_inf
import flydra_core.flydra_socket as flydra_socket
import Queue

pytables_filt = numpy.asarray
import pickle

import flydra_core.version
import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils
import flydra_core.kalman.flydra_tracker
import flydra_core._fastgeom as geom

# import flydra_core.geom as geom

import flydra_core.data_descriptions

# ensure that pytables uses numpy:
import tables
import tables.flavor

tables.flavor.restrict_flavors(keep=["numpy"])
warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

import roslib

roslib.load_manifest("rospy")
roslib.load_manifest("ros_flydra")
import rospy
import std_msgs.msg
from ros_flydra.msg import flydra_mainbrain_super_packet, CameraList
from ros_flydra.msg import flydra_mainbrain_packet, flydra_object, FlydraError
from geometry_msgs.msg import Point, Vector3

import flydra_core.rosutils

LOG = flydra_core.rosutils.Log(to_ros=True)

import flydra_core.debuglock

DebugLock = flydra_core.debuglock.DebugLock

ATTEMPT_DATA_RECOVERY = bool(int(os.environ.get("ATTEMPT_DATA_RECOVERY", "1")))

IMPOSSIBLE_TIMESTAMP = -10.0

PT_TUPLE_IDX_X = flydra_core.data_descriptions.PT_TUPLE_IDX_X
PT_TUPLE_IDX_Y = flydra_core.data_descriptions.PT_TUPLE_IDX_Y
PT_TUPLE_IDX_FRAME_PT_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX
PT_TUPLE_IDX_CUR_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_CUR_VAL_IDX
PT_TUPLE_IDX_MEAN_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_MEAN_VAL_IDX
PT_TUPLE_IDX_SUMSQF_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_SUMSQF_VAL_IDX


class CoordinateProcessor(threading.Thread):
    def __init__(
        self,
        main_brain,
        save_profiling_data,
        debug_level,
        show_sync_errors,
        show_overall_latency,
        max_reconstruction_latency_sec,
        max_N_hypothesis_test,
        use_unix_domain_sockets,
        posix_scheduler="",
    ):
        self.did_quit_successfully = False
        self.main_brain = main_brain
        self.debug_level = debug_level
        self.show_overall_latency = show_overall_latency
        self.max_reconstruction_latency_sec = max_reconstruction_latency_sec
        self.max_N_hypothesis_test = max_N_hypothesis_test
        self.posix_scheduler = posix_scheduler

        self._synchronized_cameras = []
        self.sync_cam_pub = None
        self.frame_offset_pubs = {}
        if rospy.core.is_initialized():
            self.sync_cam_pub = rospy.Publisher(
                "~synchronized_cameras", CameraList, queue_size=1, latch=True
            )

        self.save_profiling_data = save_profiling_data
        if self.save_profiling_data:
            self.data_dict_queue = []

        self.cam_ids = []
        self.absolute_cam_nos = []  # a.k.a. "camn"
        self.frame_offsets = {}
        self.last_frame_times = {}
        self.last_timestamps = []
        self.last_framenumbers_delay = []
        self.last_framenumbers_skip = []
        if ATTEMPT_DATA_RECOVERY:
            # self.request_data_lock = DebugLock('request_data_lock',True) # protect request_data
            self.request_data_lock = threading.Lock()  # protect request_data
            self.request_data = {}
        self.cam_id2cam_no = {}
        self.camn2cam_id = {}
        self.reconstructor = None
        self.tracker = None
        self.ever_synchronized = False
        self.show_sync_errors = show_sync_errors

        self.ip2hostname = {}

        self.tracker_lock = threading.Lock()
        # self.tracker_lock = DebugLock('tracker_lock',verbose=True)

        self.all_data_lock = threading.Lock()
        # self.all_data_lock = DebugLock('all_data_lock',verbose=False)
        self.quit_event = threading.Event()

        self.max_absolute_cam_nos = -1

        self.general_save_info = {}

        self.to_unlink = []

        if use_unix_domain_sockets:
            addr = "/tmp/flydra_coordinate_receiver." + str(os.getpid())
            try:
                os.remove(addr)
            except OSError as err:
                if err.errno != errno.ENOENT:  # Missing file is OK.
                    raise
            addrinfo = flydra_socket.make_addrinfo(filename=addr)
            self.to_unlink.append(addr)
        else:
            addrinfo = flydra_socket.make_addrinfo(
                host=flydra_socket.get_bind_address(), port=0
            )
        self.listen_socket = flydra_socket.FlydraTransportReceiver(
            addrinfo, socket_timeout_seconds=0.5
        )
        self.listen_address = self.listen_socket.get_listen_addrinfo().to_dict()
        if "host" in self.listen_address and self.listen_address["host"] == "0.0.0.0":
            self.listen_address["host"] = flydra_core.rosutils.get_node_hostname(
                rospy.get_name()
            )
        LOG.info("coordinate receiver listening at %r" % self.listen_address)

        self.set_reconstructor(None)

        self.realtime_ros_packets_pub = rospy.Publisher(
            "~super_packets",
            flydra_mainbrain_super_packet,
            queue_size=100,
            tcp_nodelay=True,
        )

        self.synchronze_ros_msgs_pub = rospy.Publisher(
            "~synchronize", std_msgs.msg.String, queue_size=100
        )

        self.realtime_coord_dict = {}
        self.timestamp_check_dict = {}
        self.realtime_kalman_coord_dict = collections.defaultdict(dict)
        self.oldest_timestamp_by_corrected_framenumber = {}

        threading.Thread.__init__(self, name="CoordinateProcessor")

    def get_listen_address(self):
        return self.listen_address

    def mainbrain_is_attempting_synchronizing(self):
        self.ever_synchronized = True
        with self.tracker_lock:
            if self.tracker is not None:
                self.tracker.kill_all_trackers()  # delete all old data

    def get_general_cam_info(self):
        with self.all_data_lock:
            result = self.general_save_info.copy()
        return result

    def get_missing_data_dict(self):
        # called from main thread, must lock data in realtime coord thread
        result_by_camn = {}
        with self.request_data_lock:
            for absolute_cam_no, tmp_queue in self.request_data.iteritems():
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

                        assert (
                            cam_id == this_cam_id
                        )  # make sure given camn comes from single cam_id
                        assert framenumber_offset == this_framenumber_offset

                        list_of_missing_framenumbers.extend(this_list)
                except Queue.Empty:
                    pass
                if len(list_of_missing_framenumbers):
                    result_by_camn[absolute_cam_no] = (
                        cam_id,
                        framenumber_offset,
                        list_of_missing_framenumbers,
                    )
        return result_by_camn

    def set_reconstructor(self, r):
        # called from main thread, must lock to send to realtime coord thread
        with self.all_data_lock:
            self.reconstructor = r

        R = self.reconstructor
        self.cached_calibration_by_cam_id = {}
        if R is None:
            return
        for cam_id in R.cam_ids:
            scc = R.get_SingleCameraCalibration(cam_id)
            cc = R.get_camera_center(cam_id)[:, 0]
            cc = np.array([cc[0], cc[1], cc[2], 1.0])
            self.cached_calibration_by_cam_id[cam_id] = scc, cc

    def set_new_tracker(self, kalman_model=None):
        # called from main thread, must lock to send to realtime coord thread
        tracker = flydra_core.kalman.flydra_tracker.Tracker(
            self.reconstructor, kalman_model=kalman_model
        )
        tracker.set_killed_tracker_callback(self.enqueue_finished_tracked_object)
        with self.tracker_lock:
            if self.tracker is not None:
                self.tracker.kill_all_trackers()  # save (if necessary) all old data
            self.tracker = tracker  # bind to name, replacing old tracker
            if self.save_profiling_data:
                tracker = copy.copy(self.tracker)
                tracker.kill_tracker_callbacks = []
                if 1:
                    raise NotImplementedError("")
                self.data_dict_queue.append(("tracker", tracker))

    def enqueue_finished_tracked_object(self, tracked_object):
        # this is from called within the realtime coords thread
        if self.main_brain.is_saving_data():
            self.main_brain.queue_data3d_kalman_estimates.put(
                (
                    tracked_object.obj_id,
                    tracked_object.frames,
                    tracked_object.xhats,
                    tracked_object.Ps,
                    tracked_object.timestamps,
                    tracked_object.observations_frames,
                    tracked_object.MLE_position,
                    tracked_object.observations_2d,
                    tracked_object.MLE_Lcoords,
                )
            )
        # send a ROS message that this object is dead
        this_ros_object = flydra_object(
            obj_id=tracked_object.obj_id,
            position=Point(numpy.nan, numpy.nan, numpy.nan),
        )
        try:
            reconstruction_stamp = rospy.Time.now()
            acquire_stamp = rospy.Time.from_sec(0)
        except rospy.exceptions.ROSInitException as err:
            pass
        else:
            ros_packet = flydra_mainbrain_packet(
                framenumber=tracked_object.current_frameno,
                reconstruction_stamp=reconstruction_stamp,
                acquire_stamp=acquire_stamp,
                objects=[this_ros_object],
            )
            self.realtime_ros_packets_pub.publish(
                flydra_mainbrain_super_packet([ros_packet])
            )

        if self.debug_level.isSet():
            LOG.debug("killing obj_id %d:" % tracked_object.obj_id)

    def connect(self, cam_id):
        # called from Remote-API thread on camera connect
        assert not self.main_brain.is_saving_data()

        with self.all_data_lock:
            self.cam_ids.append(cam_id)

            # find absolute_cam_no
            self.max_absolute_cam_nos += 1
            absolute_cam_no = self.max_absolute_cam_nos
            self.absolute_cam_nos.append(absolute_cam_no)

            self.camn2cam_id[absolute_cam_no] = cam_id
            self.cam_id2cam_no[cam_id] = absolute_cam_no

            self.last_timestamps.append(
                IMPOSSIBLE_TIMESTAMP
            )  # arbitrary impossible number
            self.last_framenumbers_delay.append(-1)  # arbitrary impossible number
            self.last_framenumbers_skip.append(-1)  # arbitrary impossible number
            self.general_save_info[cam_id] = {"absolute_cam_no": absolute_cam_no}

    def disconnect(self, cam_id):
        # called from Remote-API thread on camera disconnect
        cam_idx = self.cam_ids.index(cam_id)
        with self.all_data_lock:
            del self.cam_ids[cam_idx]
            del self.absolute_cam_nos[cam_idx]
            del self.last_timestamps[cam_idx]
            del self.last_framenumbers_delay[cam_idx]
            del self.last_framenumbers_skip[cam_idx]
            del self.general_save_info[cam_id]

    def quit(self):
        # called from outside of thread to quit the thread
        if self.save_profiling_data:
            fname = "data_for_kalman_profiling.pkl"
            fullpath = os.path.abspath(fname)
            LOG.info("saving data for profiling to %s" % fullpath)
            to_save = self.data_dict_queue
            save_fd = open(fullpath, mode="wb")
            pickle.dump(to_save, save_fd)
            save_fd.close()
            LOG.info("done saving")
        self.quit_event.set()
        self.join()  # wait until CoordReveiver thread quits

    def OnSynchronize(self, cam_idx, cam_id, framenumber, timestamp):

        if self.main_brain.is_saving_data():
            LOG.warn("re-synchronized while saving data!")
            return

        if 1:
            self.synchronze_ros_msgs_pub.publish(std_msgs.msg.String(cam_id))
            LOG.info("%s (re)synchronized" % cam_id)
            # discard all previous data
            for k in self.realtime_coord_dict.keys():
                del self.realtime_coord_dict[k]
                del self.timestamp_check_dict[k]
            for k in self.realtime_kalman_coord_dict.keys():
                del self.realtime_kalman_coord_dict[k]
            for k in self.oldest_timestamp_by_corrected_framenumber.keys():
                del self.oldest_timestamp_by_corrected_framenumber[k]

        # make new absolute_cam_no to indicate new synchronization state
        self.max_absolute_cam_nos += 1
        absolute_cam_no = self.max_absolute_cam_nos
        self.absolute_cam_nos[cam_idx] = absolute_cam_no

        self.camn2cam_id[absolute_cam_no] = cam_id
        self.cam_id2cam_no[cam_id] = absolute_cam_no

        self.general_save_info[cam_id]["absolute_cam_no"] = absolute_cam_no

        # because saving usually follows synchronization, ensure we have a recent
        # image to put in the h5 file
        self.main_brain.remote_api.external_request_image_async(cam_id)

    def _publish_list_of_synced_cameras(self):
        if self.sync_cam_pub is None:
            return
        msg = CameraList()
        msg.cameras = self._synchronized_cameras
        self.sync_cam_pub.publish(msg)

    def delete_list_of_synced_cameras(self):
        del self._synchronized_cameras[:]
        self._publish_list_of_synced_cameras()

    def _new_frame_offset(self, cam_id, framenumber):
        self.frame_offsets[cam_id] = framenumber
        if not rospy.core.is_initialized():
            return

        msg = std_msgs.msg.UInt64(data=framenumber)

        if cam_id not in self.frame_offset_pubs:
            pub = rospy.Publisher("/%s/frame_offset" % (cam_id,), std_msgs.msg.UInt64,)
            self.frame_offset_pubs[cam_id] = pub
        self.frame_offset_pubs[cam_id].publish(msg)

    def register_frame(self, cam_id, framenumber):
        frame_timestamp = time.time()
        try:
            last_frame_timestamp = self.last_frame_times[cam_id]
        except KeyError:
            # very first frame from this cam_id
            self._new_frame_offset(cam_id, framenumber)
            did_frame_offset_change = True
            corrected_framenumber = framenumber - self.frame_offsets[cam_id]
            self.last_frame_times[cam_id] = frame_timestamp
            return corrected_framenumber, did_frame_offset_change

        this_interval = frame_timestamp - last_frame_timestamp

        did_frame_offset_change = False
        if self.main_brain._is_synchronizing:
            if this_interval > flydra_core.common_variables.sync_duration:
                self._new_frame_offset(cam_id, framenumber)
                did_frame_offset_change = True
                self._synchronized_cameras.append(cam_id)
                self._publish_list_of_synced_cameras()

                if len(self._synchronized_cameras) == len(self.cam_ids):
                    # success, done synchronizing all cameras
                    self.main_brain._is_synchronizing = False
        else:
            if this_interval > flydra_core.common_variables.sync_duration:
                LOG.warn("long IFI not during intended synchronization detected")

        self.last_frame_times[cam_id] = frame_timestamp

        corrected_framenumber = framenumber - self.frame_offsets[cam_id]
        return corrected_framenumber, did_frame_offset_change

    def run(self):
        """main loop of CoordinateProcessor"""

        if self.posix_scheduler != "":
            import posix_sched

            mode_str, priority = self.posix_scheduler
            mode = getattr(posix_sched, mode_str)
            try:
                sched_params = posix_sched.SchedParam(priority)
                posix_sched.setscheduler(0, mode, sched_params)
                LOG.info(
                    "3D reconstruction thread running with priority %s"
                    % self.posix_scheduler
                )
            except Exception as x:
                LOG.warn(
                    "could not adjust priority (PID %d): %s" % (os.getpid(), str(x))
                )
        else:
            LOG.info("3D reconstruction thread running with default priority")
        LOG.info("CoordinateProcessor running on PID %d" % os.getpid())
        # self.main_brain.trigger_device.wait_for_estimate()
        while not self.quit_event.isSet():
            try:
                incoming_2d_data = self.listen_socket.recv()
            except socket.error as err:
                if err.errno == errno.EAGAIN:
                    # no data ready. try again (after checking if we should quit).
                    continue
                else:
                    raise
            self.process_data(incoming_2d_data)
        self.finish_processing()

    def finish_processing(self):
        with self.tracker_lock:
            if self.tracker is not None:
                self.tracker.kill_all_trackers()  # save (if necessary) all old data

        for fname in self.to_unlink:
            os.remove(fname)

        self.did_quit_successfully = True

    def process_data(self, buf_data):

        header_fmt = flydra_core.common_variables.recv_pt_header_fmt
        header_size = struct.calcsize(header_fmt)
        pt_fmt = flydra_core.common_variables.recv_pt_fmt
        pt_size = struct.calcsize(pt_fmt)

        no_point_tuple = (
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            False,
            0,
            0,
            0,
            0,
        )

        new_data_framenumbers = set()

        deferred_2d_data = []
        header = buf_data[:header_size]
        assert len(header) == header_size
        # this raw_timestamp is the remote camera's timestamp (?? from the driver, not the host clock??)
        (
            cam_id,
            raw_timestamp,
            camn_received_time,
            raw_framenumber,
            n_pts,
            n_frames_skipped,
        ) = struct.unpack(header_fmt, header)

        with self.all_data_lock:
            cam_idx = self.cam_ids.index(cam_id)
            absolute_cam_no = self.absolute_cam_nos[cam_idx]

            points_in_pluecker_coords_meters = []
            points_undistorted = []
            points_distorted = []

            assert len(buf_data) == (header_size + n_pts * pt_size)

            predicted_framenumber = (
                n_frames_skipped + self.last_framenumbers_skip[cam_idx] + 1
            )
            if raw_framenumber < predicted_framenumber:
                LOG.fatal("cam_id %s" % cam_id)
                LOG.fatal("raw_framenumber %s" % raw_framenumber)
                LOG.fatal("n_frames_skipped %s" % n_frames_skipped)
                LOG.fatal("predicted_framenumber %s" % predicted_framenumber)
                LOG.fatal(
                    "self.last_framenumbers_skip[cam_idx] %s"
                    % self.last_framenumbers_skip[cam_idx]
                )
                raise RuntimeError("got framenumber already received or skipped!")
            elif raw_framenumber > predicted_framenumber:
                if not self.last_framenumbers_skip[cam_idx] == -1:
                    # this is not the first frame
                    # probably because network buffer filled up before we emptied it
                    LOG.warn("frame data loss %s" % cam_id)
                    self.main_brain.error_ros_msgs_pub.publish(
                        FlydraError(FlydraError.FRAME_DATA_LOSS, cam_id)
                    )

                if ATTEMPT_DATA_RECOVERY:
                    if not self.last_framenumbers_skip[cam_idx] == -1:
                        # this is not the first frame
                        missing_frame_numbers = range(
                            self.last_framenumbers_skip[cam_idx] + 1, raw_framenumber
                        )

                        with self.request_data_lock:
                            tmp_queue = self.request_data.setdefault(
                                absolute_cam_no, Queue.Queue()
                            )

                        tmp_framenumber_offset = self.frame_offsets[cam_id]
                        tmp_queue.put(
                            (cam_id, tmp_framenumber_offset, missing_frame_numbers)
                        )
                        del tmp_framenumber_offset
                        del tmp_queue  # drop reference to queue
                        del missing_frame_numbers

            self.last_framenumbers_skip[cam_idx] = raw_framenumber
            start = header_size
            if n_pts:
                # valid points
                for frame_pt_idx in range(n_pts):
                    end = start + pt_size
                    (
                        x_distorted,
                        y_distorted,
                        area,
                        slope,
                        eccentricity,
                        slope_found,
                        cur_val,
                        mean_val,
                        sumsqf_val,
                    ) = struct.unpack(pt_fmt, buf_data[start:end])
                    # nan cannot get sent across network in platform-independent way

                    if slope == near_inf:
                        slope = inf
                    if eccentricity == near_inf:
                        eccentricity = inf
                    if not slope_found:
                        slope = nan

                    line_found = slope_found

                    if slope_found:
                        if np.isinf(slope):
                            run = 0.0
                            rise = 1.0
                        else:
                            run = 1.0
                            rise = slope
                    else:
                        run = nan
                        rise = nan

                    if cam_id in self.cached_calibration_by_cam_id:
                        scc, cc = self.cached_calibration_by_cam_id[cam_id]
                        x_undistorted, y_undistorted = scc.helper.undistort(
                            x_distorted, y_distorted
                        )
                        (
                            p1,
                            p2,
                            p3,
                            p4,
                            ray0,
                            ray1,
                            ray2,
                            ray3,
                            ray4,
                            ray5,
                        ) = flydra_core.reconstruct.do_3d_operations_on_2d_point(
                            scc.helper,
                            x_undistorted,
                            y_undistorted,
                            scc.pmat_inv,
                            cc,
                            x_distorted,
                            y_distorted,
                            rise,
                            run,
                        )
                        ray_valid = not np.isnan(ray0)
                    else:
                        scc, cc = None, None
                        x_undistorted, y_undistorted = np.nan, np.nan
                        (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4, ray5) = [
                            np.nan
                        ] * 10
                        ray_valid = False

                    # Keep in sync with flydra_analysis.kalmanize and flydra_core.data_descriptions
                    pt_undistorted = (
                        x_undistorted,
                        y_undistorted,
                        area,
                        slope,
                        eccentricity,
                        p1,
                        p2,
                        p3,
                        p4,
                        line_found,
                        frame_pt_idx,
                        cur_val,
                        mean_val,
                        sumsqf_val,
                    )
                    pt_distorted = (
                        x_distorted,
                        y_distorted,
                        area,
                        slope,
                        eccentricity,
                        p1,
                        p2,
                        p3,
                        p4,
                        line_found,
                        frame_pt_idx,
                        cur_val,
                        mean_val,
                        sumsqf_val,
                    )
                    if ray_valid:
                        points_in_pluecker_coords_meters.append(
                            (
                                pt_undistorted,
                                geom.line_from_HZline(
                                    (ray0, ray1, ray2, ray3, ray4, ray5)
                                ),
                            )
                        )
                    points_undistorted.append(pt_undistorted)
                    points_distorted.append(pt_distorted)
                    start = end
            else:
                # no points found
                end = start
                # append non-point to allow correlation of
                # timestamps with frame number
                points_distorted.append(no_point_tuple)
                points_undistorted.append(no_point_tuple)

            self._process_parsed_data(
                cam_idx,
                camn_received_time,
                absolute_cam_no,
                n_pts,
                cam_id,
                raw_framenumber,
                new_data_framenumbers,
                points_in_pluecker_coords_meters,
                points_distorted,
                points_undistorted,
                deferred_2d_data,
            )

    def _process_parsed_data(
        self,
        cam_idx,
        camn_received_time,
        absolute_cam_no,
        n_pts,
        cam_id,
        raw_framenumber,
        new_data_framenumbers,
        points_in_pluecker_coords_meters,
        points_distorted,
        points_undistorted,
        deferred_2d_data,
    ):
        # Note: this must be called with self.all_data_lock acquired.

        timestamp_check_dict = self.timestamp_check_dict
        convert_format = flydra_kalman_utils.convert_format  # shorthand
        realtime_coord_dict = self.realtime_coord_dict
        oldest_timestamp_by_corrected_framenumber = (
            self.oldest_timestamp_by_corrected_framenumber
        )
        realtime_kalman_coord_dict = self.realtime_kalman_coord_dict

        if 1:
            if 1:
                if 1:
                    if 1:
                        # Use camn_received_time to determine sync
                        # info. This avoids 2 potential problems:
                        #  * using raw_timestamp can fail if the
                        #    camera drivers don't provide useful data
                        #  * using time.time() can fail if the network
                        #    latency jitter is on the order of the
                        #    inter frame interval.
                        (
                            corrected_framenumber,
                            did_frame_offset_change,
                        ) = self.register_frame(cam_id, raw_framenumber)

                        trigger_timestamp = self.main_brain.trigger_device.framestamp2timestamp(
                            corrected_framenumber
                        )
                        if did_frame_offset_change:
                            self.OnSynchronize(
                                cam_idx, cam_id, raw_framenumber, trigger_timestamp
                            )
                            new_data_framenumbers.clear()

                        self.last_timestamps[cam_idx] = trigger_timestamp
                        self.last_framenumbers_delay[cam_idx] = raw_framenumber
                        self.main_brain.framenumber = corrected_framenumber

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
                                deferred_2d_data.append(
                                    (
                                        absolute_cam_no,  # defer saving to later
                                        corrected_framenumber,
                                        trigger_timestamp,
                                        camn_received_time,
                                    )
                                    + point_tuple[:5]
                                    + (frame_pt_idx, cur_val, mean_val, sumsqf_val)
                                )
                        # save new frame data

                        if corrected_framenumber not in realtime_coord_dict:
                            realtime_coord_dict[corrected_framenumber] = {}
                            timestamp_check_dict[corrected_framenumber] = {}

                        # For hypothesis testing: attempt 3D reconstruction of 1st point from each 2D view
                        realtime_coord_dict[corrected_framenumber][
                            cam_id
                        ] = points_undistorted[0]
                        # timestamp_check_dict[corrected_framenumber][cam_id]= camn_received_time
                        timestamp_check_dict[corrected_framenumber][
                            cam_id
                        ] = trigger_timestamp

                        if len(points_in_pluecker_coords_meters):
                            # save all 3D Pluecker coordinates for Kalman filtering
                            realtime_kalman_coord_dict[corrected_framenumber][
                                absolute_cam_no
                            ] = points_in_pluecker_coords_meters

                        if n_pts:
                            inc_val = 1
                        else:
                            inc_val = 0

                        if (
                            corrected_framenumber
                            in oldest_timestamp_by_corrected_framenumber
                        ):
                            (
                                orig_timestamp,
                                n,
                            ) = oldest_timestamp_by_corrected_framenumber[
                                corrected_framenumber
                            ]
                            if orig_timestamp is None:
                                oldest = trigger_timestamp  # this may also be None, but eventually won't be
                            else:
                                oldest = min(trigger_timestamp, orig_timestamp)
                            oldest_timestamp_by_corrected_framenumber[
                                corrected_framenumber
                            ] = (oldest, n + inc_val)
                            del oldest, n, orig_timestamp
                        else:
                            oldest_timestamp_by_corrected_framenumber[
                                corrected_framenumber
                            ] = (trigger_timestamp, inc_val)

                        new_data_framenumbers.add(
                            corrected_framenumber
                        )  # insert into set

                finished_corrected_framenumbers = []  # for quick deletion

                ########################################################################

                # Now we've grabbed all data waiting on network. Now it's
                # time to calculate 3D info.

                # XXX could go for latest data first to minimize latency
                # on that data.

                ########################################################################

                for corrected_framenumber in new_data_framenumbers:
                    (
                        oldest_camera_timestamp,
                        n,
                    ) = oldest_timestamp_by_corrected_framenumber[corrected_framenumber]
                    if oldest_camera_timestamp is None:
                        # LOG.info('no latency estimate available -- skipping 3D reconstruction')
                        continue
                    if (
                        time.time() - oldest_camera_timestamp
                    ) > self.max_reconstruction_latency_sec:
                        # LOG.info('maximum reconstruction latency exceeded -- skipping 3D reconstruction')
                        continue

                    data_dict = realtime_coord_dict[corrected_framenumber]
                    if len(data_dict) == len(self.cam_ids):  # all camera data arrived

                        if self.debug_level.isSet():
                            LOG.debug("frame %d" % (corrected_framenumber))

                        # mark for deletion out of data queue
                        finished_corrected_framenumbers.append(corrected_framenumber)

                        if self.reconstructor is None:
                            # can't do any 3D math without calibration information
                            self.main_brain.best_realtime_data = None
                            continue

                        if not self.main_brain.trigger_device.have_estimate():
                            # acquire_stamp (the proximate error, however latency estimates for the same reason)
                            # cannot be calculated unless the triggerbox has a clock model
                            continue

                        if 1:
                            with self.tracker_lock:
                                if (
                                    self.tracker is None
                                ):  # tracker isn't instantiated yet...
                                    self.main_brain.best_realtime_data = None
                                    continue

                                pluecker_coords_by_camn = realtime_kalman_coord_dict[
                                    corrected_framenumber
                                ]

                                if self.save_profiling_data:
                                    dumps = pickle.dumps(pluecker_coords_by_camn)
                                    self.data_dict_queue.append(
                                        (
                                            "gob",
                                            (
                                                corrected_framenumber,
                                                dumps,
                                                self.camn2cam_id,
                                            ),
                                        )
                                    )
                                pluecker_coords_by_camn = self.tracker.calculate_a_posteriori_estimates(
                                    corrected_framenumber,
                                    pluecker_coords_by_camn,
                                    self.camn2cam_id,
                                    debug2=self.debug_level.isSet(),
                                )

                                if self.debug_level.isSet():
                                    LOG.debug(
                                        "%d live objects:"
                                        % self.tracker.how_many_are_living()
                                    )
                                    results = self.tracker.get_most_recent_data()
                                    for result in results:
                                        if result is None:
                                            continue
                                        obj_id, last_xhat, P = result
                                        Pmean = numpy.sqrt(
                                            numpy.sum([P[i, i] ** 2 for i in range(3)])
                                        )
                                        LOG.debug(
                                            "obj_id %d: (%.3f, %.3f, %.3f), Pmean: %.3f"
                                            % (
                                                obj_id,
                                                last_xhat[0],
                                                last_xhat[1],
                                                last_xhat[2],
                                                Pmean,
                                            )
                                        )

                                if self.save_profiling_data:
                                    self.data_dict_queue.append(
                                        ("ntrack", self.tracker.how_many_are_living())
                                    )

                                now = time.time()
                                if self.show_overall_latency.isSet():
                                    (
                                        oldest_camera_timestamp,
                                        n,
                                    ) = oldest_timestamp_by_corrected_framenumber[
                                        corrected_framenumber
                                    ]
                                    if n > 0:
                                        if 0:
                                            LOG.info(
                                                "overall latency %d: %.1f msec (oldest: %s now: %s)"
                                                % (
                                                    n,
                                                    (now - oldest_camera_timestamp)
                                                    * 1e3,
                                                    repr(oldest_camera_timestamp),
                                                    repr(now),
                                                )
                                            )
                                        else:

                                            LOG.info(
                                                "overall latency (%d camera detected 2d points): %.1f msec (note: may exclude camera->camera computer latency)"
                                                % (
                                                    n,
                                                    (now - oldest_camera_timestamp)
                                                    * 1e3,
                                                )
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

                                    results = self.tracker.get_most_recent_data()
                                    Xs = []
                                    for result in results:
                                        if result is None:
                                            continue
                                        obj_id, last_xhat, P = result
                                        X = last_xhat[0], last_xhat[1], last_xhat[2]
                                        Xs.append(X)
                                    if len(Xs):
                                        self.main_brain.best_realtime_data = Xs, 0.0
                                    else:
                                        self.main_brain.best_realtime_data = None

                                # Convert to format accepted by find_best_3d()
                                found_data_dict, first_idx_by_cam_id = convert_format(
                                    pluecker_coords_by_camn,
                                    self.camn2cam_id,
                                    area_threshold=0.0,
                                    only_likely=True,
                                )

                                if len(found_data_dict) >= 2:
                                    # Can't do any 3D math without at least 2 cameras giving good
                                    # data.
                                    max_error = self.tracker.kalman_model[
                                        "hypothesis_test_max_acceptable_error"
                                    ]
                                    with_water = self.reconstructor.wateri is not None

                                    try:
                                        (
                                            this_observation_3d,
                                            this_observation_Lcoords,
                                            cam_ids_used,
                                            min_mean_dist,
                                        ) = ru.hypothesis_testing_algorithm__find_best_3d(
                                            self.reconstructor,
                                            found_data_dict,
                                            max_error,
                                            max_n_cams=self.max_N_hypothesis_test,
                                            with_water=with_water,
                                        )
                                    except ru.NoAcceptablePointFound:
                                        pass
                                    else:
                                        this_observation_camns = [
                                            self.cam_id2cam_no[cam_id]
                                            for cam_id in cam_ids_used
                                        ]
                                        this_observation_idxs = [
                                            first_idx_by_cam_id[cam_id]
                                            for cam_id in cam_ids_used
                                        ]
                                        ####################################
                                        #  Now join found point into Tracker
                                        if self.save_profiling_data:
                                            self.data_dict_queue.append(
                                                (
                                                    "join",
                                                    (
                                                        corrected_framenumber,
                                                        this_observation_3d,
                                                        this_observation_Lcoords,
                                                        this_observation_camns,
                                                        this_observation_idxs,
                                                    ),
                                                )
                                            )
                                        # test for novelty
                                        believably_new = self.tracker.is_believably_new(
                                            this_observation_3d
                                        )
                                        if believably_new:
                                            self.tracker.join_new_obj(
                                                corrected_framenumber,
                                                this_observation_3d,
                                                this_observation_Lcoords,
                                                this_observation_camns,
                                                this_observation_idxs,
                                            )
                                if 1:
                                    if self.tracker.how_many_are_living():
                                        # publish state to ROS
                                        results = self.tracker.get_most_recent_data()
                                        ros_objects = []
                                        for result in results:
                                            if result is None:
                                                continue
                                            obj_id, xhat, P = result
                                            this_ros_object = flydra_object(
                                                obj_id=obj_id,
                                                position=Point(*xhat[:3]),
                                                velocity=Vector3(*xhat[3:6]),
                                                posvel_covariance_diagonal=numpy.diag(
                                                    P
                                                )[:6].tolist(),
                                            )
                                            ros_objects.append(this_ros_object)
                                        ros_packet = flydra_mainbrain_packet(
                                            framenumber=corrected_framenumber,
                                            reconstruction_stamp=rospy.Time.from_sec(
                                                now
                                            ),
                                            acquire_stamp=rospy.Time.from_sec(
                                                oldest_camera_timestamp
                                            ),
                                            objects=ros_objects,
                                        )
                                        self.realtime_ros_packets_pub.publish(
                                            flydra_mainbrain_super_packet([ros_packet])
                                        )

                for finished in finished_corrected_framenumbers:
                    # check that timestamps are in reasonable agreement (low priority)
                    diff_from_start = []
                    for cam_id, tmp_trigger_timestamp in timestamp_check_dict[
                        finished
                    ].iteritems():
                        diff_from_start.append(tmp_trigger_timestamp)
                    timestamps_by_cam_id = numpy.array(diff_from_start)

                    if self.show_sync_errors:
                        if len(timestamps_by_cam_id):
                            if (
                                numpy.max(
                                    abs(timestamps_by_cam_id - timestamps_by_cam_id[0])
                                )
                                > 0.005
                            ):
                                LOG.warn(
                                    "timestamps off by more than 5 msec -- synchronization error"
                                )
                                self.main_brain.error_ros_msgs_pub.publish(
                                    FlydraError(FlydraError.CAM_TIMESTAMPS_OFF, "")
                                )

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
                if len(realtime_coord_dict) > 100:
                    # dont spam the console at startup (i.e. before a sync has been attemted)
                    if self.ever_synchronized:
                        LOG.warn(
                            "Cameras not synchronized or network dropping packets -- unmatched 2D data accumulating"
                        )
                        self.main_brain.error_ros_msgs_pub.publish(
                            FlydraError(FlydraError.NOT_SYNCHRONIZED, "")
                        )

                    k = realtime_coord_dict.keys()
                    k.sort()

                    # get one sample
                    corrected_framenumber = k[0]
                    data_dict = realtime_coord_dict[corrected_framenumber]
                    this_cam_ids = data_dict.keys()
                    missing_cam_id_guess = list(set(self.cam_ids) - set(this_cam_ids))
                    if len(missing_cam_id_guess) and self.ever_synchronized:
                        delta = list(set(self.cam_ids) - set(this_cam_ids))
                        LOG.warn("a guess at missing cam_id(s): %r" % delta)
                        for d in delta:
                            self.main_brain.error_ros_msgs_pub.publish(
                                FlydraError(FlydraError.MISSING_DATA, d)
                            )

                    for ki in k[:-50]:
                        del realtime_coord_dict[ki]
                        del timestamp_check_dict[ki]

                if len(realtime_kalman_coord_dict) > 100:
                    LOG.warn(
                        "deleting unused 3D data (this should be a rare occurrance)"
                    )
                    self.main_brain.error_ros_msgs_pub.publish(
                        FlydraError(FlydraError.UNUSED_3D_DATA, "")
                    )
                    k = realtime_kalman_coord_dict.keys()
                    k.sort()
                    for ki in k[:-50]:
                        del realtime_kalman_coord_dict[ki]

                if len(oldest_timestamp_by_corrected_framenumber) > 100:
                    k = oldest_timestamp_by_corrected_framenumber.keys()
                    k.sort()
                    for ki in k[:-50]:
                        del oldest_timestamp_by_corrected_framenumber[ki]

                if len(deferred_2d_data):
                    self.main_brain.queue_data2d.put(deferred_2d_data)
