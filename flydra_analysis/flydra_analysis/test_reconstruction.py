from __future__ import print_function
import time, sys
import socket, struct
import threading
if sys.version_info.major <= 2:
    import Queue as queue
else:
    import queue
import tempfile, os
import pprint
import pkg_resources
import shutil

import numpy as np

from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem

import flydra_core.kalman.dynamic_models
import flydra_analysis.offline_data_save
from flydra_analysis.kalmanize import kalmanize
import flydra_core.water as water
import flydra_analysis.a2.core_analysis as core_analysis
import flydra_core.flydra_socket as flydra_socket
from flydra_core.reconstruct import Reconstructor, DEFAULT_WATER_REFRACTIVE_INDEX
from flydra_analysis.a2.retrack_reuse_data_association import (
    retrack_reuse_data_association,
)

SPINUP_DURATION = 0.2
MAX_MEAN_ERROR = 0.002


def _get_cams(with_distortion):
    base = CameraModel.load_camera_default()

    lookat = np.array((0.0, 0.0, 0.0))
    up = np.array((0.0, 0.0, 1.0))

    cams = []
    cams.append(
        base.get_view_camera(eye=np.array((1.0, 0.0, 1.0)), lookat=lookat, up=up)
    )
    cams.append(
        base.get_view_camera(eye=np.array((1.2, 3.4, 5.6)), lookat=lookat, up=up)
    )
    cams.append(base.get_view_camera(eye=np.array((0, 0.3, 1.0)), lookat=lookat, up=up))

    if with_distortion:
        distortion1 = np.array([0.2, 0.3, 0.1, 0.1, 0.1])
    else:
        distortion1 = np.zeros((5,))
    cam_wide = CameraModel.load_camera_simple(
        name="cam_wide",
        fov_x_degrees=90,
        eye=np.array((-1.0, -1.0, 0.7)),
        lookat=lookat,
        distortion_coefficients=distortion1,
    )
    cams.append(cam_wide)

    for i in range(len(cams)):
        cams[i].name = "cam%02d" % i

    cam_system = MultiCameraSystem(cams)
    reconstructor = Reconstructor.from_pymvg(cam_system)
    result = dict(cams=cams, cam_system=cam_system, reconstructor=reconstructor,)
    return result


def setup_data(
    with_water=False, fps=120.0, with_orientation=False, with_distortion=True
):
    tmp = _get_cams(with_distortion=with_distortion)
    cams = tmp["cams"]
    cam_system = tmp["cam_system"]
    reconstructor = tmp["reconstructor"]

    # generate fake trajectory
    dt = 1 / fps
    t = np.arange(0.0, 1.0, dt)

    x = 0.2 * np.cos(t * 0.9)
    y = 0.3 * np.sin(t * 0.7)
    z = 0.1 * np.sin(t * 0.13) - 0.12

    pts = np.hstack((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]))

    # ------------
    # calculate 2d points for each camera
    if with_water:
        wateri = water.WaterInterface(
            refractive_index=DEFAULT_WATER_REFRACTIVE_INDEX, water_roots_eps=1e-7
        )
        reconstructor.add_water(wateri)

    data2d = {
        "2d_pos_by_cam_ids": {},
        "2d_slope_by_cam_ids": {},
    }
    for camn, cam in enumerate(cams):
        cam_id = cam.name
        assert cam_id != "t"

        if with_water:
            center_2d = water.view_points_in_water(reconstructor, cam_id, pts, wateri).T
        else:
            center_2d = cam.project_3d_to_pixel(pts)

        data2d["2d_pos_by_cam_ids"][cam_id] = center_2d

        if with_orientation:
            dx = np.gradient(center_2d[:, 0])
            dy = np.gradient(center_2d[:, 1])
            slope = np.arctan2(dy, dx)
            data2d["2d_slope_by_cam_ids"][cam_id] = slope
        else:
            data2d["2d_slope_by_cam_ids"][cam_id] = np.zeros(
                (len(data2d["2d_pos_by_cam_ids"][cam_id]),)
            )

    if with_orientation:
        eccentricity = 20.0
    else:
        eccentricity = 0.0

    data2d["t"] = t
    result = dict(
        data2d=data2d,
        reconstructor=reconstructor,
        dynamic_model_name="EKF mamarama, units: mm",
        x=x,
        y=y,
        z=z,
        eccentricity=eccentricity,
    )
    return result


def test_retracking_without_data_association():
    ca = core_analysis.get_global_CachingAnalyzer()

    orig_fname = pkg_resources.resource_filename(
        "flydra_analysis.a2", "sample_datafile-v0.4.28.h5"
    )

    tmpdir = tempfile.mkdtemp()
    try:
        retracked_fname = os.path.join(tmpdir, "retracked.h5")
        retrack_reuse_data_association(
            h5_filename=orig_fname,
            output_h5_filename=retracked_fname,
            kalman_filename=orig_fname,
        )
        with ca.kalman_analysis_context(orig_fname) as orig_h5_context:
            orig_obj_ids = orig_h5_context.get_unique_obj_ids()
            extra = orig_h5_context.get_extra_info()

            with ca.kalman_analysis_context(retracked_fname) as retracked_h5_context:
                retracked_obj_ids = retracked_h5_context.get_unique_obj_ids()

                assert len(retracked_obj_ids) > 10

                for obj_id in retracked_obj_ids[1:-1]:
                    # Cycle over retracked obj_ids, which may be subset of
                    # original (due to missing 2D data)

                    retracked_rows = retracked_h5_context.load_data(
                        obj_id,
                        use_kalman_smoothing=False,
                        dynamic_model_name=extra["dynamic_model_name"],
                        frames_per_second=extra["frames_per_second"],
                    )
                    orig_rows = orig_h5_context.load_data(
                        obj_id,
                        use_kalman_smoothing=False,
                        dynamic_model_name=extra["dynamic_model_name"],
                        frames_per_second=extra["frames_per_second"],
                    )
                    # They tracks start at the same frame...
                    assert retracked_rows["frame"][0] == orig_rows["frame"][0]
                    # and they should be no longer than the original.
                    assert len(retracked_rows) <= len(orig_rows)  # may be shorter?!
    finally:
        shutil.rmtree(tmpdir)


def test_find3d():
    fps = 120.0
    for use_kalman_smoothing in [False, True]:
        for with_orientation in [False]:
            for with_water in [False, True]:
                for with_distortion in [False, True]:
                    yield check_find3d, with_water, use_kalman_smoothing, with_orientation, fps, with_distortion


def check_find3d(
    with_water=False,
    use_kalman_smoothing=False,
    with_orientation=False,
    fps=120.0,
    with_distortion=True,
):
    assert with_orientation == False
    D = setup_data(
        fps=fps,
        with_water=with_water,
        with_orientation=with_orientation,
        with_distortion=with_distortion,
    )
    R = D["reconstructor"]
    pos2d = D["data2d"]["2d_pos_by_cam_ids"]
    rowidx = -1
    while 1:
        rowidx += 1
        cam_ids_and_points2d = []
        for cam_id in R.cam_ids:
            if rowidx >= len(pos2d[cam_id]):
                break
            points2d = pos2d[cam_id][rowidx]
            cam_ids_and_points2d.append((cam_id, points2d))
        if len(cam_ids_and_points2d) == 0:
            break
        X_actual = R.find3d(
            cam_ids_and_points2d, undistort=True, return_line_coords=False,
        )
        X_expected = np.array([D["x"][rowidx], D["y"][rowidx], D["z"][rowidx]])
        dist = np.sqrt(np.sum((X_actual - X_expected) ** 2))
        assert dist < 1e-5

    assert rowidx > 0  # make sure we did some tests


def test_offline_reconstruction():
    fps = 120.0
    for use_kalman_smoothing in [False, True]:
        for with_orientation in [False, True]:
            for with_water in [False, True]:
                for with_distortion in [False, True]:
                    yield check_offline_reconstruction, with_water, use_kalman_smoothing, with_orientation, fps, with_distortion


def check_offline_reconstruction(
    with_water=False,
    use_kalman_smoothing=False,
    with_orientation=False,
    fps=120.0,
    with_distortion=True,
):
    D = setup_data(
        fps=fps,
        with_water=with_water,
        with_orientation=with_orientation,
        with_distortion=with_distortion,
    )

    data2d_fname = tempfile.mktemp(suffix="-data2d.h5")
    to_unlink = [data2d_fname]
    try:
        flydra_analysis.offline_data_save.save_data(
            fname=data2d_fname,
            data2d=D["data2d"],
            fps=fps,
            reconstructor=D["reconstructor"],
            eccentricity=D["eccentricity"],
        )
        d1 = D["reconstructor"].get_intrinsic_nonlinear("cam03")
        d2 = flydra_core.reconstruct.Reconstructor(
            data2d_fname
        ).get_intrinsic_nonlinear("cam03")
        assert np.allclose(d1, d2)

        data3d_fname = tempfile.mktemp(suffix="-data3d.h5")
        kalmanize(
            data2d_fname,
            dest_filename=data3d_fname,
            dynamic_model_name=D["dynamic_model_name"],
            reconstructor=D["reconstructor"],
        )
        to_unlink.append(data3d_fname)

        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(
            data3d_fname
        )

        assert len(use_obj_ids) == 1
        obj_id = use_obj_ids[0]

        load_model = D["dynamic_model_name"]

        if use_kalman_smoothing:
            if load_model.startswith("EKF "):
                load_model = load_model[4:]
            smoothcache_fname = os.path.splitext(data3d_fname)[0] + ".kh5-smoothcache"
            to_unlink.append(smoothcache_fname)

        my_rows = ca.load_data(
            obj_id,
            data_file,
            use_kalman_smoothing=use_kalman_smoothing,
            dynamic_model_name=load_model,
            frames_per_second=fps,
        )

        x_actual = my_rows["x"]
        y_actual = my_rows["y"]
        z_actual = my_rows["z"]

        data_file.close()
        ca.close()

    finally:
        for fname in to_unlink:
            try:
                os.unlink(fname)
            except OSError as err:
                # file does not exist?
                pass

    assert my_rows["x"].shape == D["x"].shape
    mean_error = np.mean(
        np.sqrt(
            (D["x"] - x_actual) ** 2
            + (D["y"] - y_actual) ** 2
            + (D["z"] - z_actual) ** 2
        )
    )

    # We should have very low error
    fudge = 2 if use_kalman_smoothing else 1
    assert mean_error < fudge * MAX_MEAN_ERROR


class FakeTriggerDevice:
    def __init__(self, time_lock, time_dict):
        self.time_lock = time_lock
        self.time_dict = time_dict

    def wait_for_estimate(self):
        return

    def have_estimate(self):
        return True

    def framestamp2timestamp(self, corrected_framestamp):
        with self.time_lock:
            result = self.time_dict[corrected_framestamp]
        return result


class FakeRemoteApi:
    def external_request_image_async(self, cam_id):
        pass


class FakeMainBrain:
    def __init__(self, trigger_device):
        self.queue_error_ros_msgs = queue.Queue()
        self.trigger_device = trigger_device
        self.remote_api = FakeRemoteApi()
        self.counts = {}
        self._is_synchronizing = False

    def is_saving_data(self):
        return False


def disabled_tst_online_reconstruction():
    # This is currently disabled because it was never updated when we switched from
    # sending ROS messages from a separate thread to directly calling publish().
    for with_water in [False, True]:
        for with_orientation in [False, True]:
            for multithreaded in [True, False]:
                for skip_frames in [None, "some", "too many"]:
                    yield check_online_reconstruction, with_water, with_orientation, multithreaded, skip_frames


def check_online_reconstruction(
    with_water=False,
    with_orientation=False,
    multithreaded=True,
    skip_frames=None,
    fps=120.0,
    with_distortion=True,
):
    D = setup_data(
        fps=fps,
        with_water=with_water,
        with_orientation=with_orientation,
        with_distortion=with_distortion,
    )

    time_lock = threading.Lock()
    time_dict = {}
    trigger_device = FakeTriggerDevice(time_lock=time_lock, time_dict=time_dict)
    mb = FakeMainBrain(trigger_device=trigger_device)
    debug_level = threading.Event()
    # debug_level.set()
    show_overall_latency = threading.Event()
    # show_overall_latency.set()
    from flydra_core.coordinate_receiver import CoordinateProcessor

    coord_processor = CoordinateProcessor(
        mb,
        save_profiling_data=False,
        debug_level=debug_level,
        show_overall_latency=show_overall_latency,
        show_sync_errors=False,
        max_reconstruction_latency_sec=0.3,
        max_N_hypothesis_test=3,
        use_unix_domain_sockets=True,
    )
    if multithreaded:
        coord_processor.daemon = True
        coord_processor.start()

    # quit the coordinate sender thread so we can intercept its queue
    coord_processor.tp._quit_event = threading.Event()
    coord_processor.tp._quit_event.set()
    coord_processor.tp._queue.put("junk")  # allow blocking call to finish
    coord_processor.tp.join()

    R = D["reconstructor"]
    for cam_id in R.cam_ids:
        coord_processor.connect(cam_id)
    addr = coord_processor.get_listen_address()
    addrinfo = flydra_socket.make_addrinfo(**addr)
    sender = flydra_socket.FlydraTransportSender(addrinfo)

    coord_processor.set_reconstructor(R)
    model = flydra_core.kalman.dynamic_models.get_kalman_model(
        name=D["dynamic_model_name"], dt=(1.0 / fps)
    )
    coord_processor.set_new_tracker(model)

    max_frames_skipped = model["max_frames_skipped"]

    if multithreaded:
        # XXX remove this in the future

        # monkeypatch to allow hacky calling of .run() repeatedly
        def no_op():
            pass

        orig_kill_all_trackers = coord_processor.tracker.kill_all_trackers
        coord_processor.tracker.kill_all_trackers = no_op

    header_fmt = flydra_core.common_variables.recv_pt_header_fmt
    pt_fmt = flydra_core.common_variables.recv_pt_fmt

    data2d = D["data2d"]
    orig_timestamps = data2d.pop("t")

    area = 1.0
    if D["eccentricity"]:
        line_found = True
        slope_found = True
    else:
        line_found = False
        slope_found = False
    cur_val, mean_val, sumsqf_val = (100.0, 2.0, 3.0)

    centers = {}
    sccs = {}
    for cam_id in data2d["2d_pos_by_cam_ids"]:
        sccs[cam_id] = R.get_SingleCameraCalibration(cam_id)
        cc = R.get_camera_center(cam_id)[:, 0]
        cc = np.array([cc[0], cc[1], cc[2], 1.0])
        centers[cam_id] = cc

    dt = 1.0 / fps
    time.sleep(SPINUP_DURATION)

    n_skipped = 0
    if skip_frames is not None:
        start_skip_frame = 20
        if skip_frames == "some":
            n_skipped = max_frames_skipped - 1
        else:
            assert skip_frames == "too many"
            n_skipped = max_frames_skipped
        stop_skip_frame = start_skip_frame + n_skipped

    errors = []
    num_sync_frames = 1
    obj_id = None
    for framenumber, orig_timestamp in enumerate(orig_timestamps):
        # frame 0 - first 2D coordinates and synchronization
        # frame 1 - first saveable data

        if skip_frames is not None:
            if framenumber > start_skip_frame:
                if framenumber <= stop_skip_frame:
                    continue

        timestamp = time.time()
        with time_lock:
            time_dict[framenumber] = timestamp

        for cam_id in data2d["2d_pos_by_cam_ids"]:
            scc = sccs[cam_id]
            camn_received_time = timestamp
            pt_x, pt_y = data2d["2d_pos_by_cam_ids"][cam_id][framenumber]
            slope = data2d["2d_slope_by_cam_ids"][cam_id][framenumber]

            if np.isinf(slope):
                run = 0.0
                rise = 1.0
            else:
                run = 1.0
                rise = slope

            cc = centers[cam_id]
            if 0 <= pt_x < scc.res[0] and 0 <= pt_y < scc.res[1]:
                n_pts = 1
            else:
                n_pts = 0
            n_frames_skipped = 0
            header = (
                cam_id,
                timestamp,
                camn_received_time,
                framenumber,
                n_pts,
                n_frames_skipped,
            )
            header_buf = struct.pack(header_fmt, *header)
            if n_pts:
                assert n_pts == 1
                pt = (
                    pt_x,
                    pt_y,
                    area,
                    slope,
                    D["eccentricity"],
                    slope_found,
                    cur_val,
                    mean_val,
                    sumsqf_val,
                )
                pt_buf = struct.pack(pt_fmt, *pt)
            else:
                pt_buf = ""
            buf = header_buf + pt_buf

            if multithreaded:
                sender.send(buf)
            else:
                coord_processor.process_data(buf)

        if framenumber < num_sync_frames:
            # Before sync, we may not get data or it may be wrong, so
            # ignore it. But wait 10*dt seconds to ensure sychronization
            # has enough time to run.
            try:
                coord_processor.queue_realtime_ros_packets.get(True, 10 * dt)
            except queue.Empty:
                pass

        else:
            try:
                next = coord_processor.queue_realtime_ros_packets.get(True, 10 * dt)
            except queue.Empty:
                if skip_frames == "too many":
                    assert framenumber == (stop_skip_frame + 1)
                    # this is what we expect, the tracking should end
                    return
                else:
                    raise

            assert len(next.objects) == 1
            o1 = next.objects[0]
            if obj_id is not None:
                assert o1.obj_id == obj_id, "object id changed"
            else:
                obj_id = o1.obj_id
            actual = o1.position.x, o1.position.y, o1.position.z
            expected = np.array([D[dim][framenumber] for dim in "xyz"])
            errors.append(np.sqrt(np.sum((expected - actual) ** 2)))

        if multithreaded:
            if not coord_processor.is_alive():
                break

    if multithreaded:
        orig_kill_all_trackers()  # remove this in the future...

    if not multithreaded:
        coord_processor.finish_processing()

    t_start = time_dict[num_sync_frames]
    t_stop = time_dict[framenumber]
    dur = t_stop - t_start
    n_frames = framenumber - num_sync_frames
    fps = n_frames / (t_stop - t_start)

    if multithreaded:
        coord_processor.quit()
        coord_processor.join()
    if not coord_processor.did_quit_successfully:
        raise RuntimeError("coordinate processor thread had error")

    mean_error = np.mean(errors)
    assert len(errors) + num_sync_frames + n_skipped == len(orig_timestamps)

    # We should have very low error
    assert mean_error < MAX_MEAN_ERROR
    return {"fps": fps}


def benchmark():
    rd = check_online_reconstruction(
        with_water=False, with_orientation=False, multithreaded=False
    )
    pprint.pprint(rd)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        kcachegrind_output_fname = sys.argv[1]
    else:
        kcachegrind_output_fname = None
    if kcachegrind_output_fname is not None:
        import cProfile
        import lsprofcalltree

        p = cProfile.Profile()
        print("running test in profile mode")
        p.runctx("benchmark()", globals(), locals())
        k = lsprofcalltree.KCacheGrind(p)
        data = open(kcachegrind_output_fname, "w")
        k.output(data)
        data.close()
    else:
        benchmark()
