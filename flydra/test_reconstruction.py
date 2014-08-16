import time, sys
import socket, struct
import Queue, threading
import tempfile, os
import pprint

import numpy as np

from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem

import flydra.offline_data_save
from flydra.kalman.kalmanize import kalmanize
from flydra.a2.calculate_reprojection_errors import calculate_reprojection_errors
import flydra.water as water
import flydra.a2.core_analysis as core_analysis
from flydra.reconstruct import Reconstructor
from flydra.main_brain.coordinate_receiver import CoordinateProcessor

MB_HOSTNAME = 'localhost'
CAM_HOSTNAME = 'localhost'
SPINUP_DURATION = 0.2
MAX_MEAN_ERROR = 0.002

def setup_data(with_water=False, fps=120.0, with_orientation=False, with_distortion=True):
    # generate fake trajectory
    dt = 1/fps
    t = np.arange(0.0, 1.0, dt)

    x = 0.2*np.cos(t*0.9)
    y = 0.3*np.sin(t*0.7)
    z = 0.1*np.sin(t*0.13) - 0.12

    pts = np.hstack( (x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis] ) )

    base = CameraModel.load_camera_default()

    lookat = np.array( (0.0, 0.0, 0.0) )
    up = np.array( (0.0, 0.0, 1.0) )

    cams = []
    cams.append(  base.get_view_camera(eye=np.array((1.0,0.0,1.0)),lookat=lookat,up=up) )
    cams.append(  base.get_view_camera(eye=np.array((1.2,3.4,5.6)),lookat=lookat,up=up) )
    cams.append(  base.get_view_camera(eye=np.array((0,0.3,1.0)),lookat=lookat,up=up) )

    if with_distortion:
        distortion1 = np.array( [0.2, 0.3, 0.1, 0.1, 0.1] )
    else:
        distortion1 = np.zeros((5,))
    cam_wide = CameraModel.load_camera_simple(name='cam_wide',
                                                    fov_x_degrees=90,
                                                    eye=np.array((-1.0,-1.0,0.7)),
                                                    lookat=lookat,
                                                    distortion_coefficients=distortion1,
                                                    )
    cams.append(cam_wide)

    for i in range(len(cams)):
        cams[i].name = 'cam%02d'%i

    cam_system = MultiCameraSystem(cams)

    # ------------
    # calculate 2d points for each camera
    reconstructor = Reconstructor.from_pymvg(cam_system)
    if with_water:
        wateri = water.WaterInterface(refractive_index=1.3330,
                                      water_roots_eps=1e-7)
        reconstructor.add_water(wateri)

    data2d = {'2d_pos_by_cam_ids':{},
              '2d_slope_by_cam_ids':{},
              }
    for camn,cam in enumerate(cams):
        cam_id = cam.name
        assert cam_id!='t'

        if with_water:
            center_2d = water.view_points_in_water( reconstructor,
                                                    cam_id, pts, wateri).T
        else:
            center_2d = cam.project_3d_to_pixel(pts)

        data2d['2d_pos_by_cam_ids'][cam_id] = center_2d

        if with_orientation:
            dx = np.gradient( center_2d[:,0] )
            dy = np.gradient( center_2d[:,1] )
            slope = np.arctan2( dy, dx )
            data2d['2d_slope_by_cam_ids'][cam_id] = slope
        else:
            data2d['2d_slope_by_cam_ids'][cam_id] = np.zeros( (len(data2d['2d_pos_by_cam_ids'][cam_id]), ))

    if with_orientation:
        eccentricity=20.0
    else:
        eccentricity=0.0

    data2d['t'] = t
    result = dict(data2d=data2d,
                  reconstructor=reconstructor,
                  dynamic_model_name='EKF mamarama, units: mm',
                  x=x,
                  y=y,
                  z=z,
                  eccentricity=eccentricity,
                  )
    return result

def test_offline_reconstruction():
    fps=120.0
    for use_kalman_smoothing in [False, True]:
        for with_orientation in [False, True]:
            for with_water in [False, True]:
                for with_distortion in [False,True]:
                    yield check_offline_reconstruction, with_water, use_kalman_smoothing, with_orientation, fps, with_distortion

def check_offline_reconstruction(with_water=False,
                                 use_kalman_smoothing=False,
                                 with_orientation=False,
                                 fps=120.0,
                                 with_distortion=True):
    D = setup_data( fps=fps,
                    with_water=with_water,
                    with_orientation=with_orientation,
                    with_distortion=with_distortion,
                    )

    data2d_fname = tempfile.mktemp(suffix='-data2d.h5')
    to_unlink = [data2d_fname]
    try:
        flydra.offline_data_save.save_data( fname=data2d_fname,
                                            data2d=D['data2d'],
                                            fps=fps,
                                            reconstructor=D['reconstructor'],
                                            eccentricity=D['eccentricity'],
                                            )

        data3d_fname = tempfile.mktemp(suffix='-data3d.h5')
        kalmanize(data2d_fname,
                  dest_filename = data3d_fname,
                  dynamic_model_name = D['dynamic_model_name'],
                  )
        to_unlink.append(data3d_fname)

        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file,
         extra) = ca.initial_file_load(data3d_fname)

        assert len(use_obj_ids)==1
        obj_id = use_obj_ids[0]

        load_model = D['dynamic_model_name']

        if use_kalman_smoothing:
            if load_model.startswith('EKF '):
                load_model = load_model[4:]
            smoothcache_fname = os.path.splitext(data3d_fname)[0]+'.kh5-smoothcache'
            to_unlink.append(smoothcache_fname)

        my_rows = ca.load_data(
            obj_id, data_file,
            use_kalman_smoothing=use_kalman_smoothing,
            dynamic_model_name = load_model,
            frames_per_second=fps,
            )

        x_actual = my_rows['x']
        y_actual = my_rows['y']
        z_actual = my_rows['z']

        data_file.close()
        ca.close()

    finally:
        for fname in to_unlink:
            try:
                os.unlink(fname)
            except OSError as err:
                # file does not exist?
                pass

    mean_error = np.mean(np.sqrt((D['x']-x_actual)**2 +
                                 (D['y']-y_actual)**2 +
                                 (D['z']-z_actual)**2))

    # We should have very low error
    fudge = 2 if use_kalman_smoothing else 1
    assert mean_error < fudge*MAX_MEAN_ERROR

class FakeTriggerDevice:
    def __init__(self,time_lock,time_dict):
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

class FakeRemoteApi():
    def external_request_image_async(self,cam_id):
        pass

class FakeMainBrain:
    def __init__(self,trigger_device):
        self.hostname = MB_HOSTNAME
        self.queue_error_ros_msgs = Queue.Queue()
        self.trigger_device = trigger_device
        self.remote_api = FakeRemoteApi()
        self.counts = {}
    def is_saving_data(self):
        return False

def test_online_reconstruction():
    for with_water in [False, True]:
        for with_orientation in [False]:#,True]:
            yield check_online_reconstruction, with_water, with_orientation

def check_online_reconstruction(with_water=False,
                                with_orientation=False,
                                fps=120.0,
                                multithreaded=True,
                                with_distortion=True,
                                ):
    D = setup_data( fps=fps,
                    with_water=with_water,
                    with_orientation=with_orientation,
                    with_distortion=with_distortion,
                    )

    time_lock = threading.Lock()
    time_dict = {}
    trigger_device = FakeTriggerDevice(time_lock=time_lock, time_dict=time_dict)
    mb = FakeMainBrain(trigger_device=trigger_device)
    debug_level=threading.Event()
    #debug_level.set()
    show_overall_latency=threading.Event()
    #show_overall_latency.set()
    coord_processor = CoordinateProcessor(mb,
                                          save_profiling_data=False,
                                          debug_level=debug_level,
                                          show_overall_latency=show_overall_latency,
                                          show_sync_errors=False,
                                          max_reconstruction_latency_sec=0.3,
                                          max_N_hypothesis_test=3,
                                          hostname=MB_HOSTNAME,
                                          )
    if multithreaded:
        coord_processor.daemon = True
        coord_processor.start()

    # quit the coordinate sender thread so we can intercept its queue
    coord_processor.tp._quit_event = threading.Event()
    coord_processor.tp._quit_event.set()
    coord_processor.tp._queue.put('junk') # allow blocking call to finish
    coord_processor.tp.join()

    ports = {}
    R = D['reconstructor']
    for cam_id in R.cam_ids:
        port = coord_processor.connect(cam_id,CAM_HOSTNAME)
        ports[cam_id] = port

    coord_processor.set_reconstructor(R)
    model = flydra.kalman.dynamic_models.get_kalman_model(name=D['dynamic_model_name'],dt=(1.0/fps))
    coord_processor.set_new_tracker(model)
    if multithreaded:
        # XXX remove this in the future

        # monkeypatch to allow hacky calling of .run() repeatedly
        def no_op():
            pass
        orig_kill_all_trackers = coord_processor.tracker.kill_all_trackers
        coord_processor.tracker.kill_all_trackers = no_op

    header_fmt = flydra.common_variables.recv_pt_header_fmt
    pt_fmt = flydra.common_variables.recv_pt_fmt

    data2d=D['data2d']
    orig_timestamps = data2d.pop('t')

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    area = 1.0
    if D['eccentricity']:
        line_found = True
        slope_found = True
    else:
        line_found = False
        slope_found = False
    ray_valid = True
    cur_val, mean_val, sumsqf_val = (100.0, 2.0, 3.0)

    centers = {}
    sccs = {}
    for cam_id in data2d['2d_pos_by_cam_ids']:
        sccs[cam_id] = R.get_SingleCameraCalibration(cam_id)
        cc = R.get_camera_center(cam_id)[:,0]
        cc = np.array([cc[0],cc[1],cc[2],1.0])
        centers[cam_id] = cc

    dt = 1.0/fps
    time.sleep(SPINUP_DURATION)

    errors = []
    num_sync_frames = 1
    obj_id = None
    for framenumber, orig_timestamp in enumerate(orig_timestamps):
        # frame 0 - first 2D coordinates and synchronization
        # frame 1 - first saveable data

        timestamp = time.time()
        with time_lock:
            time_dict[framenumber]=timestamp

        for cam_id in data2d['2d_pos_by_cam_ids']:
            scc = sccs[cam_id]
            camn_received_time = timestamp
            pt_x,pt_y = data2d['2d_pos_by_cam_ids'][cam_id][framenumber]
            slope = data2d['2d_slope_by_cam_ids'][cam_id][framenumber]

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
            header = (cam_id,timestamp, camn_received_time, framenumber,
                      n_pts,n_frames_skipped)
            header_buf = struct.pack(header_fmt,*header)
            if n_pts:
                assert n_pts==1
                x_undistorted,y_undistorted = R.undistort(cam_id,(pt_x,pt_y))

                (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4,
                 ray5) = flydra.reconstruct.do_3d_operations_on_2d_point(
                    scc.helper,x_undistorted,y_undistorted,
                    scc.pmat_inv,
                    cc,
                    pt_x,pt_y,
                    rise, run)

                pt = (pt_x,pt_y,area,slope,D['eccentricity'],
                      p1,p2,p3,p4,line_found,slope_found,
                      x_undistorted,y_undistorted,
                      ray_valid,
                      ray0, ray1, ray2, ray3, ray4, ray5, # pluecker coords from cam center to detected point
                      cur_val, mean_val, sumsqf_val,
                      )
                pt_buf = struct.pack(pt_fmt,*pt)
            else:
                pt_buf = ''
            buf = header_buf + pt_buf

            if multithreaded:
                port = ports[cam_id]
                sender.sendto(buf,(MB_HOSTNAME,port))
            else:
                coord_processor.process_data(buf)

        if framenumber < num_sync_frames:
            # Before sync, we may not get data or it may be wrong, so
            # ignore it. But wait dt seconds to ensure sychronization
            # has enough time to run.
            try:
                coord_processor.queue_realtime_ros_packets.get(True,dt)
            except Queue.Empty:
                pass

        else:
            next = coord_processor.queue_realtime_ros_packets.get()

            assert len(next.objects)==1
            o1 = next.objects[0]
            if obj_id is not None:
                assert o1.obj_id==obj_id, 'object id changed'
            else:
                obj_id = o1.obj_id
            actual = o1.position.x, o1.position.y, o1.position.z
            expected = np.array([D[dim][framenumber] for dim in 'xyz'])
            errors.append( np.sqrt(np.sum((expected-actual)**2)) )

        if multithreaded:
            if not coord_processor.is_alive():
                break


    if multithreaded:
        orig_kill_all_trackers() # remove this in the future...

    if not multithreaded:
        coord_processor.finish_processing()

    t_start = time_dict[num_sync_frames]
    t_stop = time_dict[framenumber]
    dur = t_stop-t_start
    n_frames = framenumber-num_sync_frames
    fps = n_frames/(t_stop-t_start)

    if multithreaded:
        coord_processor.quit()
        coord_processor.join()
    if not coord_processor.did_quit_successfully:
        raise RuntimeError('coordinate processor thread had error')

    mean_error = np.mean(errors)
    assert len(errors)+num_sync_frames == len(orig_timestamps)

    # We should have very low error
    assert mean_error < MAX_MEAN_ERROR
    return {'fps':fps}

def benchmark():
    rd = check_online_reconstruction(with_water=False,
                                     with_orientation=False,
                                     multithreaded=False)
    pprint.pprint(rd)

if __name__=='__main__':
    if len(sys.argv) == 2:
        kcachegrind_output_fname = sys.argv[1]
    else:
        kcachegrind_output_fname = None
    if kcachegrind_output_fname is not None:
        import cProfile
        import lsprofcalltree
        p = cProfile.Profile()
        print 'running test in profile mode'
        p.runctx('benchmark()',globals(),locals())
        k = lsprofcalltree.KCacheGrind(p)
        data = open(kcachegrind_output_fname, 'w')
        k.output(data)
        data.close()
    else:
        benchmark()
