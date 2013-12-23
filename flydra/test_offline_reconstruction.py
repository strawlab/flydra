import numpy as np
import pymvg
import tempfile, os
import flydra.offline_data_save
from flydra.kalman.kalmanize import kalmanize
from flydra.a2.calculate_reprojection_errors import calculate_reprojection_errors
import pandas
import flydra.a2.core_analysis as core_analysis
from flydra.reconstruct import Reconstructor

def test_reconstruction():
    for use_kalman_smoothing in [False, True]:
        for with_water in [False]:
        #for with_water in [False, True]:
            yield check_reconstruction, with_water, use_kalman_smoothing

def check_reconstruction(with_water=False, use_kalman_smoothing=False, duration=1.0, fps=120.0):
    # generate fake trajectory
    dynamic_model_name = 'EKF mamarama, units: mm'
    dt = 1/fps
    t = np.arange(0.0, duration, dt)

    x = 0.2*np.cos(t*0.9)
    y = 0.3*np.sin(t*0.7)
    z = 0.1*np.sin(t*0.13) - 0.12

    pts = np.hstack( (x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis] ) )

    base = pymvg.CameraModel.load_camera_default()

    lookat = np.array( (0.0, 0.0, 0.0) )
    up = np.array( (0.0, 0.0, 1.0) )

    cams = []
    cams.append(  base.get_view_camera(eye=np.array((1.0,0.0,1.0)),lookat=lookat,up=up) )
    cams.append(  base.get_view_camera(eye=np.array((1.0,1.0,0.5)),lookat=lookat,up=up) )
    cams.append(  base.get_view_camera(eye=np.array((-1.0,-1.0,0.3)),lookat=lookat,up=up) )

    for i in range(len(cams)):
        cams[i].name = 'cam%02d'%i

    cam_system = pymvg.MultiCameraSystem(cams)

    # ------------
    # calculate 2d points for each camera
    reconstructor = Reconstructor.from_pymvg(cam_system)
    data2d = {}
    for camn,cam in enumerate(cams):
        cam_id = cam.name
        assert cam_id!='t'

        data2d[cam_id] = cam.project_3d_to_pixel(pts, distorted=True)

    data2d['t'] = t

    data2d_fname = tempfile.mktemp(suffix='-data2d.h5')
    to_unlink = [data2d_fname]
    try:
        flydra.offline_data_save.save_data( fname=data2d_fname,
                                            data2d=data2d,
                                            fps=fps,
                                            reconstructor=reconstructor,
                                            )

        data3d_fname = tempfile.mktemp(suffix='-data3d.h5')
        kalmanize(data2d_fname,
                  dest_filename = data3d_fname,
                  dynamic_model_name = dynamic_model_name,
                  )
        to_unlink.append(data3d_fname)

        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file,
         extra) = ca.initial_file_load(data3d_fname)

        assert len(use_obj_ids)==1
        obj_id = use_obj_ids[0]

        load_model = dynamic_model_name

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

    mean_error = np.mean(np.sqrt((x-x_actual)**2 +
                                 (y-y_actual)**2 +
                                 (z-z_actual)**2))

    # We should have very low error
    assert mean_error < 0.02, ('mean error was %.3f, '
                              'but should have been < 0.02.'%(
        mean_error,))
