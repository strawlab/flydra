import flydra.reconstruct as reconstruct
import pkg_resources
import numpy as np
import pymvg
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem

sample_cal = pkg_resources.resource_filename('flydra.a2',
                                             'sample_calibration.xml')

def test_pymvg():
    R1 = reconstruct.Reconstructor(sample_cal)
    R2 = R1.convert_to_pymvg()

    pts_3d = [ (0.0, 0.0, 0.0),
               (0.1, 0.0, 0.0),
               (0.0, 0.1, 0.0),
               (0.0, 0.0, 0.1),
               (0.1, 0.1, 0.0),
               (0.1, 0.0, 0.1),
               (0.0, 0.1, 0.1),
               ]
    pts_3d = np.array(pts_3d)

    # make homogeneous coords
    pts_3d_h = np.ones( (len(pts_3d),4) )
    pts_3d_h[:, :3] = pts_3d

    # check 3D->2D projection
    for cam_id in R1.get_cam_ids():
        fpix = R1.find2d( cam_id, pts_3d_h )
        cpix = R2.find2d( cam_id, pts_3d )

        assert np.allclose( fpix, cpix)

    # check 2D->3D triangulation
    for X in pts_3d:
        inputs = []
        for cam_id in R1.get_cam_ids():
            inputs.append( (cam_id, R1.find2d(cam_id,X)) )

        tri1 = R1.find3d( inputs, return_line_coords=False )
        tri2 = R2.find3d( inputs )
        assert np.allclose(tri1, tri2)

def test_distortion():
    base = CameraModel.load_camera_default()
    lookat = np.array( (0.0, 0.0, 0.0) )
    up = np.array( (0.0, 0.0, 1.0) )

    cams = []
    cams.append(  base.get_view_camera(eye=np.array((1.0,0.0,1.0)),lookat=lookat,up=up) )

    distortion1 = np.array( [0.2, 0.3, 0.1, 0.1, 0.1] )
    cam_wide = CameraModel.load_camera_simple(name='cam_wide',
                                              fov_x_degrees=90,
                                              eye=np.array((-1.0,-1.0,0.7)),
                                              lookat=lookat,
                                              distortion_coefficients=distortion1,
                                              )
    cams.append(cam_wide)

    cam_ids = []
    for i in range(len(cams)):
        cams[i].name = 'cam%02d'%i
        cam_ids.append(cams[i].name)

    cam_system = MultiCameraSystem(cams)
    R = reconstruct.Reconstructor.from_pymvg(cam_system)
    for cam_id in cam_ids:
        nl_params = R.get_intrinsic_nonlinear(cam_id)
        mvg_cam = cam_system.get_camera_dict()[cam_id]
        assert np.allclose(mvg_cam.distortion, nl_params)

if __name__=='__main__':
    test_pymvg()
    test_distortion()
