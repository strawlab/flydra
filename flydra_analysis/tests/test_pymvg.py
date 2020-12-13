from __future__ import print_function
import os
import flydra_core.reconstruct as reconstruct
import pkg_resources
import numpy as np
import pymvg
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem

sample_cal = pkg_resources.resource_filename('flydra_analysis.a2',
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

def test_pymvg_roundtrip():
    from pymvg.camera_model import CameraModel
    from pymvg.multi_camera_system import MultiCameraSystem
    from flydra_core.reconstruct import Reconstructor

    # ----------- with no distortion ------------------------
    center1 = np.array( (0, 0.0, 5) )
    center2 = np.array( (1, 0.0, 5) )

    lookat = np.array( (0,1,0))

    cam1 = CameraModel.load_camera_simple(fov_x_degrees=90,
                                          name='cam1',
                                          eye=center1,
                                          lookat=lookat)
    cam2 = CameraModel.load_camera_simple(fov_x_degrees=90,
                                          name='cam2',
                                          eye=center2,
                                          lookat=lookat)
    mvg = MultiCameraSystem( cameras=[cam1, cam2] )
    R = Reconstructor.from_pymvg(mvg)
    mvg2 = R.convert_to_pymvg()

    cam_ids = ['cam1','cam2']
    for distorted in [True,False]:
        for cam_id in cam_ids:
            v1 = mvg.find2d(  cam_id, lookat, distorted=distorted )
            v2 = R.find2d(    cam_id, lookat, distorted=distorted )
            v3 = mvg2.find2d( cam_id, lookat, distorted=distorted )
            assert np.allclose(v1,v2)
            assert np.allclose(v1,v3)

    # ----------- with distortion ------------------------
    cam1dd = cam1.to_dict()
    cam1dd['D'] = (0.1, 0.2, 0.3, 0.4, 0.0)
    cam1d = CameraModel.from_dict(cam1dd)

    cam2dd = cam2.to_dict()
    cam2dd['D'] = (0.11, 0.21, 0.31, 0.41, 0.0)
    cam2d = CameraModel.from_dict(cam2dd)

    mvgd = MultiCameraSystem( cameras=[cam1d, cam2d] )
    Rd = Reconstructor.from_pymvg(mvgd)
    mvg2d = Rd.convert_to_pymvg()
    cam_ids = ['cam1','cam2']
    for distorted in [True,False]:
        for cam_id in cam_ids:
            v1 = mvgd.find2d(  cam_id, lookat, distorted=distorted )
            v2 = Rd.find2d(    cam_id, lookat, distorted=distorted )
            v3 = mvg2d.find2d( cam_id, lookat, distorted=distorted )
            assert np.allclose(v1,v2)
            assert np.allclose(v1,v3)

    # ------------ with distortion at different focal length ------
    mydir = os.path.dirname(__file__)
    caldir = os.path.join(mydir,'sample_calibration')
    print(mydir)
    print(caldir)
    R3 = Reconstructor(caldir)
    mvg3 = R3.convert_to_pymvg()
    #R4 = Reconstructor.from_pymvg(mvg3)
    mvg3b = MultiCameraSystem.from_mcsc( caldir )

    for distorted in [True,False]:
        for cam_id in R3.cam_ids:
            v1 = R3.find2d(   cam_id, lookat, distorted=distorted )
            v2 = mvg3.find2d( cam_id, lookat, distorted=distorted )
            #v3 = R4.find2d(   cam_id, lookat, distorted=distorted )
            v4 = mvg3b.find2d( cam_id, lookat, distorted=distorted )
            assert np.allclose(v1,v2)
            #assert np.allclose(v1,v3)
            assert np.allclose(v1,v4)

if __name__=='__main__':
    test_pymvg()
    test_distortion()
    test_pymvg_roundtrip()
