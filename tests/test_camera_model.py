import flydra.reconstruct as reconstruct
import pkg_resources
import numpy as np

sample_cal = pkg_resources.resource_filename('flydra.a2',
                                             'sample_calibration.xml')

def test_camera_model():
    R1 = reconstruct.Reconstructor(sample_cal)
    R2 = R1.convert_to_camera_model()

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
