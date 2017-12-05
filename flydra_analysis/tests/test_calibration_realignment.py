import flydra_core.reconstruct as reconstruct
import flydra_core.align as align
import pkg_resources
import numpy as np

sample_cal = pkg_resources.resource_filename('flydra_analysis.a2',
                                             'sample_calibration.xml')

def test_homography():
    for distorted in [True,False]:
        yield check_homography, distorted

def check_homography(distorted=True):
    """check that realignment of calibration doesn't shift 2D projections
    """
    srcR = reconstruct.Reconstructor(sample_cal)
    cam_ids = srcR.cam_ids

    s=0.180548337471
    R=[[-0.90583342, -0.41785822,  0.06971599],
       [ 0.42066911, -0.90666599,  0.03153224],
       [ 0.05003311, 0.05789032,  0.9970684 ]]
    t=[ 0.0393599, -0.01713819,  0.71691032]

    M = align.build_xform(s,R,t)

    alignedR = srcR.get_aligned_copy(M)

    srcX = [[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]]

    for pt in srcX:
        all_src_pts = []
        for cam_id in cam_ids:
            uv = srcR.find2d(cam_id,pt,distorted=distorted)
            all_src_pts.append( (cam_id,uv) )

        pth = np.ones((4,1))
        pth[:3,0] = pt
        expected_pth = np.dot( M, pth )
        expected_pt = expected_pth[:3,0]/expected_pth[3]

        aligned_pt = alignedR.find3d( all_src_pts,
                                      undistort=not distorted,
                                      return_line_coords=False)
        assert np.allclose(expected_pt, aligned_pt)
        all_reproj_pts = []
        for cam_id in cam_ids:
            uv = alignedR.find2d(cam_id,aligned_pt,distorted=distorted)
            all_reproj_pts.append( (cam_id,uv) )

        for i in range(len(all_src_pts)):
            orig2d = all_src_pts[i][1]
            new2d = all_reproj_pts[i][1]
            assert np.allclose( orig2d, new2d )
