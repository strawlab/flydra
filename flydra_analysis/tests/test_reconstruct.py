import os
import pickle
from io import BytesIO
import xml.etree.ElementTree as ET
import flydra_core.reconstruct as reconstruct
import pkg_resources
import numpy as np
import unittest

sample_cal = pkg_resources.resource_filename('flydra_analysis.a2',
                                             'sample_calibration.xml')

class TestReconstructor(unittest.TestCase):
    def test_from_sample_directory(self):
        caldir = os.path.join(os.path.split(__file__)[0],"sample_calibration")
        reconstruct.Reconstructor(caldir)
    def test_pickle(self):
        caldir = os.path.join(os.path.split(__file__)[0],"sample_calibration")
        x=reconstruct.Reconstructor(caldir)
        xpickle = pickle.dumps(x)
        x2=pickle.loads(xpickle)
        assert x2==x
    def test_xml(self):
        caldir = os.path.join(os.path.split(__file__)[0],"sample_calibration")
        x=reconstruct.Reconstructor(caldir)

        root = ET.Element("xml")
        x.add_element(root)

        tree = ET.ElementTree(root)
        fd = BytesIO()
        tree.write(fd)
        if 0:
            tree.write("test.xml")

        fd.seek(0)

        root = ET.parse(fd).getroot()
        assert root.tag == 'xml'
        assert len(root)==1
        y = reconstruct.Reconstructor_from_xml(root[0])
        assert x==y

def test_norm_vec():
    a = np.array([1,2,3.0])
    an = reconstruct.norm_vec(a)
    expected = a/np.sqrt(1+4+9.0)
    assert np.allclose( an, expected )

    a = np.array([[1,2,3.0],
                  [0,0,.1],
                  ])
    an = reconstruct.norm_vec(a)
    expected = a[:]
    expected[0] = expected[0]/np.sqrt(1+4+9.0)
    expected[1] = np.array([0,0,1])
    assert np.allclose( an, expected )

def test_pluecker():
    # define a line by pair of 3D points
    pts_3d = np.array([ (0.0, 0.0, 0.0),
                        (0.1, 0.0, 0.0),
                        ])
    ptA, ptB = pts_3d
    dir_expected = reconstruct.norm_vec(ptB-ptA)
    p = reconstruct.pluecker_from_verts(ptA,ptB)
    dir_actual = reconstruct.line_direction(p)
    assert np.allclose(dir_expected, dir_actual)

def test_find_line():
    R = reconstruct.Reconstructor(sample_cal)

    # define a line by pair of 3D points
    pts_3d = np.array([ (0.0, 0.0, 0.0),
                        (0.1, 0.0, 0.0),
                        ])
    ptA, ptB = pts_3d
    pluecker_expected = reconstruct.pluecker_from_verts(ptA,ptB)

    # build 3D->2D projection of line
    inputs = []
    for cam_id in R.get_cam_ids():
        hlper = R.get_reconstruct_helper_dict()[cam_id]
        pmat_inv = R.get_pmat_inv(cam_id)
        cc = R.get_camera_center(cam_id)[:,0]
        cc = np.array([cc[0],cc[1],cc[2],1.0])
        a = R.find2d( cam_id, ptA )
        b = R.find2d( cam_id, ptB )

        x0_abs, y0_abs = a
        x0u, y0u = hlper.undistort( x0_abs, y0_abs )

        x,y = a
        x1,y1 = b
        rise = y1-y
        run = x1-x
        slope = rise/run
        area = 1.0
        eccentricity = 4.0

        if 1:
            # ugly. :(
            (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4,
             ray5) = reconstruct.do_3d_operations_on_2d_point(hlper,x0u,y0u,
                                                              pmat_inv,
                                                              cc,
                                                              x0_abs, y0_abs,
                                                              rise, run)

        value_tuple = x,y,area,slope,eccentricity, p1,p2,p3,p4

        inputs.append( (cam_id, value_tuple) )
    X, Lcoords = R.find3d( inputs, return_X_coords=True, return_line_coords=True )

    assert np.allclose(X, ptA)

    v1 = reconstruct.norm_vec(pluecker_expected)
    v2 = reconstruct.norm_vec(Lcoords)
    same1 = np.allclose( v1, v2)
    same2 = np.allclose( v1, -v2)
    assert same1 or same2
