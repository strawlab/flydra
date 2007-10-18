import unittest
import reconstruct
import geom

try:
    import numpy.testing.parametric as parametric
except ImportError,err:
    # old numpy (without module), use local copy
    import numpy_testing_parametric as parametric
    
class TestGeom(parametric.ParametricTestCase):
    #: Prefix for tests with independent state.  These methods will be run with
    #: a separate setUp/tearDown call for each test in the group.
    _indepParTestPrefix = 'test_geom'
    
    def tstXX(self,x1,y1,z1,x2):
        pt_a = [x1,y1,z1,1]
        pt_b = [x2,5,6,1]
        hz_p = reconstruct.pluecker_from_verts(pt_a,pt_b)

        a=geom.ThreeTuple(pt_a[:3])
        b=geom.ThreeTuple(pt_b[:3])
        L = geom.line_from_points(a,b)

        hzL = geom.line_from_HZline(hz_p)

        strL = str(L)
        strhzL = str(hzL)
        assert strL==strhzL
        if 0:
            print 'hz_p',hz_p
            print 'correct',L
            print 'test   ',hzL
            print

    def test_geom(self):
        for x1 in [1,100,10000]:
            for y1 in [5,50,500]:
                for z1 in [-10,234,0]:
                    for x2 in [3,50]:
                        yield (self.tstXX,x1,y1,z1,x2)

    def test_line_from_points(self):
        line=geom.line_from_points(geom.ThreeTuple((2,1,0)),
                                   geom.ThreeTuple((2,0,0)))
        line.closest()
        line.dist2()

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestGeom),
                           ])
    return ts
