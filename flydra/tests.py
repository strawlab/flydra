import unittest
import reconstruct
import pkg_resources

import flydra.geom
import flydra.fastgeom

try:
    import numpy.testing.parametric as parametric
except ImportError,err:
    # old numpy (without module), use local copy
    import numpy_testing_parametric as parametric

##class TestGeom(parametric.ParametricTestCase):
##    _indepParTestPrefix = 'test_geom'
##    def test_geom(self):
##        for mod in [flydra.geom,
##                    flydra.fastgeom]:
##                yield (test,mod)

class TestGeomParametric(parametric.ParametricTestCase):
    #: Prefix for tests with independent state.  These methods will be run with
    #: a separate setUp/tearDown call for each test in the group.
    _indepParTestPrefix = 'test_geom'

    def test_geom(self):
        for mod in [flydra.geom,
                    flydra.fastgeom]:
            for x1 in [1,100,10000]:
                for y1 in [5,50,500]:
                    for z1 in [-10,234,0]:
                        for x2 in [3,50]:
                            yield (self.tstXX,mod,x1,y1,z1,x2)
            for test in [self.tst_tuple_neg,
                         self.tst_tuple_multiply1,
                         self.tst_tuple_multiply2,
                         self.tst_line_closest1,
                         self.tst_line_translate,
                         self.tst_line_from_points,
                         self.tst_init]:
                yield (test, mod)


    def tstXX(self,geom,x1,y1,z1,x2):
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

    def tst_line_from_points(self,geom):
        line=geom.line_from_points(geom.ThreeTuple((2,1,0)),
                                   geom.ThreeTuple((2,0,0)))
        line.closest()
        line.dist2()

    def tst_line_closest1(self,geom):
        if geom is flydra.fastgeom:
            return # not implemented
        xaxis=geom.line_from_points(geom.ThreeTuple((0,0,0)),
                                    geom.ThreeTuple((1,0,0)))
        zline=geom.line_from_points(geom.ThreeTuple((.5,0,0)),
                                    geom.ThreeTuple((.5,0,1)))
        result = xaxis.get_my_point_closest_to_line( zline )
        eps = 1e-10
        assert result.dist_from( geom.ThreeTuple( (0.5, 0, 0) )) < eps

    def tst_init(self,geom):
        a = geom.ThreeTuple((1,2,3))
        b = geom.ThreeTuple(a)
        assert a==b

    def tst_tuple_neg(self,geom):
        a = geom.ThreeTuple((1,2,3))
        b = -a
        c = geom.ThreeTuple((-1,-2,-3))
        assert b == c

    def tst_tuple_multiply1(self,geom):
        x = 2.0
        a = geom.ThreeTuple((1,2,3))
        b = x*a
        c = a*x
        assert b == c

    def tst_tuple_multiply2(self,geom):
        x = -1.0
        a = geom.ThreeTuple((1,2,3))
        b = x*a
        c = -a
        assert b == c

    def tst_line_translate(self,geom):
        a = geom.ThreeTuple((0,0,1))
        b = geom.ThreeTuple((0,1,0))
        c = geom.ThreeTuple((1,0,0))
        ln = geom.PlueckerLine(a,b)
        lnx = ln.translate(c)
        assert lnx == geom.PlueckerLine(geom.ThreeTuple((0,0,-1)),
                                        geom.ThreeTuple((0,-2,0)))


class TestReconstructor(unittest.TestCase):
    def test_from_sample_directory(self):
        caldir = pkg_resources.resource_filename(__name__,"sample_calibration")
        reconstruct.Reconstructor(caldir)
    def test_pickle(self):
        caldir = pkg_resources.resource_filename(__name__,"sample_calibration")
        x=reconstruct.Reconstructor(caldir)
        import pickle
        pickle.dumps(x)

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestGeomParametric),
                           unittest.makeSuite(TestReconstructor),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
