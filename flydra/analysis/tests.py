import result_utils
import unittest
import numpy
import numpy.testing.parametric as parametric # requires numpy 1.0.4

class TestQuickFrameIndexer(parametric.ParametricTestCase):
    _indepParTestPrefix = 'test_x'
    def test_x(self):
        frames = numpy.array([0,0,1,2,3,1,2,3,4,5,6,7,8],
                             dtype=numpy.uint64)
        qfi = result_utils.QuickFrameIndexer(frames)
        for test_val in [-1,0,1,2,3,4,9]:
            yield (self.tstXX,frames,qfi,test_val)
            
    def tstXX(self,frames,qfi,test_val):
        idx_qfi = qfi.get_frame_idxs(test_val)
        idx_nz = numpy.nonzero(frames==test_val)[0]
        assert numpy.allclose(idx_nz,idx_qfi)
    
def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestQuickFrameIndexer),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()

