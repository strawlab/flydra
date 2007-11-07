import unittest
import flydra_tracker
import flydra.reconstruct
import pkg_resources

class TestTracker(unittest.TestCase):
    def setUp(self):
        caldir = pkg_resources.resource_filename('flydra','sample_calibration')
        self.reconst_orig_units = flydra.reconstruct.Reconstructor(caldir)
        self.reconstructor_meters = self.reconst_orig_units.get_scaled(self.reconst_orig_units.get_scale_factor())
        
    def test_pickle(self):
        x = flydra_tracker.Tracker(self.reconstructor_meters,
                                   scale_factor=self.reconst_orig_units.get_scale_factor())
        
        import pickle
        pickle.dumps(x)
        
def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestTracker),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
