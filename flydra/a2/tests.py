import unittest
import pkg_resources
import flydra.a2.core_analysis as core_analysis
import tables as PT
import numpy

class TestCoreAnalysis(unittest.TestCase):
    def setUp(self):
        self.ca = core_analysis.CachingAnalyzer()
        filename=pkg_resources.resource_filename(__name__,'sample_kalman_trajectories.h5')
        self.data_file = PT.openFile(filename,mode="r")
        self.test_obj_ids = [497,1369]

    def tearDown(self):
        self.data_file.close()

    def test_CachingAnalyzer_nonexistant_object(self):
        obj_id = 123456789 # does not exist in file
        try:
            results = self.ca.calculate_trajectory_metrics(obj_id,
                                                           self.data_file,
                                                           use_kalman_smoothing=False,
                                                           frames_per_second=100.0,
                                                           method='position based',
                                                           method_params={'downsample':1,
                                                                          })
        except core_analysis.NoObjectIDError:
            pass
        else:
            raise RuntimeError('We should not get here - a NoObjectIDError should be raised')

    def test_smooth(self):
        for obj_id in self.test_obj_ids:
            ######## 1. load observations
            obs_obj_ids = self.data_file.root.kalman_observations.read(field='obj_id')
            obs_idxs = numpy.nonzero(obs_obj_ids == obj_id)[0]

            # Kalman observations are already always in meters, no scale factor needed
            orig_rows = self.data_file.root.kalman_observations.readCoordinates(obs_idxs)

            ######## 2. perform Kalman smoothing
            rows = core_analysis.observations2smoothed(obj_id,orig_rows)  # do Kalman smoothing

            ######## 3. compare observations with smoothed
            orig = []
            smooth = []

            for i in range(len(rows)):
                frameno = rows[i]['frame']
                idxs = numpy.nonzero(orig_rows['frame']==frameno)[0]
                #print rows[i]
                if len(idxs):
                    assert len(idxs)==1
                    idx = idxs[0]
                    #print '<-',orig_rows[idx]
                    orig.append( (orig_rows[idx]['x'], orig_rows[idx]['y'], orig_rows[idx]['z']) )
                    smooth.append( (rows[i]['x'], rows[i]['y'], rows[i]['z']) )
                #print
            orig = numpy.array(orig)
            smooth = numpy.array(smooth)
            dist = numpy.sqrt(numpy.sum((orig-smooth)**2, axis=1))
            mean_dist = numpy.mean(dist)
            #print 'mean_dist',mean_dist
            assert mean_dist < 1.0 # should certainly be less than 1 meter!

    def test_CachingAnalyzer_load_data(self):

        # Test that load_data() loads similar values for (presumably)
        # forward-only filter and smoother.

        for obj_id in self.test_obj_ids:
            rows_smooth = self.ca.load_data(obj_id,
                                            self.data_file,
                                            use_kalman_smoothing=True)
            rows_filt = self.ca.load_data(obj_id,
                                          self.data_file,
                                          use_kalman_smoothing=False)
            filt = []
            smooth = []
            for i in range(len(rows_smooth)):
                smooth.append( (rows_smooth['x'][i], rows_smooth['y'][i], rows_smooth['z'][i]) )
                filt.append( (rows_filt['x'][i], rows_filt['y'][i], rows_filt['z'][i]) )
            filt = numpy.array(filt)
            smooth = numpy.array(smooth)
            dist = numpy.sqrt(numpy.sum((filt-smooth)**2,axis=1))
            mean_dist = numpy.mean(dist)
            assert mean_dist < 0.1

##     def test_CachingAnalyzer_kalman_smoothing(self):
##         for obj_id in self.test_obj_ids:
##             results_smooth = self.ca.calculate_trajectory_metrics(obj_id,
##                                                                   self.data_file,
##                                                                   use_kalman_smoothing=True,
##                                                                   frames_per_second=100.0,
##                                                                   method='position based',
##                                                                   method_params={'downsample':1,
##                                                                                  })
##             results_orig = self.ca.calculate_trajectory_metrics(obj_id,
##                                                                 self.data_file,
##                                                                 use_kalman_smoothing=False,
##                                                                 frames_per_second=100.0,
##                                                                 method='position based',
##                                                                 method_params={'downsample':1,
##                                                                                })
##             Xsmooth = results_smooth['X_kalmanized']
##             Xorig = results_orig['X_kalmanized']
##             print '-'*80,obj_id
##             for i in range(len(Xsmooth)):
##                 print Xsmooth[i], Xorig[i]
##             print

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestCoreAnalysis),
                           ]
                          )
    return ts

if __name__=='__main__':
    if 1:
        suite = get_test_suite()
        suite.debug()
