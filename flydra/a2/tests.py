import pkg_resources
import unittest
import tables as PT
import numpy
import numpy as np
import flydra.a2.core_analysis as core_analysis
import flydra.a2.utils as utils

class TestCoreAnalysis(unittest.TestCase):
    def setUp(self):
        self.ca = core_analysis.CachingAnalyzer()

        filename1=pkg_resources.resource_filename(__name__,'sample_kalman_trajectories.h5')
        filename2=pkg_resources.resource_filename(__name__,'sample_kalman_trajectory.mat')

        filenames = [filename1,
                     filename2,
                     ]
        self.test_obj_ids_list = [[497,1369], #filename1
                                  [1606], #filename2
                                  ]

        self.data_files = []
        self.is_mat_files = []
        self.fps = []
        self.dynamic_model = []
        for filename in filenames:
            obj_ids, use_obj_ids, is_mat_file, data_file, extra = self.ca.initial_file_load(filename)
            self.data_files.append( data_file )
            self.is_mat_files.append( is_mat_file )
            fps = 100.0
            kalman_model = 'fly dynamics, high precision calibration, units: mm'
            self.fps.append( fps )
            self.dynamic_model.append(kalman_model)

    def tearDown(self):
        for data_file,is_mat_file in zip(self.data_files,self.is_mat_files):
            if not is_mat_file:
                data_file.close()

    def test_fast_startstopidx_on_sorted_array_scalar(self):
        sorted_array = numpy.arange(10)
        for value in [-1,0,2,5,6,11]:
            idx_fast_start, idx_fast_stop = core_analysis.fast_startstopidx_on_sorted_array( sorted_array, value )
            idx_slow = numpy.nonzero(sorted_array==value)[0]
            idx_fast = numpy.arange( idx_fast_start, idx_fast_stop )
            self.failUnless( idx_fast.shape == idx_slow.shape )
            self.failUnless( numpy.allclose(idx_fast,idx_slow) )

    def test_fast_startstopidx_on_sorted_array_1d(self):
        sorted_array = numpy.arange(10)
        values = [-1,0,2,5,6,11]

        idx_fast_start, idx_fast_stop = core_analysis.fast_startstopidx_on_sorted_array( sorted_array, values )

        for i,value in enumerate(values):
            idx_slow = numpy.nonzero(sorted_array==value)[0]
            if not len(idx_slow):
                self.failUnless( idx_fast_start[i]==idx_fast_stop[i] )
            else:
                self.failUnless( idx_fast_start[i]==idx_slow[0])
                self.failUnless( idx_fast_stop[i]==(idx_slow[-1]+1))

    def test_CachingAnalyzer_nonexistant_object(self):
        for (data_file,is_mat_file) in zip(self.data_files,
                                           self.is_mat_files,
                                           ):
            for use_kalman_smoothing in [True,False]:
                if is_mat_file and not use_kalman_smoothing:
                    # all data is kalman smoothed in matfile
                    continue

                obj_id = 123456789 # does not exist in file
                try:
                    results = self.ca.calculate_trajectory_metrics(obj_id,
                                                                   data_file,
                                                                   use_kalman_smoothing=use_kalman_smoothing,
                                                                   frames_per_second=100.0,
                                                                   method='position based',
                                                                   method_params={'downsample':1,
                                                                                  },
                                                                   dynamic_model_name='fly dynamics, high precision calibration, units: mm',
                                                                   )
                except core_analysis.NoObjectIDError:
                    pass
                else:
                    raise RuntimeError('We should not get here - a NoObjectIDError should be raised')

    def test_smooth(self):
        for data_file,test_obj_ids,is_mat_file,fps,model in zip(self.data_files,
                                                                self.test_obj_ids_list,
                                                                self.is_mat_files,
                                                                self.fps,
                                                                self.dynamic_model,
                                                                ):
            if is_mat_file:
                # all data is kalman smoothed in matfile
                continue
            for obj_id in test_obj_ids:
                ######## 1. load observations
                obs_obj_ids = data_file.root.kalman_observations.read(field='obj_id')
                obs_idxs = numpy.nonzero(obs_obj_ids == obj_id)[0]

                # Kalman observations are already always in meters, no scale factor needed
                orig_rows = data_file.root.kalman_observations.readCoordinates(obs_idxs)

                ######## 2. perform Kalman smoothing
                rows = core_analysis.observations2smoothed(obj_id,orig_rows,
                                                           frames_per_second=fps,
                                                           dynamic_model_name=model,
                                                           )  # do Kalman smoothing

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
        for data_file,test_obj_ids,is_mat_file,fps,model in zip(self.data_files,
                                                                self.test_obj_ids_list,
                                                                self.is_mat_files,
                                                                self.fps,
                                                                self.dynamic_model,
                                                                ):
            for obj_id in test_obj_ids:

            # Test that load_data() loads similar values for (presumably)
            # forward-only filter and smoother.

                rows_smooth = self.ca.load_data(obj_id,
                                                data_file,
                                                use_kalman_smoothing=True,
                                                frames_per_second=fps,
                                                dynamic_model_name=model,
                                                )
                if is_mat_file:
                    # all data is kalman smoothed in matfile
                    continue
                rows_filt = self.ca.load_data(obj_id,
                                              data_file,
                                              use_kalman_smoothing=False,
                                              frames_per_second=fps,
                                              dynamic_model_name=model,
                                              )
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

    def test_CachingAnalyzer_calculate_trajectory_metrics(self):
        for data_file,test_obj_ids,is_mat_file,fps,model in zip(self.data_files,
                                                                self.test_obj_ids_list,
                                                                self.is_mat_files,
                                                                self.fps,
                                                                self.dynamic_model,
                                                                ):
            for use_kalman_smoothing in [True,False]:
                for obj_id in test_obj_ids:
                    results = self.ca.calculate_trajectory_metrics(obj_id,
                                                                   data_file,
                                                                   use_kalman_smoothing=use_kalman_smoothing,
                                                                   frames_per_second=fps,
                                                                   dynamic_model_name=model,
                                                                   hide_first_point=False,
                                                                   method='position based',
                                                                   method_params={'downsample':1,
                                                                                  })

                    rows = self.ca.load_data( obj_id, data_file,
                                              use_kalman_smoothing=use_kalman_smoothing,
                                              frames_per_second=fps,
                                              dynamic_model_name=model,
                                              ) # load kalman data

                    # if rows are missing in original kalman data, we can interpolate here:

                    #print "len(results['X_kalmanized']),len(rows),obj_id",len(results['X_kalmanized']),len(rows),obj_id
                    assert len(results['X_kalmanized']) == len(rows)


    def test_CachingAnalyzer_load_data(self):
        for data_file,test_obj_ids,is_mat_file,fps,model in zip(self.data_files,
                                                                self.test_obj_ids_list,
                                                                self.is_mat_files,
                                                                self.fps,
                                                                self.dynamic_model,
                                                                ):
            #for use_kalman_smoothing in [True,False]:
            for use_kalman_smoothing in [False,True]:
                for obj_id in test_obj_ids:
                    rows = self.ca.load_data( obj_id, data_file,
                                              use_kalman_smoothing=use_kalman_smoothing,
                                              frames_per_second=fps,
                                              dynamic_model_name=model,
                                              ) # load kalman data
                    #print 'use_kalman_smoothing',use_kalman_smoothing
                    test_obj_ids = obj_id*numpy.ones_like(rows['obj_id'])
                    #print "rows['obj_id'], test_obj_ids",rows['obj_id'], test_obj_ids
                    assert numpy.allclose( rows['obj_id'], test_obj_ids )
                    #print

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

class TestUtils(unittest.TestCase):
    def test_fast_finder(self):
        a = numpy.array([1,2,3,3,2,1,2.3])
        bs = [0, 1, 2, 1.1]
        af = utils.FastFinder(a)
        for b in bs:
            idxs1 = af.get_idxs_of_equal(b)
            idxs2 = numpy.nonzero(a==b)[0]
            assert idxs1.shape == idxs2.shape
            assert numpy.allclose( idxs1, idxs2 )
        for b in bs:
            idx1 = af.get_first_idx_of_assumed_equal(b)
            aval = a[idx1]
            if aval != b:
                if b in a:
                    raise ValueError('b in a, but not found')

class TestChooseOrientations(unittest.TestCase):
    def test_choose_orientations(self):
        x = numpy.linspace(0,1000,10)
        y = numpy.ones_like(x)
        z = numpy.ones_like(x)
        rows = np.recarray( x.shape, dtype=[('x',np.float),
                                            ('y',np.float),
                                            ('z',np.float)])
        rows['x']=x; rows['y']=y; rows['z']=z
        directions = np.zeros((len(x),3))
        directions[:,0] = np.random.randn( len(x) )

        mag = np.sqrt(np.sum(directions**2,axis=1))
        directions = directions/mag[:,np.newaxis]
        #print directions
        new_dir = core_analysis.choose_orientations(rows,directions,frames_per_second=1,
                                                    elevation_up_bias_degrees=0)
        assert new_dir.shape == directions.shape
        assert np.alltrue(~np.isnan(new_dir))


def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestCoreAnalysis),
                           unittest.makeSuite(TestUtils),
                           unittest.makeSuite(TestChooseOrientations),
                           ]
                          )
    return ts

if __name__=='__main__':
    if 1:
        suite = get_test_suite()
        suite.debug()
    else:
        unittest.main()

