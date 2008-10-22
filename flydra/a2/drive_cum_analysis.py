import os, sys

import flydra.a2.posts as posts
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.core_analysis as core_analysis
import flydra.a2.analysis_options as analysis_options
import flydra.analysis.result_utils as result_utils
import flydra.a2.flypos

class FlyId(object):

    """

    Abstraction to make it possbile (easy?) to add support for
    multiple flies per .kh5 file or a fly split across .kh5 files.

    """

    def __init__(self,kalman_filename):
        if not os.path.exists(kalman_filename):
            raise ValueError('kalman_filename %s does not exist'%kalman_filename)
        self._kalman_filename = kalman_filename
        orig_dir = os.path.split(os.path.realpath(kalman_filename))[0]
        test_fanout_filename = os.path.join( orig_dir, 'fanout.xml' )
        if os.path.exists(test_fanout_filename):
            self._fanout_filename = test_fanout_filename
        else:
            raise RuntimeError('could not find fanout file name (guessed %s)'%test_fanout_filename)
        self._fanout = xml_stimulus.xml_fanout_from_filename( self._fanout_filename )
        ca = core_analysis.get_global_CachingAnalyzer()
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(self._kalman_filename)
        file_timestamp = data_file.filename[4:19]
        self._stim_xml = self._fanout.get_stimulus_for_timestamp(timestamp_string=file_timestamp)

    def get_list_of_kalman_rows(self):
        ca = core_analysis.get_global_CachingAnalyzer()

        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(self._kalman_filename)
        fps = result_utils.get_fps( data_file )
        self._fps = fps

        if 1:
            dynamic_model = extra['dynamic_model_name']
            if dynamic_model.startswith('EKF '):
                dynamic_model = dynamic_model[4:]
        self._dynamic_model = dynamic_model

        file_timestamp = data_file.filename[4:19]
        include_obj_ids, exclude_obj_ids = self._fanout.get_obj_ids_for_timestamp( timestamp_string=file_timestamp )
        walking_start_stops = self._fanout.get_walking_start_stops_for_timestamp( timestamp_string=file_timestamp )
        if include_obj_ids is not None:
            use_obj_ids = include_obj_ids
        if exclude_obj_ids is not None:
            use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )

        result = []
        for obj_id in use_obj_ids:
            try:
                kalman_rows = ca.load_data( obj_id, data_file,
                                            dynamic_model_name = dynamic_model,
                                            frames_per_second=fps)
            except core_analysis.NotEnoughDataToSmoothError:
                continue
            except:
                print >> sys.stderr, "error (below) while processing %s %d"%(data_file.filename, obj_id)
                raise
            else:
                result.append(kalman_rows)
        return result

class Treatment(list):
    pass

class EmptyClass(object):
    pass

def do_turning_plots( subplot, treatment ):
    list_of_kalman_rows = []
    for flyid in treatment:
        list_of_kalman_rows.extend ( flyid.get_list_of_kalman_rows() )


if __name__=='__main__':
    single_post_experiments = Treatment([
        FlyId('DATA20080528_201023.kh5'),
        FlyId('DATA20080528_203038.kh5'),
        FlyId('DATA20080528_204034.kh5'),
        FlyId('DATA20080528_205525.kh5'),
        ])
    no_post_experiments = Treatment([
        FlyId('DATA20080602_201151.kh5'),
        FlyId('DATA20080602_203633.kh5'),
        FlyId('DATA20080602_204408.kh5'),
        FlyId('DATA20080605_200750.kh5'),
        FlyId('DATA20080605_201343.kh5'),
        FlyId('DATA20080605_202141.kh5'),
        FlyId('DATA20080605_204242.kh5'),
        FlyId('DATA20080605_204712.kh5'),
        FlyId('DATA20080605_204918.kh5'),
        ])
    post_vs_nopost = {'post':single_post_experiments,
                      'empty':no_post_experiments,
                      }

    if 1:
        # quiver plots of turning

        import matplotlib.pyplot as plt

        conditions = post_vs_nopost.keys()
        conditions.sort()

        n_rows = len(conditions)

        n_cols = 2
        fig = plt.figure()
        for row, condition in enumerate(conditions):
            subplot = {}
            subplot['quiver'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1)
            subplot['lines']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2)

            do_turning_plots( subplot, post_vs_nopost[condition] )
        plt.show()

