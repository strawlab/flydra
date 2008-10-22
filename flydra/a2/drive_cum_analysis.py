import os, sys

import numpy as np

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
        fps = result_utils.get_fps( data_file )
        self._fps = fps
        file_timestamp = data_file.filename[4:19]
        self._stim_xml = self._fanout.get_stimulus_for_timestamp(timestamp_string=file_timestamp)
    def __repr__(self):
        return 'FlyId("%s")'%self._kalman_filename
    def get_fps(self):
        """return frames per second"""
        return self._fps
    def get_stim_xml(self):
        return self._stim_xml
    def get_overriden(self,stim_xml):
        return OverriddenFlyId(self._kalman_filename,stim_xml)
    def get_list_of_kalman_rows(self,flystate='flying'):
        ca = core_analysis.get_global_CachingAnalyzer()
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(self._kalman_filename)

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
        dropped_obj_ids=[]
        for obj_id in use_obj_ids:
            try:
                kalman_rows = ca.load_data( obj_id, data_file,
                                            dynamic_model_name = dynamic_model,
                                            frames_per_second=self._fps,
                                            flystate='flying',
                                            walking_start_stops=walking_start_stops,
                                            )
            except core_analysis.NotEnoughDataToSmoothError:
                dropped_obj_ids.append(obj_id)
                continue
            except:
                print >> sys.stderr, "error (below) while processing %s %d"%(data_file.filename, obj_id)
                raise
            else:
                if len(kalman_rows) < 3:
                    dropped_obj_ids.append(obj_id)
                    continue
                result.append(kalman_rows)
        if len( dropped_obj_ids ):
            print >> sys.stderr, 'due to short length of data, dropped obj_ids (in %s):'%data_file.filename, dropped_obj_ids
        return result

class OverriddenFlyId(FlyId):
    def __init__(self,kalman_filename,forced_stim_xml):
        super(OverriddenFlyId,self).__init__(kalman_filename)
        self._stim_xml = forced_stim_xml
    def __repr__(self):
        return 'OverriddenFlyId("%s",%s)'%(self._kalman_filename,self._stim_xml)

class Treatment(list):
    pass

class OverrideIterator(object):
    def __init__(self,orig):
        self._orig = orig
        self._place = 0
    def next(self):
        if self._place >= len(self._orig):
            raise StopIteration()
        result = self._orig[self._place]
        self._place += 1
        return result

class TreatmentOverride(Treatment):
    def __init__(self,*args,**kwargs):
        self.newkws = {}
        if 'stim_xml' in kwargs:
            self.newkws['stim_xml']=kwargs['stim_xml']
            del kwargs['stim_xml']
        super(TreatmentOverride,self).__init__(*args,**kwargs)
    def __iter__(self):
        return OverrideIterator(self)
    def __getitem__(self,name):
        orig = super(TreatmentOverride,self).__getitem__(name)
        overriden = orig.get_overriden(self.newkws['stim_xml'])
        return overriden

def make_giant_arrays( treatment, graphical_debug=False ):
    all_kalman_rows = []
    rcoords = []
    saccades = []
    for flyid in treatment:
        list_of_kalman_rows = flyid.get_list_of_kalman_rows() # one for each obj_id
        list_of_rcoords = [ posts.calc_retinal_coord_array( kr,
                                                          flyid.get_fps(),
                                                          flyid.get_stim_xml() ) \
                          for kr in list_of_kalman_rows ]
        rcoords.extend( list_of_rcoords )
        saccades.extend( [ core_analysis.detect_saccades( kr,
                                                          frames_per_second=flyid.get_fps() )
                           for kr in list_of_kalman_rows ] )
        all_kalman_rows.extend( list_of_kalman_rows )

        if 0:
            # verify the above works as intended
            saccade_results = saccades[-1]
            for i,(search_frame, sX) in enumerate(zip(saccade_results['frames'],
                                                    saccade_results['X'],
                                                    )):
                print
                print 'search_frame',search_frame
                print 'sX',sX
                print "rcoords[-1]['frame'][0],rcoords[-1]['frame'][-1]",rcoords[-1]['frame'][0],rcoords[-1]['frame'][-1]
                cond = rcoords[-1]['frame'] ==search_frame
                assert np.sum(cond)==1
                idx = np.nonzero( cond )[0]
                #assert len(idx)==1
                idx = idx[0]
                print 'rcoords[-1][idx]',rcoords[-1][idx]
                if i == 10:
                    break
            sys.exit(0)
        if graphical_debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax1=fig.add_subplot(3,1,1)
            ax2=fig.add_subplot(3,1,2)
            ax3=fig.add_subplot(3,1,3,sharex=ax1,sharey=ax1)
            for kr,this_rcoords in zip(list_of_kalman_rows, list_of_rcoords):
                this_obj_ids = kr['obj_id']
                obj_id = np.unique1d(this_obj_ids)
                assert len(obj_id)==1
                obj_id == obj_id[0]
                line,=ax1.plot( kr['x'], kr['y'], '.', label='obj %d'%obj_id )
                stim_xml = flyid.get_stim_xml()
                closest_dist = np.ma.array(this_rcoords[ 'closest_dist' ],mask=this_rcoords[ 'closest_dist_mask' ])
                angle_of_closest_dist = np.ma.array(this_rcoords[ 'angle_of_closest_dist' ],mask=this_rcoords[ 'closest_dist_mask' ])
                #ax2.plot( closest_dist.data, angle_of_closest_dist.data,
                #          '.', color=line.get_color() )
                ax2.plot( closest_dist.compressed(), angle_of_closest_dist.compressed(),
                          '.', color=line.get_color() )
                ax3.plot( this_rcoords['x'], this_rcoords['y'], '.', label='obj %d'%obj_id, color=line.get_color() )
            stim_xml.plot_stim( ax1, projection=xml_stimulus.SimpleOrthographicXYProjection())
            stim_xml.plot_stim( ax3, projection=xml_stimulus.SimpleOrthographicXYProjection())
            ax1.legend()
            ax3.legend()
            ax1.set_aspect('equal')
            ax3.set_aspect('equal')
            fig.text(0,0,repr(flyid))
            plt.show()

    results_recarray = np.concatenate( rcoords )

    # find row idx for each saccade in saccades
    offset = 0
    all_saccade_idxs = []
    for i in range(len(saccades)):
        assert len(all_kalman_rows[i]) == len(rcoords[i])
        saccade_results = saccades[i]
        # XXX fixme: speedup with searchsorted type thing
        search_frames = saccade_results['frames']
        for j,search_frame in enumerate(search_frames):
            cond = all_kalman_rows[i]['frame'] == search_frame
            assert np.sum(cond)==1
            found_idx = np.nonzero(cond)[0]
            all_saccade_idxs.append( found_idx + offset )
            if 0:
                print 'j,search_frame',j,search_frame
                # verify workings...
                print 'rcoords[i][found_idx]',rcoords[i][found_idx]
                print saccade_results['X'][j]
        offset += len( all_kalman_rows[i] )
    all_saccade_idxs = np.array( all_saccade_idxs )
    del rcoords
    return results_recarray, all_saccade_idxs

def do_turning_plots( subplot, treatment, condition_name ):
    results_recarray, all_saccade_idxs = make_giant_arrays( treatment )
    closest_dist = np.ma.array(results_recarray[ 'closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])
    angle_of_closest_dist = np.ma.array(results_recarray[ 'angle_of_closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])
    #approaching = abs(post_angle) < np.pi # heading with 90 degrees of post center

    if 'lines' in subplot:
        ax = subplot['lines']
        ax.plot(closest_dist, angle_of_closest_dist, '.' )
        for idx in all_saccade_idxs:
            if not closest_dist.mask[idx]:
                ax.plot([closest_dist[idx]], [angle_of_closest_dist[idx]], 'rd' )
    if 'hexbin' in subplot:
        ax = subplot['hexbin']
        ax.hexbin(closest_dist, angle_of_closest_dist)
    if 'lines_angular_vel' in subplot:
        ax = subplot['lines']
        ax.plot(closest_dist, angle_of_closest_dist, '.' )
        for idx in all_saccade_idxs:
            if not closest_dist.mask[idx]:
                ax.plot([closest_dist[idx]], [angle_of_closest_dist[idx]], 'rd' )
    n_pts = len( closest_dist.filled() )
    print '%s: %d data points (%.1f seconds at 60 fps)'%(condition_name, n_pts, n_pts/60.0 )

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
    virtual_post_stim_xml = xml_stimulus.xml_stimulus_from_filename('virtual_post.xml')
    post_vs_virtualpost = {'post':single_post_experiments,
                           'virtual post':TreatmentOverride(no_post_experiments,
                                                            stim_xml=virtual_post_stim_xml),
                           }
    comparisons = {'post_vs_virtualpost':post_vs_virtualpost,
                   'post_vs_nopost':post_vs_nopost}

    if 1:
        # quiver plots of turning

        import matplotlib.pyplot as plt

        comparison_name = 'post_vs_virtualpost'
        comparison = comparisons[comparison_name]
        condition_names = comparison.keys()
        condition_names.sort()

        n_rows = len(condition_names)

        n_cols = 3
        fig = plt.figure()
        ax = None
        for row, condition_name in enumerate(condition_names):
            subplot = {}
            subplot['quiver'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
            if ax is None:
                ax = subplot['quiver']
            subplot['lines']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax,sharey=ax)
            subplot['hexbin']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+3,sharex=ax,sharey=ax)

            subplot['quiver'].set_title( condition_name )
            do_turning_plots( subplot, comparison[condition_name], condition_name )
        plt.show()

