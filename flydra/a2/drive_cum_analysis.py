from __future__ import division
import os, sys, warnings

import numpy as np

import flydra.a2.posts as posts
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.core_analysis as core_analysis
import flydra.a2.analysis_options as analysis_options
import flydra.analysis.result_utils as result_utils
import flydra.a2.flypos
import flydra.a2.post_clustering as post_clustering

import scipy.stats
import random

D2R = np.pi/180
R2D = 180/np.pi

global hashed_values, max_hash
hashed_values = {}
max_hash = 0

def calc_hash(mystr):
    global hashed_values, max_hash

    # XXX fixme: should switch to a real hash-generation algorithm

    hash_int = hashed_values.get(mystr,None)
    if hash_int is None:
        hash_int = max_hash
        max_hash += 1
        hashed_values[mystr] = hash_int
    return hash_int

def make_trace_id_from_flyid_and_objid(flyid,obj_id):
    return 'trace(%s,%d)'%(repr(flyid),obj_id)
    #return calc_hash('trace(%s,%d)'%(repr(flyid),obj_id))

def monte_carlo_resample_hist( x, x_edges,
                               N_resamples = 100,
                               frac_observations_per_resample = 0.01,
                               normalize=False,
                               weights=None,
                               ):
    def resample( *args, **kwargs ):
        N=kwargs['N']
        del kwargs['N']
        if len(kwargs.keys()): raise ValueError('not expecting more keyword arguments')
        mylen = len(args[0])
        for a in args[1:]:
            assert mylen==len(a)
        idx = np.arange(mylen)
        np.random.shuffle(idx)

        return [a[idx[:N]] for a in args]

    assert np.allclose( np.ma.getmaskarray(x), np.ma.getmaskarray(weights))

    all_bin_idxs = x_edges.searchsorted( x, side='left' )
    if np.any(all_bin_idxs < 1):
        raise ValueError('Value less x_edges[0] found.')

    assert np.allclose( np.ma.getmaskarray(x), np.ma.getmaskarray(all_bin_idxs))

    N_bins = len(x_edges)-1
    samps = np.zeros( (N_resamples, N_bins) )

    n_observations_per_sample = int(np.round(len(x)*frac_observations_per_resample))

    N_total_obs = len(x)

    if weights is None:
        weights = np.ones_like(x)

    for samp_no in range(N_resamples):
        for i in range(N_bins):
            target_bin_idx = i+1
            resampled_bin_idxs, resampled_weights = resample( all_bin_idxs, weights, N=n_observations_per_sample )
            cond = np.nonzero(resampled_bin_idxs==target_bin_idx)[0] # nonzero() works better than binary condition arrays when masked values involved
            this_rw = resampled_weights[cond]
            N_in_bin = np.sum(np.ma.array(this_rw).compressed())
            samps[samp_no,i] = N_in_bin

    mean_val = np.mean(samps,axis=0)
    std_val = np.std(samps,axis=0)
    if normalize:
        denom = np.sum(mean_val)
        mean_val /= denom
        std_val /= denom
    return mean_val, std_val

if 0:
    obs1 = np.hstack([np.random.randn(1000),
                      0.1*np.random.randn(50)+2.5,
                      ])
    obs2 = np.random.randn(4000)

    obs1 = np.clip(obs1,-5,5)
    obs2 = np.clip(obs2,-5,5)

    x_edges = np.linspace(-5,5,30)
    x_edge_centers = (x_edges[1:]+x_edges[:-1])/2

    mean1,std1 = monte_carlo_resample_hist( obs1, x_edges, normalize=True )
    mean2,std2 = monte_carlo_resample_hist( obs2, x_edges, normalize=True )

    import pylab
    pylab.subplot(2,2,1)
    pylab.hist( obs1, bins=x_edges )
    pylab.title('all, 2 distributions')

    pylab.subplot(2,2,1+2)
    line,=pylab.plot( x_edge_centers, mean1, lw=2, label = 'MC, 2 distributions')
    pylab.plot( x_edge_centers, mean1+std1, color=line.get_color())
    pylab.plot( x_edge_centers, mean1-std1, color=line.get_color())

    line,=pylab.plot( x_edge_centers, mean2, lw=2, label= 'MC, 1 distributions')
    pylab.plot( x_edge_centers, mean2+std2, color=line.get_color())
    pylab.plot( x_edge_centers, mean2-std2, color=line.get_color())

    pylab.subplot(2,2,2)
    pylab.hist( obs2, bins=x_edges )
    pylab.title('all, 1 distribution')

    pylab.show()


try:
    FlyId
except NameError:
    define_classes = True
else:
    define_classes = False

if define_classes:
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
        def get_hash(self):
            mystr = repr(self)
            hash_int = calc_hash(mystr)
            return hash_int
        def get_fps(self):
            """return frames per second"""
            return self._fps
        def get_stim_xml(self):
            return self._stim_xml
        def get_overriden(self,stim_xml):
            return OverriddenFlyId(self._kalman_filename,stim_xml)
        def get_list_of_kalman_rows(self,flystate='flying'):
            return self.get_list_of_kalman_rows_by_source(source='kalman',
                                                          flystate=flystate)
        def get_list_of_kalman_rows_by_source(self,source=None,
                                              flystate='flying',
                                              dynamic_model_name=None,
                                              ):
            """return a list, with one entry per obj_id, of arrays of data"""
            ca = core_analysis.get_global_CachingAnalyzer()
            obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(self._kalman_filename)

            if dynamic_model_name is None:
                dynamic_model_name = extra['dynamic_model_name']
                if dynamic_model_name.startswith('EKF '):
                    dynamic_model_name = dynamic_model_name[4:]
            self._dynamic_model_name = dynamic_model_name

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
                if source=='kalman':
                    try:
                        kalman_rows = ca.load_data(
                            obj_id, data_file,
                            dynamic_model_name = dynamic_model_name,
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
                elif source=='MLE_position':
                    kalman_rows = ca.load_dynamics_free_MLE_position( obj_id, data_file,
                                                                      flystate='flying',
                                                                      walking_start_stops=walking_start_stops,
                                                                      )
                if 1:
                    if len(kalman_rows) < 3:
                        dropped_obj_ids.append(obj_id)
                        continue
                    result.append(kalman_rows)
            if len( dropped_obj_ids ):
                print >> sys.stderr, 'due to short length of data, dropped obj_ids (in %s):'%data_file.filename, dropped_obj_ids

            newresult = []
            self._final_used_obj_ids = []
            # XXX perhaps orig_data_present should be true for all frames of an obj_id...
            for kr in result:
                arrays = []
                names = []
                for name in kr.dtype.names:
                    arrays.append( kr[name] )
                    names.append( name )
                # Add extra field that fuse_obj_ids also adds
                # All true:
                orig_data_present = np.ones( kr['x'].shape, dtype=bool )
                arrays.append( orig_data_present )
                names.append( 'orig_data_present' )
                newresult.append( np.rec.fromarrays( arrays, names=names ) )
                self._final_used_obj_ids.append( int(kr['obj_id'][0]))
            return newresult

        def get_obj_ids(self):
            if not hasattr(self,'_final_used_obj_ids'):
                self.get_list_of_kalman_rows()
            return self._final_used_obj_ids

        def get_saccade_analysis_arrays(self):
            if hasattr(self,'_saccade_cluster_cache'):
                return self._saccade_cluster_cache
            fps = self.get_fps()
            stim_xml = self.get_stim_xml()
            all_obj_ids = []
            all_new_coords = []
            all_rowlabels = []
            trace_ids = []
            A_names = None
            for kalman_rows in self.get_list_of_kalman_rows(): # by fly id
                obj_id = kalman_rows['obj_id'][0]
                assert np.all( kalman_rows['obj_id']==obj_id )
                trace_id = make_trace_id_from_flyid_and_objid(self,obj_id)
                saccade_results = core_analysis.detect_saccades( kalman_rows,
                                                                 frames_per_second=fps )
                results = load_saccade_based_A_matrix(kalman_rows,fps,stim_xml,saccade_results)
                if 1:
                    assert len(results['rowlabels']) == len(kalman_rows)
                A_names = results['A_names']
                if len(results['A']):
                    all_obj_ids.append( obj_id )
                    all_new_coords.append( results['A'] )
                    all_rowlabels.append( results['rowlabels'] )
                    trace_ids.append( trace_id )
            self._saccade_cluster_cache = (all_obj_ids, all_new_coords, all_rowlabels,
                                           trace_ids, A_names)
            return self._saccade_cluster_cache

    class OverriddenFlyId(FlyId):
        def __init__(self,kalman_filename,forced_stim_xml):
            super(OverriddenFlyId,self).__init__(kalman_filename)
            self._stim_xml = forced_stim_xml
        def __repr__(self):
            return 'OverriddenFlyId("%s",%s)'%(self._kalman_filename,self._stim_xml)

    class Treatment(list):
        def get_giant_arrays(self):
            if not hasattr(self, '_giant_cache'):
                self._giant_cache = make_giant_arrays( self )
            return self._giant_cache

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
            self._cached_overriden_by_name = {}
        def __iter__(self):
            return OverrideIterator(self)
        def __getitem__(self,name):
            if name not in self._cached_overriden_by_name:
                orig = super(TreatmentOverride,self).__getitem__(name)
                overriden = orig.get_overriden(self.newkws['stim_xml'])
                self._cached_overriden_by_name[name] = overriden
            return self._cached_overriden_by_name[name]
        def get_non_overriden_item(self,name):
            orig = super(TreatmentOverride,self).__getitem__(name)
            return orig

def make_giant_arrays( treatment, graphical_debug=False ):
    all_kalman_rows = []
    rcoords = []
    saccades = []
    for flyid in treatment:
        try:
            list_of_kalman_rows = flyid.get_list_of_kalman_rows() # one for each obj_id
            flyid_hash = [ flyid.get_hash()*np.ones(kr.shape,dtype=int)
                           for kr in list_of_kalman_rows ]
            list_of_rcoords = [ posts.calc_retinal_coord_array( kr,
                                                                flyid.get_fps(),
                                                                flyid.get_stim_xml(),
                                                                extra_columns=[('flyid_hash',tfh)],
                                                                ) \
                                for (kr,tfh) in zip(list_of_kalman_rows,flyid_hash) ]
            rcoords.extend( list_of_rcoords )
            saccades.extend( [ core_analysis.detect_saccades( kr,
                                                              frames_per_second=flyid.get_fps() )
                               for kr in list_of_kalman_rows ] )
            all_kalman_rows.extend( list_of_kalman_rows )
        except:
            print 'while loading data from',flyid
            raise

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
                ## ax4.plot( closest_dist.compressed(), np.cos(angle_of_closest_dist).compressed(),
                ##           '.', color=line.get_color() )

            stim_xml.plot_stim( ax1,
                                projection=xml_stimulus.SimpleOrthographicXYProjection())
            stim_xml.plot_stim( ax3,
                                projection=xml_stimulus.SimpleOrthographicXYProjection())
            ax1.legend()
            ax3.legend()
            ax1.set_aspect('equal')
            ax3.set_aspect('equal')
            fig.text(0,0,repr(flyid))
            plt.show()

    results_recarray = np.concatenate( rcoords )
    flyid_hash = np.concatenate( flyid_hash )

    # find row idx for each saccade in saccades
    offset = 0
    all_saccade_idxs = []
    for i in range(len(saccades)): # iterate over per-fly results
        assert len(all_kalman_rows[i]) == len(rcoords[i]) # double check assumption
        saccade_results = saccades[i] # get per-fly saccade_idxs
        # XXX fixme: speedup with searchsorted type thing
        search_frames = saccade_results['frames']
        for j,search_frame in enumerate(search_frames):
            # find each row corresponding to a saccade
            cond = all_kalman_rows[i]['frame'] == search_frame
            assert np.sum(cond)==1 # there should be only one
            found_idx = np.nonzero(cond)[0]
            all_saccade_idxs.append( found_idx + offset )
            if 0:
                print 'j,search_frame',j,search_frame
                # verify workings...
                print 'rcoords[i][found_idx]',rcoords[i][found_idx]
                print saccade_results['X'][j]
        offset += len( all_kalman_rows[i] )
    all_saccade_idxs = np.array( all_saccade_idxs )
    print 'orig all_saccade_idxs.shape',all_saccade_idxs.shape
    del rcoords
    return results_recarray, all_saccade_idxs

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab

LUTSIZE = mpl.rcParams['image.lut']
# x y0 y1 tuples
_magenta_green_data = {'red':   ((0., 1., 1.),(1.0, 0.0, 0.0)),
                       'green': ((0., 0.0, 0.0),(1.0, 1.0, 1.0)),
                       'blue':  ((0., 1., 1.),(1.0, 0.0, 0.0))}
magenta_green = colors.LinearSegmentedColormap('magenta_green', _magenta_green_data, LUTSIZE)


_magenta_white_green_data = {'red':   ((0., 1., 1.),  (0.5, 1.0, 1.0),   (1.0, 0.0, 0.0)),
                             'green': ((0., 0.0, 0.0), (0.5, 1.0, 1.0),                  (1.0, 1.0, 1.0)),
                             'blue':  ((0., 1., 1.), (0.5, 1.0, 1.0),    (1.0, 0.0, 0.0))}
magenta_white_green = colors.LinearSegmentedColormap('magenta_white_green', _magenta_white_green_data, LUTSIZE)

_white_magenta_data = {'red':   ((0.0, 1.0, 1.0),   (1.0, 1.0, 1.0)),
                       'green': ((0.0, 1.0, 1.0),   (1.0, 0.0, 0.0)),
                       'blue':  ((0.0, 1.0, 1.0),   (1.0, 1.0, 1.0))}
white_magenta = colors.LinearSegmentedColormap('white_magenta', _white_magenta_data, LUTSIZE)

gray_level = 0.5
_white_gray_data = {'red':   ((0.0, 1.0, 1.0),   (1.0, gray_level, gray_level)),
                     'green': ((0.0, 1.0, 1.0),   (1.0, gray_level, gray_level)),
                     'blue':  ((0.0, 1.0, 1.0),   (1.0, gray_level, gray_level))}
white_gray = colors.LinearSegmentedColormap('white_gray', _white_gray_data, LUTSIZE)


def do_turning_plots( orig_subplot, treatment, condition_name):
    subplot = {}
    subplot.update(orig_subplot)
    results_recarray, all_saccade_idxs = treatment.get_giant_arrays()
    closest_dist = np.ma.array(results_recarray[ 'closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])
    closest_dist_speed = np.ma.array(results_recarray[ 'closest_dist_speed' ],mask=results_recarray[ 'closest_dist_mask' ])
    angle_of_closest_dist = np.ma.array(results_recarray[ 'angle_of_closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])
    #approaching = abs(post_angle) < np.pi # heading with 90 degrees of post center

    key = 'lines'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        ax.plot(closest_dist, angle_of_closest_dist, '.',ms=1)#, ms=0.5 )
        for idx in all_saccade_idxs:
            if not closest_dist.mask[idx]:
                ax.plot([closest_dist[idx]], [angle_of_closest_dist[idx]], 'rx')

    for key in subplot.keys():
        key_start = 'post_angle_hist '
        if key.startswith(key_start):
            ax = subplot[key]
            del subplot[key]
            print 'key',key
            key = key[len(key_start):]
            args = np.array(map(int,key.strip().split()))
            dist_min_cm = args[0]
            dist_max_cm = args[1]

            dist_min = dist_min_cm /100.0
            dist_max = dist_max_cm /100.0

            cond = (dist_min <= closest_dist) & (closest_dist < dist_max)
            this_angle = angle_of_closest_dist[cond]
            this_angle = np.ma.masked_where( np.isnan(this_angle), this_angle)

            this_angle = this_angle.compressed()
            bin_edges = np.linspace(-np.pi,np.pi,20)
            ax.hist( this_angle, bins=bin_edges, alpha=0.5, normed=True)#, new=True )
            ax.grid(True)

    for key in subplot.keys():
        if key.startswith('post_angle_at_dist'):
            ax = subplot[key]
            del subplot[key]
            print 'key',key
            key = key[len('post_angle_at_dist'):]
            args = np.array(map(int,key.strip().split()))
            dist_min_cm = args[0]
            dist_max_cm = args[1]

            dist_min = dist_min_cm /100.0
            dist_max = dist_max_cm /100.0

            cond = (dist_min <= closest_dist) & (closest_dist < dist_max)
            this_angle = angle_of_closest_dist[cond]
            this_angle = np.ma.masked_where( np.isnan(this_angle), this_angle)

            VEL_NORM=True # normalize by velocity to not over-weight slow moving flies
            if VEL_NORM:
                this_horiz_vel = results_recarray['vel_horiz'][cond]
                this_horiz_vel = np.ma.masked_where( np.isnan(this_horiz_vel), this_horiz_vel )
                newmask = np.logical_or( np.ma.getmaskarray(this_angle), np.ma.getmaskarray(this_horiz_vel))
                this_angle.mask = newmask
                this_horiz_vel.mask = newmask

            this_angle.compressed()

            if np.any(np.isnan(this_angle)): raise ValueError('should not have nans here')


            N_angle_bins = 9
            angle_bin_edges = np.linspace(-np.pi, np.pi, N_angle_bins+1 )

            angle_bin_center = (angle_bin_edges[1:]+angle_bin_edges[:-1])/2
            ax.angle_bin_center = angle_bin_center # return result

            if 1:
                angle_bin_idxs = angle_bin_edges.searchsorted( this_angle, side='left' )
                if np.any(angle_bin_idxs < 1):
                    raise ValueError('Angle less than -pi found.')

                N_angle_binned = np.empty((N_angle_bins,))
                for i in range(N_angle_bins):
                    bin_idx = i+1
                    cond = np.ma.array(angle_bin_idxs==bin_idx).compressed()
                    N_angle_binned[i] = np.sum( cond )

                total_N_angle_binned = float(np.sum(N_angle_binned))
                normed_N_angle_binned = N_angle_binned/total_N_angle_binned
                ax.plot( angle_bin_center, normed_N_angle_binned, '-', label=condition_name )
            else:
                N_resamples = 100
                mean_N_angle_binned, std_N_angle_binned = monte_carlo_resample_hist( this_angle,
                                                                                     angle_bin_edges,
                                                                                     N_resamples=N_resamples,
                                                                                     normalize=True,
                                                                                     #weights = this_horiz_vel,
                                                                                     )

                line,=ax.plot( angle_bin_center, mean_N_angle_binned, '-', lw=2, label=condition_name )
                ax.plot( angle_bin_center, (mean_N_angle_binned+std_N_angle_binned), '-', color=line.get_color())
                ax.plot( angle_bin_center, (mean_N_angle_binned-std_N_angle_binned), '-', color=line.get_color())


    # hexbin stuff:
    gridsize = 30,20
    key='hexbin_counts'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        ax.hexbin(closest_dist, angle_of_closest_dist, cmap=white_magenta, gridsize=gridsize)
        ax.set_frame_on(False)

    key='hexbin_flux'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        # With reduce function as sum and the variable speed, 3
        # observations at .333 m/s will contribute equally as 1
        # observation at 1 m/s. Thus, slower flies don't get
        # disproportionately counted.

        # This doesn't seem to work as (well as?) I expected.

        # Which speed to use?
        C = results_recarray[ 'vel_horiz' ]
        #C = closest_dist_speed
        collection = ax.hexbin(closest_dist, angle_of_closest_dist, C=C,
                               vmin = 0, vmax= 1, reduce_C_function=np.sum,
                               cmap=white_magenta, gridsize=gridsize, bins='log',
                               )
        ax.set_frame_on(False)

    key='hexbin_angular_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        ax.hexbin(closest_dist, angle_of_closest_dist,
                  C=results_recarray['horizontal_angular_velocity'],
                  vmin = -10,vmax= 10,
                  cmap=magenta_white_green, gridsize=gridsize,
                  )
        ax.set_frame_on(False)

    key='hexbin_abs_angular_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        ax.hexbin(closest_dist, angle_of_closest_dist,
                  C=abs(results_recarray['horizontal_angular_velocity']),
                  vmin = -10,vmax= 10,
                  cmap=magenta_white_green, gridsize=gridsize,
                  )
        ax.set_frame_on(False)

    key='hexbin_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        C = results_recarray[ 'vel_horiz' ]
        #C = closest_dist_speed
        collection = ax.hexbin(closest_dist, angle_of_closest_dist, C=C,
                               vmin = 0, vmax= 1,
                               cmap=white_magenta, gridsize=gridsize,
                               )
        ax.set_frame_on(False)

    key='lines_angular_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]
        ax.scatter(closest_dist, angle_of_closest_dist,
                   c=horizontal_angular_velocity, s=3,
                   vmin=-200*D2R, vmax=200*D2R,
                   edgecolors='none' )
        ax.set_xlabel('post distance (m)')
        ax.set_ylabel('post angle (rad)')

    key='saccade_rate_vs_dist'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        if 0:
            fps = 60.0
            warnings.warn('WARNING: fixing fps to %.1f'%fps)
        else:
            fps = treatment[0].get_fps()
            for i in range(1,len(treatment)):
                assert fps == treatment[i].get_fps()

        bin_edges = np.linspace(0,.6,32)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        if 1:
            # calculate per-fly results
            all_flyid_hash = results_recarray['flyid_hash']
            all_flyid_hash = np.unique1d(all_flyid_hash)

            bin_assignments = bin_edges.searchsorted( closest_dist, side='left')
            result = []
            set1 = set(map(int,all_saccade_idxs))
            for bin_number in range(1,len(bin_edges)):
                cond_in_this_bin = bin_assignments==bin_number
                this_bin_result = []
                # per-fly means and then std of those means
                for flyid_hash in all_flyid_hash:
                    this_fly_cond = results_recarray['flyid_hash']==flyid_hash
                    this_fly_in_this_bin_cond = cond_in_this_bin & this_fly_cond

                    N_observations = np.ma.sum(this_fly_in_this_bin_cond)
                    this_fly_in_this_bin_idx = np.nonzero(this_fly_in_this_bin_cond)[0]

                    set2 = set(map(int,this_fly_in_this_bin_idx))
                    N_saccades = len( set1.intersection( set2) )
                    if N_observations > 0:
                        rate = (N_saccades/float(N_observations))*fps
                    else:
                        rate = np.nan
                    this_bin_result.append( rate )
                this_bin_result = np.array(this_bin_result)
                result.append(this_bin_result)
            result = np.array(result)
            result = np.ma.masked_where( np.isnan(result), result )

            # mean and std across per-fly result
            result_mean = result.mean(axis=1)
            result_std = result.std(axis=1)

            result_N = np.sum(~np.ma.getmaskarray(result),axis=1)
            line,=ax.plot( bin_centers*100, result_mean, lw=3, label=condition_name )
            if 0:
                ax.plot( bin_centers*100, result_mean-result_std, lw=1,color=line.get_color())
                ax.plot( bin_centers*100, result_mean+result_std, lw=1,color=line.get_color())
            else:
                xs,ys=mlab.poly_between( bin_centers*100, result_mean-result_std, result_mean+result_std )
                ax.fill(xs,ys,facecolor=line.get_color(),edgecolor='none',alpha=0.4)

        else:
            # accumulate across all flies
            bin_assignments = bin_edges.searchsorted( closest_dist, side='left')
            result = []
            set1 = set(map(int,all_saccade_idxs))

            for bin_number in range(1,len(bin_edges)):
                idxs_in_this_bin = np.nonzero(bin_assignments==bin_number)[0]
                N_observations = len(idxs_in_this_bin)


                set2 = set(map(int,idxs_in_this_bin))
                N_saccades = len( set1.intersection( set2) )
                rate = (N_saccades/float(N_observations))*fps
                result.append(rate)
            ax.plot( bin_centers*100, result, lw=3, label=condition_name )

        ax.set_xlabel('post distance (cm)')
        ax.set_ylabel('saccade rate (/s)')
        ax.set_yticks([0, 2, 4])

    key='horiz_vel_vs_dist'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        bin_edges = np.linspace(0,.6,32)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        if 1:
            all_flyid_hash = results_recarray['flyid_hash']
            all_flyid_hash = np.unique1d(all_flyid_hash)

            bin_assignments = bin_edges.searchsorted( closest_dist, side='left')
            result = []
            for bin_number in range(1,len(bin_edges)):
                cond_in_this_bin = bin_assignments==bin_number
                this_bin_result = []
                # per-fly means and then std of those means
                for flyid_hash in all_flyid_hash:
                    this_fly_cond = results_recarray['flyid_hash']==flyid_hash
                    this_fly_in_this_bin_cond = cond_in_this_bin & this_fly_cond
                    this_horiz_vel = results_recarray['vel_horiz'][this_fly_in_this_bin_cond]
                    this_horiz_vel = np.ma.masked_where( np.isnan(this_horiz_vel), this_horiz_vel ).compressed()
                    this_bin_result.append( np.mean( this_horiz_vel ))
                this_bin_result = np.array(this_bin_result)
                result.append(this_bin_result)
            result = np.array(result)
            result = np.ma.masked_where( np.isnan(result), result )
            # mean and std across per-fly result
            result_mean = result.mean(axis=1)
            result_std = result.std(axis=1)

            result_N = np.sum(~np.ma.getmaskarray(result),axis=1)
            line,=ax.plot( bin_centers*100, result_mean, lw=3, label=condition_name )
            if 0:
                ax.plot( bin_centers*100, result_mean-result_std, lw=1,color=line.get_color())
                ax.plot( bin_centers*100, result_mean+result_std, lw=1,color=line.get_color())
            else:
                xs,ys=mlab.poly_between( bin_centers*100, result_mean-result_std, result_mean+result_std )
                ax.fill(xs,ys,facecolor=line.get_color(),edgecolor='none',alpha=0.4)

        else:
            bin_assignments = bin_edges.searchsorted( closest_dist, side='left')

            result_mean = []
            result_median = []
            for bin_number in range(1,len(bin_edges)):
                idxs_in_this_bin = np.nonzero(bin_assignments==bin_number)[0]
                this_horiz_vel = results_recarray['vel_horiz'][idxs_in_this_bin]
                this_horiz_vel = np.ma.masked_where( np.isnan(this_horiz_vel), this_horiz_vel ).compressed()

                result_mean.append( np.mean( this_horiz_vel ))
                result_median.append( np.median( this_horiz_vel ))
            ax.plot( bin_centers*100, result_mean, lw=3, label=('mean '+condition_name ) )
            #ax.plot( bin_centers, result_median, label=('median '+condition_name ) )

        ax.set_xlabel('post distance (cm)')
        ax.set_ylabel('mean horizontal velocity (m/s)')
        ax.set_ylim( (0, 0.55))
        ax.set_yticks( [ 0, .2, .4 ])

    key='z_vs_dist'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        bin_edges = np.linspace(0,.6,32)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_assignments = bin_edges.searchsorted( closest_dist, side='left')

        result_mean = []
        result_median = []
        result_min = []
        result_max = []
        for bin_number in range(1,len(bin_edges)):
            idxs_in_this_bin = np.nonzero(bin_assignments==bin_number)[0]
            this_z = results_recarray['z'][idxs_in_this_bin]
            this_z = np.ma.masked_where( np.isnan(this_z), this_z ).compressed()

            result_mean.append( np.mean( this_z ))
            result_median.append( np.median( this_z ))
            result_min.append( np.min( this_z ))
            result_max.append( np.max( this_z ))
        ax.plot( bin_centers*100, result_mean, lw=3, label=('mean '+condition_name ) )
        ax.plot( bin_centers*100, result_median, label=('median '+condition_name ) )
        ax.plot( bin_centers*100, result_min, label=('min '+condition_name ) )
        ax.plot( bin_centers*100, result_max, label=('max '+condition_name ) )

        ax.set_xlabel('post distance (cm)')
        ax.set_ylabel('z (m)')

    key='horiz_vel_vs_dist_hexbin'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        collection = ax.hexbin(closest_dist, results_recarray['vel_horiz'],
                               #vmin = 0, vmax= 1,
                               cmap=white_gray, gridsize=(20,30),
                               )
        ax.set_xlabel('post distance (m)')
        ax.set_ylabel('mean horizontal velocity (m/s)')

    for key in subplot.keys():
        if key.startswith('turn_func'):
            ax = subplot[key]
            del subplot[key]
            print 'key',key
            key = key[9:]
            args = np.array(map(int,key.strip().split()))
            this_delay = args[0] # in msec
            dist_min, dist_max = args[1]/100.0, args[2]/100.0 # convert from cm to m
            this_cond = (closest_dist >= dist_min) & (closest_dist < dist_max)
            this_post_angle = angle_of_closest_dist[this_cond]
            if this_delay==0:
                colname ='horizontal_angular_velocity'
            else:
                colname ='horizontal_angular_velocity_%dmsec_delay'%this_delay
            this_turn_vel = results_recarray[colname][this_cond]
            if 1:
                #x = np.clip(this_post_angle,-np.pi,np.pi)
                x = this_post_angle
                y = np.clip(this_turn_vel,-550*D2R,550*D2R)
                ax.hexbin( x*R2D,y*R2D,
                           gridsize=(33,51),
                           cmap=white_gray,
                           )
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
                ax.grid(True)
                ax.text(0,0,'%d pts (%d flies)'%( len(this_turn_vel), len(treatment)),
                        transform=ax.transAxes,
                        )
            if 1:
                N_angle_bins = 9
                angle_bin_edges = np.linspace(-np.pi, np.pi, N_angle_bins+1 )

                mask_A = np.ma.getmaskarray(this_post_angle)
                mask_B = np.ma.getmaskarray(this_turn_vel)
                mask = reduce(np.logical_or,(mask_A, mask_B, np.isnan(this_post_angle), np.isnan(this_turn_vel)))

                angle = np.ma.array(this_post_angle,mask=mask).compressed()
                vel = np.ma.array(this_turn_vel,mask=mask).compressed()

                angle_bin_idxs = angle_bin_edges.searchsorted( angle, side='left' )
                if np.any(angle_bin_idxs < 1):
                    raise ValueError('Angle less than -pi found.')

                mean_vel_binned = np.empty((N_angle_bins,))
                median_vel_binned = np.empty((N_angle_bins,))
                std_vel_binned = np.empty((N_angle_bins,))
                N_vel_binned = np.empty((N_angle_bins,))

                mean_angle_binned = np.empty((N_angle_bins,))
                median_angle_binned = np.empty((N_angle_bins,))
                std_angle_binned = np.empty((N_angle_bins,))
                N_angle_binned = np.empty((N_angle_bins,))

                vels_by_bin = []
                for i in range(N_angle_bins):
                    bin_idx = i+1
                    cond = angle_bin_idxs==bin_idx
                    vels_in_bin = vel[cond]
                    angles_in_bin = angle[cond]

                    vels_by_bin.append( vels_in_bin )
                    mean_vel_binned[i] = np.mean( vels_in_bin )
                    median_vel_binned[i] = np.median( vels_in_bin )
                    std_vel_binned[i] = np.std( vels_in_bin )
                    N_vel_binned[i] = len( vels_in_bin )

                    mean_angle_binned[i] = np.mean( angles_in_bin )
                    median_angle_binned[i] = np.median( angles_in_bin )
                    std_angle_binned[i] = np.std( angles_in_bin )
                    N_angle_binned[i] = len( angles_in_bin )

                ax.vels_by_bin = vels_by_bin # return result

                angle_bin_center = (angle_bin_edges[1:]+angle_bin_edges[:-1])/2
                ax.angle_bin_center = angle_bin_center # return result

                ax.plot( angle_bin_center*R2D, mean_vel_binned*R2D, 'b-', lw=3 )
                ax.plot( angle_bin_center*R2D, median_vel_binned*R2D, 'g-', lw=3 )

            ax.text(0.5,1,key,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='center')
            ## ax.set_xlabel('post angle (rad)')
            ## ax.set_ylabel('fly angular velocity (rad/sec)')
            ax.set_xlabel('post angle (deg)')
            ax.set_ylabel('fly angular velocity (deg/sec)')
            ax.xaxis.get_label().set_size(16)
            ax.yaxis.get_label().set_size(16)
            ax.set_ylim((-500,500))
            ax.set_xticks([-180,-90,0,90,180])
            ax.set_yticks([-500,-250,0,250,500])

    key='top_view_abs_angular_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        C = abs(results_recarray[ 'horizontal_angular_velocity' ])
        x = results_recarray[ 'x' ]
        y = results_recarray[ 'y' ]
        z = results_recarray['z']
        cond = (z < 0.4)

        ax.hexbin(x[cond],y[cond], C=C[cond]*R2D,
                  vmin=0*D2R, vmax=500,
                  gridsize=25,
                  #cmap=white_gray,
                  cmap=cm.hot,
                  )
        if hasattr( treatment, 'get_non_overriden_item' ):
            # don't draw posts unless they were really there
            flyid = treatment.get_non_overriden_item(0)
        else:
            flyid = treatment[0]
        stim_xml = flyid.get_stim_xml()
        # equality checking not implemented...
        #for i in range(1,len(treatment)):
        #    assert stim_xml == treatment[i].get_stim_xml()
        post_colors = {}
        for post_num in range(4):
            post_colors[post_num]='b'
        stim_xml.plot_stim( ax,
                            projection=xml_stimulus.SimpleOrthographicXYProjection(),
                            draw_post_as_circle=True,
                            post_colors=post_colors,
                            )
        ax.set_aspect('equal')

    key='top_view_horiz_vel'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        C = abs(results_recarray[ 'vel_horiz'])
        x = results_recarray[ 'x' ]
        y = results_recarray[ 'y' ]
        z = results_recarray['z']
        cond = (z < 0.4)

        ax.hexbin(x[cond],y[cond], C=C[cond],
                  vmin=0*D2R, vmax=0.5,
                  gridsize=25,
                  #cmap=white_gray,
                  cmap=cm.hot,
                  )
        if hasattr( treatment, 'get_non_overriden_item' ):
            # don't draw posts unless they were really there
            flyid = treatment.get_non_overriden_item(0)
        else:
            flyid = treatment[0]
        stim_xml = flyid.get_stim_xml()
        # equality checking not implemented...
        #for i in range(1,len(treatment)):
        #    assert stim_xml == treatment[i].get_stim_xml()
        post_colors = {}
        for post_num in range(4):
            post_colors[post_num]='b'
        stim_xml.plot_stim( ax,
                            projection=xml_stimulus.SimpleOrthographicXYProjection(),
                            draw_post_as_circle=True,
                            post_colors=post_colors,
                            )
        ax.set_aspect('equal')

    n_pts = len( closest_dist.filled() )
    print '%s: %d data points (%.1f seconds at 60 fps)'%(condition_name, n_pts, n_pts/60.0 )
    if len(subplot.keys()):
        warnings.warn('unprocessed subplots: %s'%str(subplot.keys()))
    return

def create_saccade_analysis_array( results_recarray,
                                   frames_per_second=None,
                                   saccade_results=None,
                                   ):
    # should add horizontal angular rate to this
    A = []
    A_names = ['final_post_dist',
               'post_x',
               'post_y',
               #'final_horiz_velocity',
               ]
    prev_saccade_idx = 0
    frames = results_recarray['frame']
    rowlabels = np.nan*np.ones(frames.shape)
    current_row = 0
    saccade_frames = np.sort(saccade_results['frames'])
    for next_saccade_frame in saccade_frames:
        cond = frames==next_saccade_frame
        assert np.ma.sum(cond)==1
        next_saccade_idx = np.nonzero(cond)[0][0]

        # break into chunks of ISI
        this_rcoords = results_recarray[prev_saccade_idx:next_saccade_idx]

        closest_dist = np.ma.array(this_rcoords[ 'closest_dist' ],mask=this_rcoords[ 'closest_dist_mask' ])
        angle_of_closest_dist = np.ma.array(this_rcoords[ 'angle_of_closest_dist' ],mask=this_rcoords[ 'closest_dist_mask' ])

        angle_of_closest_dist = np.ma.masked_where( np.isnan(angle_of_closest_dist), angle_of_closest_dist)

        # analyze for results
        post_x_all = np.cos(angle_of_closest_dist)
        post_y_all = np.sin(angle_of_closest_dist)

        post_x = post_x_all.mean()
        post_y = post_y_all.mean()

        good_post_dist = np.ma.compressed(closest_dist)
        if not len(good_post_dist):
            prev_saccade_idx = next_saccade_idx
            continue

        #print '** post_x_mean',post_x

        final_post_dist = good_post_dist[-1]
        A.append( (final_post_dist, post_x, post_y) )

        rowlabels[prev_saccade_idx:next_saccade_idx] = current_row
        current_row += 1
        prev_saccade_idx = next_saccade_idx
        del next_saccade_idx
    A = np.array(A)
    if 1:
        rl = np.ma.masked_where(np.isnan(rowlabels),rowlabels)
        rlu = np.unique1d(rl)
        for i in range(len(A)):
            assert (i in rlu), 'row not in rowlabel'
    return A, rowlabels, A_names

def load_saccade_based_A_matrix( kalman_rows,
                                 fps, stim_xml,
                                 saccade_results):
    """modeled after post_clustering.load_A_matrix()"""
    results_recarray = posts.calc_retinal_coord_array( kalman_rows, fps, stim_xml)
    A, rowlabels, A_names = create_saccade_analysis_array( results_recarray,
                                                           frames_per_second=fps,
                                                           saccade_results=saccade_results,
                                                           )
    results = dict(A=A,
                   A_names=A_names,
                   rowlabels=rowlabels,
                   results_recarray=results_recarray,
                   )
    return results


def do_saccade_clusters( orig_subplot, treatment, condition_name):
    subplot = {}
    subplot.update(orig_subplot)

    all_new_coords = []
    print 'getting data for condition',condition_name,'='*50
    for flyid in treatment:
        stim_xml = flyid.get_stim_xml()
        fps = flyid.get_fps()
        (this_obj_ids, this_new_coords, this_rowlabels, this_trace_ids, this_A_names) = \
                          flyid.get_saccade_analysis_arrays()
        if not len(this_new_coords):
            continue
        all_new_coords.extend( this_new_coords )
        A_names = this_A_names # should always be the same
        if 1:
            #print 'this_new_coords ',this_new_coords
            this_A = np.concatenate( this_new_coords )
            Amean = this_A.mean(axis=0)
            print condition_name,'Fly mean,',flyid, Amean

    A = np.concatenate( all_new_coords )

    key = 'coord_space_hexbin'
    if key in subplot:
        ax = subplot[key]
        del subplot[key]

        #ax.plot( A[:,0], A[:,1], '.' )
        ax.hexbin( np.clip(A[:,0],0,1), A[:,1], gridsize=(30,20),
                   extent = (0,1,-1,1),
                   )
        ax.set_xlabel( A_names[0])
        ax.set_ylabel( A_names[1])

    for key in subplot.keys():
        key_start = 'top_view '
        if 1 and key.startswith(key_start):
            ax = subplot[key]
            del subplot[key]
            flyid_repr = key[len(key_start):]

            found = False
            for flyid in treatment:
                if repr(flyid)==flyid_repr:
                    found=True
                    break
            if not found:
                raise ValueError('I could not find %s'%flyid_repr)

            (this_obj_ids,this_new_coords, this_rowlabels, this_trace_ids, this_A_names) = \
                              flyid.get_saccade_analysis_arrays()
            list_of_kalman_rows = flyid.get_list_of_kalman_rows()
            for kr in list_of_kalman_rows:
                obj_id = kr['obj_id'][0]
                line,=ax.plot( kr['x'], kr['y'], '.', color='0.5' )
                if obj_id not in this_obj_ids:
                    continue
                i = this_obj_ids.index(obj_id)
                A = this_new_coords[i]
                rowlabels = this_rowlabels[i]
                for Arow,Avalue in enumerate(A):
                    cond = Arow == rowlabels
                    idx = np.ma.nonzero(cond)[0]
                    Acol_idx = 1
                    if np.isnan(A[Arow,Acol_idx]):
                        continue
                    #colors = -np.arccos(A[Arow,Acol_idx])*np.ones( (len(idx),))
                    colors = A[Arow,Acol_idx]*np.ones( (len(idx),))
                    this_coll = ax.scatter( kr['x'][idx], kr['y'][idx], c=colors,
                                            vmin=-1, vmax=1, edgecolor='none',
                                            cmap=cm.hot)
                    this_coll.set_zorder(10)

                    line, = ax.plot( [kr['x'][idx][0]], [kr['y'][idx][0]], 'ko', zorder=11)

            stim_xml = flyid.get_stim_xml()
            stim_xml.plot_stim( ax,
                                projection=xml_stimulus.SimpleOrthographicXYProjection())

    for key in subplot.keys():
        key_start = 'segmented '
        if 1 and key.startswith(key_start):
            ax = subplot[key]
            del subplot[key]
            flyid_repr = key[len(key_start):]

            found = False
            for flyid in treatment:
                if repr(flyid)==flyid_repr:
                    found=True
                    break
            if not found:
                raise ValueError('I could not find %s'%flyid_repr)

            (this_obj_ids,this_new_coords, this_rowlabels, this_trace_ids, this_A_names) = \
                              flyid.get_saccade_analysis_arrays()
            list_of_kalman_rows = flyid.get_list_of_kalman_rows()
            for kr in list_of_kalman_rows:
                obj_id = kr['obj_id'][0]
                line,=ax.plot( kr['x'], kr['y'], '.', color='0.5', ms=2 )
                if obj_id not in this_obj_ids:
                    continue
                i = this_obj_ids.index(obj_id)
                A = this_new_coords[i]
                rowlabels = this_rowlabels[i]
                for Arow,Avalue in enumerate(A):
                    cond = Arow == rowlabels
                    idx = np.ma.nonzero(cond)[0]
                    Acol_idx = 1
                    if np.isnan(A[Arow,Acol_idx]):
                        continue

                    if not ((A[Arow,0] <= 0.113) and (A[Arow,1] >= 0.8)):
                        continue

                    ax.plot( kr['x'][idx], kr['y'][idx], 'b.', ms=4, zorder=10)
                    line, = ax.plot( [kr['x'][idx][0]], [kr['y'][idx][0]], 'go', zorder=11,
                                     mec='g')

            stim_xml = flyid.get_stim_xml()
            stim_xml.plot_stim( ax,
                                projection=xml_stimulus.SimpleOrthographicXYProjection(),
                                draw_post_as_circle=True,
                                )

    if len(subplot.keys()):
        warnings.warn('unprocessed subplots: %s'%str(subplot.keys()))
    return

try:
    single_post_experiments
except NameError:
    load_data = True
else:
    load_data = False

if load_data:
    single_post_experiments = Treatment([
        FlyId('DATA20080528_201023.kh5'),
        FlyId('DATA20080528_203038.kh5'),
        FlyId('DATA20080528_204034.kh5'),
        FlyId('DATA20080528_205525.kh5'),

        FlyId('DATA20080606_193651.kh5'),
        FlyId('DATA20080606_194015.kh5'),
        FlyId('DATA20080606_194315.kh5'),
        FlyId('DATA20080606_194959.kh5'),
        FlyId('DATA20080606_195421.kh5'),

        FlyId('DATA20080609_200202.kh5'),
        FlyId('DATA20080609_200850.kh5'),
        FlyId('DATA20080609_202604.kh5'),

        FlyId('DATA20080610_184938.kh5'),
        FlyId('DATA20080610_185947.kh5'),
        FlyId('DATA20080610_194704.kh5'),

        FlyId('DATA20080611_192338.kh5'),
        FlyId('DATA20080611_193158.kh5'),
        FlyId('DATA20080611_195528.kh5'),

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

        FlyId('DATA20080606_200514.kh5'),

        FlyId('DATA20080609_191212.kh5'),
        FlyId('DATA20080609_194405.kh5'),
        FlyId('DATA20080609_195124.kh5'),
        FlyId('DATA20080609_200850.kh5'),
        FlyId('DATA20080609_202604.kh5'),
        FlyId('DATA20080609_204234.kh5'),
        FlyId('DATA20080609_205215.kh5'),

        ])
    no_post_experiments_short = Treatment([
        FlyId('DATA20080602_201151.kh5'),
        #FlyId('DATA20080602_203633.kh5'),
        ])

    four_post_experiments = Treatment([
        ## ## FlyId('DATA20080618_200651.kh5'),
        ## ## FlyId('DATA20080618_201015.kh5'),
        ## ## FlyId('DATA20080618_201833.kh5'),
        ## ## FlyId('DATA20080618_204324.kh5'),

        FlyId('DATA20080619_170010.kh5'),
        FlyId('DATA20080619_172513.kh5'),
        FlyId('DATA20080619_174254.kh5'),
        FlyId('DATA20080619_174954.kh5'),
        FlyId('DATA20080619_180839.kh5'),
        FlyId('DATA20080619_181701.kh5'),
        FlyId('DATA20080619_182104.kh5'),


        FlyId('DATA20080626_204403.kh5'),
        FlyId('DATA20080626_205055.kh5'),
        FlyId('DATA20080626_205255.kh5'),
        FlyId('DATA20080626_211119.kh5'),
        FlyId('DATA20080626_211341.kh5'),

        ])
    four_post_experiments_short = Treatment([
        FlyId('DATA20080619_170010.kh5'),
        #FlyId('DATA20080619_172513.kh5'),
        ])
    post_vs_nopost = {'post':single_post_experiments,
                      'empty':no_post_experiments,
                      }
    virtual_post_stim_xml = xml_stimulus.xml_stimulus_from_filename('virtual_post.xml')
    post_vs_virtualpost = {'post':single_post_experiments,
                           'virtual post':TreatmentOverride(no_post_experiments,
                                                            stim_xml=virtual_post_stim_xml),
                           }

    four_virtual_posts_stim_xml = xml_stimulus.xml_stimulus_from_filename('4postsA.xml')
    posts_vs_virtualposts = {'posts':four_post_experiments,
                             'virtual posts':TreatmentOverride(no_post_experiments,
                                                              stim_xml=four_virtual_posts_stim_xml),
                             }
    posts_vs_virtualposts_short = {'posts':four_post_experiments_short,
                                   'virtual posts':TreatmentOverride(no_post_experiments_short,
                                                                     stim_xml=four_virtual_posts_stim_xml),
                                   }

    comparisons = {'post_vs_virtualpost':post_vs_virtualpost,
                   'post_vs_nopost':post_vs_nopost,
                   'posts_vs_virtualposts':posts_vs_virtualposts,
                   'posts_vs_virtualposts_short':posts_vs_virtualposts_short,
                   }

if __name__=='__main__':

    if 1:
        import matplotlib.pyplot as plt

        #comparison_name = 'post_vs_virtualpost'
        comparison_name = 'posts_vs_virtualposts'
        #comparison_name = 'posts_vs_virtualposts_short'
        comparison = comparisons[comparison_name]
        condition_names = comparison.keys()
        condition_names.sort()

        if 1:
            n_rows = 1
            n_cols = len(condition_names)
            fig = plt.figure()
            row=0
            ax=None
            for col, condition_name in enumerate(condition_names):
                subplot={}
                this_ax = fig.add_subplot(n_rows,
                                          n_cols,
                                          (row*n_cols)+col+1,sharex=ax,sharey=ax)
                if ax is None:
                    ax = this_ax
                subplot['coord_space_hexbin'] = this_ax
                do_saccade_clusters( subplot,
                                     comparison[condition_name],
                                     condition_name )
        if 0:
            for condition_name in condition_names:
                treatment = comparison[condition_name]
                for count,flyid in enumerate(treatment):
                    if count >3:
                        break
                    fig = plt.figure()
                    fig.text(0,0,'%s: %s'%(condition_name, repr(flyid)))
                    subplot = {}
                    this_ax = fig.add_subplot(1,1,1)
                    this_ax.set_frame_on(False)
                    subplot['top_view %s'%repr(flyid)] = this_ax
                    do_saccade_clusters( subplot,
                                         comparison[condition_name],
                                         condition_name )
                    this_ax.set_xticks([])
                    this_ax.set_yticks([])
                    this_ax.set_aspect('equal')
                    collection=this_ax.collections[0]
                    fig.colorbar(collection,ax=this_ax,cax=None)
        if 1:
            for condition_name in condition_names:
                treatment = comparison[condition_name]
                for count,flyid in enumerate(treatment):
                    fig = plt.figure()
                    fig.text(0,0,'%s: %s'%(condition_name, repr(flyid)))
                    subplot = {}
                    this_ax = fig.add_subplot(1,1,1)
                    this_ax.set_frame_on(False)
                    subplot['segmented %s'%repr(flyid)] = this_ax
                    do_saccade_clusters( subplot,
                                         comparison[condition_name],
                                         condition_name )
                    this_ax.set_xticks([])
                    this_ax.set_yticks([])
                    this_ax.set_aspect('equal')

        if 0:
            n_rows = 1
            n_cols = 2

            fig = plt.figure()
            ax = None
            row = 0
            ## hexbin_vel_collection = None

            subplot = {}
            subplot['post_angle_at_dist 0 15'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
            if ax is None:
                ax = subplot['post_angle_at_dist 0 15']
            subplot['post_angle_at_dist 15 45']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax,sharey=ax)

            subplot['post_angle_at_dist 0 15'].set_title( condition_name )

            for row, condition_name in enumerate(condition_names):
                do_turning_plots( subplot, comparison[condition_name], condition_name )

            subplot['post_angle_at_dist 0 15'].legend()
            subplot['post_angle_at_dist 15 45'].legend()

        if 0:
            n_rows = 1
            n_cols = 3

            fig = plt.figure()
            ax = None
            row = 0
            ## hexbin_vel_collection = None

            subplot = {}
            subplot['post_angle_at_dist 0 20'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
            if ax is None:
                ax = subplot['post_angle_at_dist 0 20']
            subplot['post_angle_at_dist 20 40']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax,sharey=ax)
            subplot['post_angle_at_dist 40 60']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+3,sharex=ax,sharey=ax)

            subplot['post_angle_at_dist 0 20'].set_title( condition_name )

            for row, condition_name in enumerate(condition_names):
                do_turning_plots( subplot, comparison[condition_name], condition_name )

            subplot['post_angle_at_dist 0 20'].legend()
            subplot['post_angle_at_dist 20 40'].legend()
            subplot['post_angle_at_dist 40 60'].legend()

        if 0:
            # top view abs angular velocity
            n_rows = len(condition_names)
            n_cols = 1

            ax = None
            #fig = plt.figure(frameon=False,figsize=(4.5,9))
            fig = plt.figure(figsize=(4.5,9))
            for row, condition_name in enumerate(condition_names):
                subplot = {}
                this_ax = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
                subplot['top_view_abs_angular_vel'] = this_ax
                if ax is None:
                    ax = this_ax
                do_turning_plots( subplot, comparison[condition_name], condition_name )
                this_ax.set_frame_on(False)
                ylim = this_ax.get_ylim()
                this_ax.set_ylim( (ylim[0],ylim[1]+.1))
                this_ax.set_xticks([])
                this_ax.set_yticks([])

            collection = ax.collections[0]
            cax = fig.add_axes( (0.85, 0.05, .05, .9))
            cbar = fig.colorbar(collection, cax=cax, ax=ax, ticks=[0,250,500])#,750])

            import pylab
            pylab.subplots_adjust(left=0.12, right=.88)

            for ext in ['.png','.svg']:
            #for ext in ['.png','.pdf','.svg']:
                fname = 'top_view_abs_angular_velocity'+ext
                fig.savefig(fname)#,dpi=55)
                print 'saved',fname

        if 0:
            # top view horiz velocity
            n_rows = len(condition_names)
            n_cols = 1

            ax = None
            #fig = plt.figure(frameon=False,figsize=(4.5,9))
            fig = plt.figure(figsize=(4.5,9))
            for row, condition_name in enumerate(condition_names):
                subplot = {}
                this_ax = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
                subplot['top_view_horiz_vel'] = this_ax
                if ax is None:
                    ax = this_ax
                do_turning_plots( subplot, comparison[condition_name], condition_name )
                this_ax.set_frame_on(False)
                ylim = this_ax.get_ylim()
                this_ax.set_ylim( (ylim[0],ylim[1]+.1))
                this_ax.set_xticks([])
                this_ax.set_yticks([])

            collection = ax.collections[0]
            cax = fig.add_axes( (0.85, 0.05, .05, .9))
            cbar = fig.colorbar(collection, cax=cax, ax=ax, ticks=[0,.25,.5] )

            import pylab
            pylab.subplots_adjust(left=0.12, right=.88)

            for ext in ['.png','.svg']:
            #for ext in ['.png','.pdf','.svg']:
                fname = 'top_view_horiz_vel'+ext
                fig.savefig(fname)#,dpi=55)
                print 'saved',fname

        if 0: # saccade rate vs distance & horiz_vel vs distance
            # maybe nice to put posts.py static angle plot in here?
            n_rows = 2
            n_cols = 1

            fig = plt.figure(figsize=(4.5,5))
            ax = None
            row = 0

            subplot = {}
            subplot['saccade_rate_vs_dist'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax)
            subplot['horiz_vel_vs_dist']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax)
            #subplot['z_vs_dist']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax)

            for condition_name in condition_names:
                do_turning_plots( subplot, comparison[condition_name], condition_name )

            ax = subplot['saccade_rate_vs_dist']
            ax.set_xticklabels([])
            ax.set_xlabel('')

            #subplot['saccade_rate_vs_dist'].legend()
            #subplot['horiz_vel_vs_dist'].legend()
            #subplot['z_vs_dist'].legend()
            import pylab
            pylab.subplots_adjust(left=0.15, right=.96, hspace=0)

            for ext in ['.png','.svg']:
            #for ext in ['.png','.pdf','.svg']:
                fname = 'saccades_and_velocity'+ext
                fig.savefig(fname)#,dpi=55)
                print 'saved',fname

        if 0: # histogram of closest post angle by distance
            n_rows = 1
            n_cols = 6

            fig = plt.figure()
            key_start = 'post_angle_hist '
            ax = None
            row = 0

            subplot = {}
            subplot[key_start+'50 60'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax)
            if 1:
                ax = subplot[key_start+'50 60']
            subplot[key_start+'40 50'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax)
            subplot[key_start+'30 40'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+3,sharex=ax)
            subplot[key_start+'20 30'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+4,sharex=ax)
            subplot[key_start+'10 20'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+5,sharex=ax)
            subplot[key_start+ '0 10'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+6,sharex=ax)

            for condition_name in condition_names:
                do_turning_plots( subplot, comparison[condition_name], condition_name )

        if 0:
            n_rows = 2
            n_cols = 1

            fig = plt.figure()
            ax = None

            #PRETTY_NONINTERACTIVE=False # make pretty
            PRETTY_NONINTERACTIVE=True # make pretty

            for row, condition_name in enumerate(condition_names):
                subplot = {}
                subplot['horiz_vel_vs_dist'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
                if (not PRETTY_NONINTERACTIVE) and (ax is None):
                    ax = subplot['horiz_vel_vs_dist']
                subplot['horiz_vel_vs_dist_hexbin']  = subplot['horiz_vel_vs_dist'] # same axes
                do_turning_plots( subplot, comparison[condition_name], condition_name )

                this_ax = subplot['horiz_vel_vs_dist_hexbin']
                if not PRETTY_NONINTERACTIVE:
                    this_ax.set_title( condition_name )
                else:
                    this_ax.set_xlabel('')
                    if row==0:
                        this_ax.set_xticklabels([])
                    this_ax.set_xlim((0,50))

        if 0:
            n_rows = len(condition_names)
            n_cols = 6

            fig = plt.figure()
            ax = None
            ## hexbin_vel_collection = None
            for row, condition_name in enumerate(condition_names):
                subplot = {}
                subplot['lines_angular_vel'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
                if ax is None:
                    ax = subplot['lines_angular_vel']
                subplot['lines']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax,sharey=ax)
                subplot['hexbin_counts']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+3,sharex=ax,sharey=ax)
                subplot['hexbin_abs_angular_vel']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+4,sharex=ax,sharey=ax)
                subplot['hexbin_angular_vel']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+5,sharex=ax,sharey=ax)

                subplot['hexbin_vel']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+6,sharex=ax,sharey=ax)
                ## if hexbin_vel_collection is None:
                ##     hexbin_vel_collection = subplot['hexbin_vel'].hexbin_vel_collection

                subplot['lines_angular_vel'].set_title( condition_name )

                do_turning_plots( subplot, comparison[condition_name], condition_name )


        if 0:
            # distance dependent turning functions

            #PRETTY_NONINTERACTIVE=False
            PRETTY_NONINTERACTIVE=True # make pretty, but nice interactive features (sharex) disabled
            #for delay_msec in [0,50,100,150]:
            for delay_msec in [0]:
                plot_p_values = False
                n_rows = len(condition_names)+int(plot_p_values)
                n_cols = 6
                #fig = plt.figure(frameon=False,figsize=(14.2,9.1875))
                fig = plt.figure(figsize=(14.2,9.1875))
                if PRETTY_NONINTERACTIVE:
                    import pylab
                    pylab.subplots_adjust(left=0.07, right=.99,top=0.96, wspace=0.0, hspace=0.05)
                if not PRETTY_NONINTERACTIVE:
                    fig.text(0,0,'assuming %d msec latency'%delay_msec)
                ax = None
                key_start = 'turn_func %d '%delay_msec
                if plot_p_values:
                    result = {}
                for row, condition_name in enumerate(condition_names):
                    subplot = {}
                    if PRETTY_NONINTERACTIVE:
                        ax=None
                    subplot[key_start+'50 60'] = fig.add_subplot(n_rows,n_cols,(row*n_cols)+1,sharex=ax,sharey=ax)
                    if (not PRETTY_NONINTERACTIVE) and (ax is None):
                        ax = subplot[key_start+'50 60']
                    subplot[key_start+'40 50']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+2,sharex=ax,sharey=ax)
                    subplot[key_start+'30 40']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+3,sharex=ax,sharey=ax)
                    subplot[key_start+'20 30']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+4,sharex=ax,sharey=ax)
                    subplot[key_start+'10 20']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+5,sharex=ax,sharey=ax)
                    subplot[key_start+'0 10']  = fig.add_subplot(n_rows,n_cols,(row*n_cols)+6,sharex=ax,sharey=ax)

                    if not PRETTY_NONINTERACTIVE:
                        subplot[key_start+'50 60'].set_title( condition_name )
                    do_turning_plots( subplot, comparison[condition_name], condition_name )
                    if PRETTY_NONINTERACTIVE:
                        if row == 0:
                            for key in subplot:
                                subplot[key].set_xlabel('')
                                subplot[key].set_xticklabels([])
                        for key,ax in subplot.iteritems():
                            if key != (key_start+'50 60'): # for all but first column
                                ax.set_ylabel('')
                                ax.set_yticklabels( [] )
                        for ax in subplot.itervalues():
                            del ax.texts[:] # remove all axes text
                            ax.set_ylim((-560,560))
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                            for label in ax.get_xticklabels():
                                plt.setp(label,
                                         rotation=-75)
                        ## #print 'key',key
                        ## ax = subplot[(key_start+'0 10')]
                        ## #print 'ax.get_ylabel()',ax.get_ylabel()
                        ## #print 'ax.get_yticklabels()',[label for label in ax.get_yticklabels()]

                if plot_p_values:
                    # make this optional so we don't have to keep subplot names in sync
                    angle_bin_center =  subplot[key_start+'0 10'].angle_bin_center # these values are the same across all trials
                    result[condition_name] = []
                    result[condition_name].append( subplot[key_start+'0 10'].vels_by_bin ) # column_number 0
                    result[condition_name].append( subplot[key_start+'10 20'].vels_by_bin ) # column_number 1
                    result[condition_name].append( subplot[key_start+'20 30'].vels_by_bin )
                    result[condition_name].append( subplot[key_start+'30 40'].vels_by_bin )
                    result[condition_name].append( subplot[key_start+'40 50'].vels_by_bin )
                    result[condition_name].append( subplot[key_start+'50 60'].vels_by_bin )
                    #result[condition_name].append( subplot[key_start+'50 60'].vels_by_bin )

                if plot_p_values:
                    row = 3
                    assert len(condition_names)==2
                    pax = None
                    for column_number in range(len(result[condition_names[0]])):
                        p_values = np.zeros( (len(angle_bin_center),) )
                        for i in range(len(angle_bin_center)):
                            dist_a = result[condition_names[0]][column_number][i]
                            dist_b = result[condition_names[1]][column_number][i]
                            if 1:
                                newlen = min(len(dist_a), len(dist_b))
                                if newlen < 20:
                                    warnings.warn('N for Wilcoxon test is less than 20')
                                print 'trimming N samples in distribution (down to %d)'%newlen
                                idxs_a = range(len(dist_a))
                                idxs_b = range(len(dist_b))
                                random.shuffle( idxs_a )
                                random.shuffle( idxs_b )
                                idxs_a = idxs_a[:newlen]
                                idxs_b = idxs_b[:newlen]
                                dist_a = dist_a[idxs_a]
                                dist_b = dist_b[idxs_b]
                            T,pval = scipy.stats.wilcoxon(dist_a,dist_b)
                            p_values[i] = pval
                        print (n_rows,
                               n_cols,
                               (row-1)*n_cols + column_number+1)
                        pax = fig.add_subplot(n_rows,
                                              n_cols,
                                              (row-1)*n_cols + column_number+1,
                                              sharex=ax, sharey=pax )
                        pax.plot( angle_bin_center, p_values, 'o' )
                        pax.set_yscale('log')
                        pax.set_ylim((1e-10,1))

                if PRETTY_NONINTERACTIVE:
                    for ext in ['.png']:#,'.pdf','.svg']:
                    #for ext in ['.png','.pdf','.svg']:
                        fname = 'turn_functions'+ext
                        fig.savefig(fname,dpi=55)
                        print 'saved',fname
        plt.show()

