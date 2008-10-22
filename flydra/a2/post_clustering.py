from __future__ import division
from __future__ import with_statement
import pkg_resources
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, time, collections, warnings

import numpy
import numpy as np
from optparse import OptionParser
import scipy.integrate
import scipy.cluster.vq
import flydra.a2.core_analysis as core_analysis
import flydra.a2.flypos
import flydra.a2.analysis_options as analysis_options

import flydra.a2.posts as posts
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.utils as utils

import matplotlib
matplotlib.rcParams['ps.useafm']=True
matplotlib.rcParams['ps.usedistiller'] = 'None'#'xpdf'

def mytake(arr, ma_idx):
    ma_idx = np.ma.masked_where( np.isnan( ma_idx ), ma_idx ).astype(int)
    ma_idx.set_fill_value(0)
    idx_filled = ma_idx.filled()

    result = np.asarray( arr[idx_filled,...], dtype=np.float ) # cast to float so that nan works
    result[ ma_idx.mask ] = np.nan
    return result

def make_branch_cut_pi(arr):
    """shift the branch cut of angles (in radians) from 0/2pi to -pi/+pi."""
    return np.mod(arr+np.pi,2*np.pi)-np.pi

def create_analysis_array( rec, subsample_factor=5, frames_per_second=None, skip_missing=True ):
    """from densely-spaced input data, create analysis array

    array has rows that are observations and columns that are features.

    Inputs
    ======
    rec - record array of densely-spaced input

    Outputs
    =======
    A - an MxN array of M observations over N features
    rowlabels - an array the length of rec labeling the corresponding row of A

    rough example:

    input

    a   b   c
    --- --- ---
    0   10  20
    1   11  30
    2   12  40
    3   13  50
    4   14  60

    output
    ------

    A = [[2,12,40]]
    rowlabls = [ -1, 0, 0, 0, -1 ]

    """
    def downsamp(arr):
        N_observations = len(arr)//subsample_factor
        x = np.ma.reshape(arr, (N_observations,subsample_factor))
        if 0:
            x = np.ma.mean(x,axis=1)
        else:
            result = []
            for i in range(x.shape[0]):
                #col = x[:,j]
                col = x[i,:]
                col = np.ma.asarray(col).compressed()
                if len(col)>2:
                    result.append( np.median( col ) )
                else:
                    result.append( np.mean( col ) )
            x = np.array(result)
        #x = np.ma.median(x,axis=1)
        return x

    rowlabels = np.nan*np.ones( (len(rec),))

    rowlabels_non_nan = np.arange( len(rec) // subsample_factor )
    rowlabels_non_nan = rowlabels_non_nan.repeat( subsample_factor )

    start_idx = 1 # skip nan at start
    stop_idx = len(rowlabels_non_nan)+1
    if stop_idx==len(rowlabels):
        stop_idx -= subsample_factor # skip nan at stop
    rowlabels[start_idx:stop_idx] = rowlabels_non_nan

    horizontal_angular_velocity = downsamp(rec['horizontal_angular_velocity'][start_idx:stop_idx])
    closest_dist = np.ma.array(rec[ 'closest_dist' ],mask=rec[ 'closest_dist_mask' ])
    closest_dist_speed = np.ma.array(rec[ 'closest_dist_speed' ],mask=rec[ 'closest_dist_mask' ])
    closest_dist_accel = np.ma.array(rec[ 'closest_dist_accel' ],mask=rec[ 'closest_dist_mask' ])
    angle_of_closest_dist = np.ma.array(rec[ 'angle_of_closest_dist' ],mask=rec[ 'closest_dist_mask' ])

    post_angle_x = rec[ 'closest_post_angle_x'][start_idx:stop_idx]
    post_angle_y = rec[ 'closest_post_angle_y'][start_idx:stop_idx]
    post_angular_velocity = downsamp(rec[ 'closest_post_angular_velocity'][start_idx:stop_idx])
    vel_mag = np.sqrt(rec['vel_x']**2 + rec['vel_y']**2 + rec['vel_z']**2)

    A = [ abs(horizontal_angular_velocity),
          downsamp(rec['vel_horiz'][start_idx:stop_idx]),
          downsamp(rec['vel_z'][start_idx:stop_idx]),
          downsamp(vel_mag[start_idx:stop_idx]),

          downsamp(closest_dist[start_idx:stop_idx]), # NL func or binarize on this?
          downsamp(closest_dist_speed[start_idx:stop_idx]),
          downsamp(closest_dist_accel[start_idx:stop_idx]),
          downsamp(post_angle_x),
          downsamp(post_angle_y),
          post_angular_velocity,
          ]
    A_names = [
        # motor-only information
        'absolute angular velocity about Z axis (rad/sec)',
        'horizontal velocity (m/sec)',
        'vertical velocity (m/sec)',
        'speed (m/sec)',

        # visual parameters
        #   closest post
        'distance to closest post (m)',
        'speed from closest post (m/sec)',
        'accel from closest post (m/sec/sec)',
        'angle to post (X)',
        'angle to post (Y)',
        'angular velocity to post (rad/sec)',

        #   arena wall

        ]

    A = [np.ma.array(ai,fill_value=np.nan) for ai in A]
    A = [ai.filled() for ai in A] # convert masked entries to nan
    if 1:
        shape = None
        for i,ai in enumerate(A):
            if shape is None:
                shape=ai.shape
            try:
                assert shape==ai.shape
            except:
                print 'assertion failed for row %d'%i
                raise
    DEBUG=False
    if DEBUG:
        N_observations = len(A[0])
        for i in range(N_observations):

            this_start_idx = start_idx + i*subsample_factor
            this_stop_idx = start_idx + (i+1)*subsample_factor
            if 1:
                # test indexing
                idx1 = np.nonzero(rowlabels == i)[0]
                idx2 = np.arange(this_start_idx,this_stop_idx)
                assert np.allclose(idx1,idx2)
            if 1:
                import pylab

                # make plot
                xi = rec['x'][this_start_idx:this_stop_idx]
                yi = rec['y'][this_start_idx:this_stop_idx]
                fig=pylab.figure(figsize=(3,3))
                ax=fig.add_subplot(1,1,1)
                ax.plot(xi,yi,'o-')
                ax.text(xi[0],yi[0],'i %d, turn: %.1f, dist %.1f'%(
                    i,horizontal_angular_velocity[i],A[3][i]))
                ax.set_aspect('equal')
                fig.savefig('debug%05d.png'%i)
                pylab.close(fig)
    A = np.array(A).T # observations = rows, attributes = columns
    if skip_missing:
        Anew = []
        inew = 0

        for i,ai in enumerate(A):
            this_start_idx = start_idx + i*subsample_factor
            this_stop_idx = start_idx + (i+1)*subsample_factor
            #print
            if np.any(~np.isfinite(ai)):
                ## #rowlabels[this_start_idx:] -= 1
                ## #rowlabels[rowlabels==i] = UNUSED_ROWLABEL

                ## #rowlabels = np.ma.masked_where(rowlabels==i, rowlabels) # mask these rows
                ## rowlabels.mask[rowlabels==i] = True
                ## rowlabels[this_start_idx:] -= 1
                rowlabel=np.nan
            else:
                Anew.append( ai )
                rowlabel=inew
                inew += 1

            rowlabels[this_start_idx:this_stop_idx] = rowlabel

            if 0 and i <=45:
                np.set_printoptions(linewidth=150,suppress=True)
                print 'i',i
                print 'np.any(~np.isfinite(ai))',np.any(~np.isfinite(ai))
                #print 'A[:i+1]'
                #print A[:i+1]
                print np.array(Anew)
                print
                print rowlabels[ : (start_idx + (i+1)*subsample_factor + 10) ]
                print
                if 0 and i==30:
                    sys.exit(0)
        A = np.array(Anew)

    return A, rowlabels, A_names

def normalize_array(A):
    column_means = np.mean(A,axis=0)
    column_std = np.std(A,axis=0)
    norm_info = {'means':column_means,
                 'std':column_std}
    normA = (A - column_means)/column_std
    return normA,norm_info

PICK=1
if PICK:
    global line2data
    global ax_xy, ax_xz
    global all_figs

    line2data = {}
    all_figs = []
    def onpick1(event):

        global line2data
        global ax_xy, ax_xz
        global all_figs

        import matplotlib.lines

        print 'onpick1 called:',event

        if isinstance(event.artist,matplotlib.lines.Line2D):
            thisline = event.artist
            data,myax = line2data[thisline]
            # data is DataAssoc instance
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind

            rows_by_trace_id = data.elems_by_trace_id_for_Aidxs( ind )
            trace_ids = rows_by_trace_id.keys()
            for trace_id in trace_ids:
                rows = rows_by_trace_id[trace_id]

                ax_xy[trace_id].plot( rows['x'], rows['y'], 'r.' )
                ax_xz[trace_id].plot( rows['x'], rows['z'], 'r.' )

            for fig in all_figs:
                fig.canvas.draw()

class DataAssoc(object):
    def __init__(self, all_rowlabels, trace_ids, orig_data_by_trace_id, orig_row_offset_by_trace_id):
        self._all_rowlabels = np.asarray(all_rowlabels)
        assert len(self._all_rowlabels.shape) == 1 # 1d array
        self._trace_ids = np.asarray(trace_ids)
        assert len(self._trace_ids.shape)==1
        self._orig_data_by_trace_id = orig_data_by_trace_id
        for value in self._orig_data_by_trace_id.itervalues():
            if not isinstance(value,np.ndarray):
                raise TypeError("original data must be numpy arrays!") # required for indexing
        self._orig_row_offset_by_trace_id = orig_row_offset_by_trace_id

        if 1:
            # find all rowlabels corresponding to each trace_id...
            start_Aidxs = {} # index into the rowlabels array starting again for each trace id
            for trace_id, start_Aidx in self._orig_row_offset_by_trace_id.iteritems():
                start_Aidxs[start_Aidx]=trace_id
            sas = start_Aidxs.keys()
            sas.sort()

            self._rowlabel_startstop_by_trace_id = {}
            for i,start_Aidx in enumerate(sas):
                trace_id = start_Aidxs[start_Aidx]
                if (i+1)>=len(sas):
                    stop_idx = len(self._all_rowlabels)
                else:
                    stop_idx = sas[i+1]
                self._rowlabel_startstop_by_trace_id[trace_id] = (start_Aidx, stop_idx)

        self._orig_trace_id_and_idxs_from_A_row = []

        for i,trace_id in enumerate(self._trace_ids):
            # i is index into A (selects row)
            cond = self._all_rowlabels==i # find the rowlabels for this row of A
            cum_orig_idxs = np.nonzero(cond)[0]  # convert to indices into original data (modulo offset)
            if len(cum_orig_idxs)==0:
                raise RuntimeError('the generated data must come from somewhere, but could not find it in rowlabels!')

            offset = orig_row_offset_by_trace_id[trace_id] # index into A
            orig_idxs = cum_orig_idxs - offset # convert to original rowlabels

            orig_idxs = orig_idxs.tolist()
            print 'i,trace_id',i,trace_id
            print 'orig_idxs'
            print orig_idxs
            print
            self._orig_trace_id_and_idxs_from_A_row.append( (trace_id, orig_idxs) )

    def get_rowlabels_for_trace_id(self,trace_id):
        start_idx,stop_idx = self._rowlabel_startstop_by_trace_id[trace_id]
        cond = np.arange(start_idx,stop_idx)
        #cond = self._trace_ids == trace_id
        result = self._all_rowlabels[cond]
        return result

    def idxs_by_trace_id_for_Aidxs(self,ind):
        """return indices of original data given an indices into A

        Return value is a dict of (key,values) where key is trace_id
        and values are indices into original data.

        """

        x=collections.defaultdict(list)
        for ii in ind:
            trace_id, orig_idxs = self._orig_trace_id_and_idxs_from_A_row[ii]
            x[trace_id].extend( orig_idxs )

        # convert to numpy arrays (in normal dict) and return
        idxs_by_trace_id = {}
        for trace_id in x.keys():
            idxs_by_trace_id[trace_id] = np.array( x[trace_id] )

        return idxs_by_trace_id

    def elems_by_trace_id_for_Aidxs(self,ind):
        idxs_by_trace_id = self.idxs_by_trace_id_for_Aidxs(ind)
        results = {}
        for trace_id, orig_idxs in idxs_by_trace_id.iteritems():
            all_orig_elements = self._orig_data_by_trace_id[trace_id]
            try:
                selected_orig_elements = all_orig_elements[orig_idxs]
            except:
                print 'ind',ind
                print 'orig_idxs',orig_idxs
                raise
            results[trace_id] = selected_orig_elements
        return results

def test_data_assoc_1():
    # define the original data:
    a='trace A'
    b='trace B'
    orig_data_by_trace_id = {a:np.array([ 0.1, 0.2, 0.3, 0.4,   # becomes A row 0
                                          1.1, 1.2, ]),          # becomes A row 1
                             b:np.array([ 2.1,                  # becomes A row 2
                                          3.1, 3.2, 3,3,3,3,3])} # becomes A row 3

    # now A matrix is somehow generated from it:
    A = [[1,2], # from trace A rows 0,1,2,3
         [2,3], # from trace A rows 4,5
         [3,4], # from trace B rows 0
         [5,6]] # from trace B rows 1,2,3,4,5,6,7
    trace_ids = [a,a,b,b] # must be same length as A
    all_rowlabels = [ 0, 0, 0, 0, 1,1, 2, 3,3,3,3,3,3,3] # one element for every element of all orig_data
    #               [ a, a, a, a, a,a, b, b,b,b,b,b,b,b]
    orig_row_offset_by_trace_id = {a:0, # of all_rowlabels into trace_id specific elements
                                   b:6}
    d = DataAssoc(all_rowlabels, trace_ids, orig_data_by_trace_id, orig_row_offset_by_trace_id)

    rowlabels_a = d.get_rowlabels_for_trace_id(a)
    print 'rowlabels_a',rowlabels_a
    assert np.allclose( rowlabels_a, [ 0, 0, 0, 0, 1,1 ] )

    rowlabels_b = d.get_rowlabels_for_trace_id(b)
    print 'rowlabels_b',rowlabels_b
    assert np.allclose( rowlabels_b, [ 2, 3,3,3,3,3,3,3] )

    # get all data from a
    results1 = d.idxs_by_trace_id_for_Aidxs( [0,1] )
    assert results1.keys() == [a]
    assert np.allclose(results1[a], [0,1,2,3, 4,5] )

    # check element view
    results1 = d.elems_by_trace_id_for_Aidxs( [0,1] )
    assert results1.keys() == [a]
    assert np.allclose(results1[a], [ 0.1, 0.2, 0.3, 0.4, 1.1, 1.2 ] )

    # get some data from a
    results1 = d.idxs_by_trace_id_for_Aidxs( [0] )
    assert results1.keys() == [a]
    assert np.allclose(results1[a], [0,1,2,3] )

    # get all data from a
    results1 = d.idxs_by_trace_id_for_Aidxs( [1] )
    assert results1.keys() == [a]
    assert np.allclose(results1[a], [4,5] )



    # get all data from b
    results2 = d.idxs_by_trace_id_for_Aidxs( [2,3] )
    assert results2.keys() == [b]
    assert np.allclose(results2[b], [0,1,2,3,4,5,6,7] )

    # check element view
    results2 = d.elems_by_trace_id_for_Aidxs( [2,3] )
    assert results2.keys() == [b]
    assert np.allclose(results2[b], [ 2.1, 3.1, 3.2, 3,3,3,3,3])

    # get some data from b
    results2 = d.idxs_by_trace_id_for_Aidxs( [2] )
    assert results2.keys() == [b]
    assert np.allclose(results2[b], [0])

    # get some data from b
    results2 = d.idxs_by_trace_id_for_Aidxs( [3] )
    assert results2.keys() == [b]
    assert np.allclose(results2[b], [1,2,3,4,5,6,7] )

    # get empty
    results3 = d.idxs_by_trace_id_for_Aidxs( [] )
    assert len(results3.keys())==0

def load_A_matrix( options=None ):

    if options.obj_only is not None:
        raise ValueError('obj_only is not a valid option for this function')

    kalman_rows, fps, stim_xml, saccade_results = posts.read_files_and_fuse_ids(options=options)
    trace_id = options.kalman_filename
    results_recarray = posts.calc_retinal_coord_array(kalman_rows, fps, stim_xml)

    # pretend (for now) that we have lots of data files:
    orig_data_by_trace_id = {trace_id:results_recarray} # keep organized

    all_A = []
    all_rowlabels = []
    trace_ids = []
    rowlabels_inc = 0
    orig_row_offset_by_trace_id = {}
    for trace_id,results_recarray in orig_data_by_trace_id.iteritems():
        A, rowlabels, A_names = create_analysis_array( results_recarray,
                                                       frames_per_second=fps,
                                                       skip_missing=True, # temporary
                                                       )
        all_A.append(A)
        rowlabels += rowlabels_inc
        all_rowlabels.append(rowlabels)
        orig_row_offset_by_trace_id[trace_id] = rowlabels_inc
        rowlabels_inc += len(A)
        trace_ids.extend( [trace_id]*len(A) )

    # combine all observations into one giant array
    A=np.vstack(all_A)
    all_rowlabels = np.hstack(all_rowlabels)

    data = DataAssoc(all_rowlabels, trace_ids, orig_data_by_trace_id, orig_row_offset_by_trace_id)
    results = dict(A=A,
                   data=data,
                   A_names=A_names,
                   all_rowlabels=all_rowlabels,
                   orig_data_by_trace_id=orig_data_by_trace_id,
                   stim_xml=stim_xml,
                   )
    return results, saccade_results

def doit(options=None):
    my_results, saccade_results =load_A_matrix( options )

    A=my_results['A']
    data=my_results['data']
    A_names=my_results['A_names']
    all_rowlabels=my_results['all_rowlabels']
    if 0:
        # convert from ma to nan
        all_rowlabels_data = np.ma.getdata( all_rowlabels )
        all_rowlabels_mask = np.ma.getmask( all_rowlabels )
        all_rowlabels_data[all_rowlabels_mask] = np.nan
        all_rowlabels = all_rowlabels_data

    orig_data_by_trace_id=my_results['orig_data_by_trace_id']
    stim_xml=my_results['stim_xml']

    print 'normalizing array'
    normA,norm_info = normalize_array(A)
    U,s,Vh = np.linalg.svd(normA,full_matrices=False)


    DO_KMEANS = True

    #MOTOR_CLUSTER_ONLY = True # ignore sensory inputs when computing clusters, and don't use SVD outputs for clustering
    MOTOR_CLUSTER_ONLY = False

    if DO_KMEANS:
        if MOTOR_CLUSTER_ONLY:
            n_vals = 2
            cluster_mat = normA[:,:n_vals]
        else:
            n_vals_to_skip = 2
            n_vals = U.shape[1]-n_vals_to_skip
            print 'with %d of %d top SVs'%(n_vals,U.shape[1])
            cluster_mat = U[:,:n_vals]

        n_clusters = 5
        numpy.random.seed(3)
        print 'doing kmeans..',
        sys.stdout.flush()
        code_book, cluster_labels = scipy.cluster.vq.kmeans2(cluster_mat, n_clusters, iter=2000, minit='random')
        print 'done'


    if 1:
        # print summary information
        print 'A names:'
        for i,A_name in enumerate(A_names):
            print '  %d: %s'%(i,A_name)

        print np.set_printoptions(linewidth=150,suppress=True)
        print 'array size',normA.shape
        print 'U.shape',U.shape
        print 'U[:10]'
        print U[:10]
        print 's'
        print s
        print 'Vh.shape',Vh.shape
        print 'Vh'
        print Vh

        if 1:
            # show reconstruction with reduced dimensionality
            n_vals_show = 7
            n_rows = 5

            normA_test = np.dot( U[:n_rows, :n_vals_show], np.dot( np.diag(s[:n_vals_show]), Vh[:n_vals_show,:] ) )

            norm_A_actual = normA[:n_rows]

            print 'normA_test'
            print normA_test
            print 'norm_A_actual'
            print norm_A_actual

    if 0:
        # plot non-contiguous timeseries of all data in A matrix and original data. skips missing data
        import matplotlib.pyplot as plt

        orig_col_names = ['x','y','z']
        A_col_nums = [0,1,2]
        U_col_nums = [0,1,2]

        accum = collections.defaultdict(list)
        for i in range(len(U)):
            rows_by_trace_id = data.elems_by_trace_id_for_Aidxs( [i] )
            for trace_id, rows in rows_by_trace_id.iteritems():
                for orig_col_name in orig_col_names:
                    accum[orig_col_name].append( rows[orig_col_name] )
                for A_col_num in A_col_nums:
                    A_col_name = A_names[A_col_num]
                    accum[A_col_name].append( A[i,A_col_num] * np.ones( (len(rows),)) )
                for U_col_num in U_col_nums:
                    U_col_name = 'U%d'%U_col_num
                    accum[U_col_name].append( U[i,U_col_num] * np.ones( (len(rows),)) )

        subplot_keys = accum.keys()
        subplot_keys.sort()
        for i, subplot_key in enumerate(subplot_keys):
            accum[subplot_key] = np.ma.hstack( accum[subplot_key] )
        if 1:
            del subplot_keys[subplot_keys.index('x')]
            del subplot_keys[subplot_keys.index('y')]
            del subplot_keys[subplot_keys.index('z')]
            subplot_keys.insert(0,'z')
            subplot_keys.insert(0,'y')
            subplot_keys.insert(0,'x')
        N_subplots = len(subplot_keys)
        fig = plt.figure()
        ax = None
        for i, subplot_key in enumerate(subplot_keys):
            ax = fig.add_subplot(N_subplots,1,1+i,sharex=ax)
            line, = ax.plot( accum[subplot_key], '.' )

            ax.text(0,1,subplot_key,
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    )

        plt.show()

    if 0:
        # plot timeseries of all original data and corresponding A matrix (time-contiguous)
        import matplotlib.pyplot as plt

        orig_col_names = ['x','y','z']
        A_col_nums = [0,1,2,3,4,7,9]
        U_col_nums = []#0,1,2]

        accum = collections.defaultdict(list)
        for trace_id,rows in orig_data_by_trace_id.iteritems():
            this_rowlabels = data.get_rowlabels_for_trace_id(trace_id)
            assert len(this_rowlabels) == len(rows)

            this_A = mytake( A, this_rowlabels )
            this_U = mytake( U, this_rowlabels )

            for orig_col_name in orig_col_names:
                accum[orig_col_name].append( rows[orig_col_name] )
            for A_col_num in A_col_nums:
                A_col_name = A_names[A_col_num]
                accum[A_col_name].append( this_A[:,A_col_num] * np.ones( (len(rows),)) )
            for U_col_num in U_col_nums:
                U_col_name = 'U%d'%U_col_num
                accum[U_col_name].append( this_U[:,U_col_num] * np.ones( (len(rows),)) )

        subplot_keys = accum.keys()
        subplot_keys.sort()
        for i, subplot_key in enumerate(subplot_keys):
            accum[subplot_key] = np.ma.hstack( accum[subplot_key] )
        if 1:
            del subplot_keys[subplot_keys.index('x')]
            del subplot_keys[subplot_keys.index('y')]
            del subplot_keys[subplot_keys.index('z')]
            subplot_keys.insert(0,'z')
            subplot_keys.insert(0,'y')
            subplot_keys.insert(0,'x')
        N_subplots = len(subplot_keys)
        fig = plt.figure()
        ax = None
        for i, subplot_key in enumerate(subplot_keys):
            ax = fig.add_subplot(N_subplots,1,1+i,sharex=ax)
            line, = ax.plot( accum[subplot_key], '.' )

            from matplotlib.transforms import offset_copy
            transOffset = offset_copy(ax.transData, fig=fig,
                                      x = 5, y=5, units='dots')

            ax.text(0,1,subplot_key,
                    transform=ax.transAxes,
                    #transform=transOffset,
                    horizontalalignment='left',
                    verticalalignment='top',
                    )

#        plt.show()


    if 0:
        # top and side views colored by attributes

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        arrnames = ['A','A','A', 'A', 'A']# 'U','U','U']
        cols = [ 0,1,2, 3, 7,]# 0,1,2 ]

        ax = None
        for arrname, j in zip(arrnames,cols):
            if arrname == 'A':
                arr = A
                col_name = A_names[j]
            elif arrname == 'U':
                arr = U
                col_name = 'U%d'%j
            col = arr[:,j]
            all_rows_idx = np.arange(len(col))

            x = []
            y = []
            z = []
            c = []
            post_angle_x = []
            post_angle_y = []
            post_speed = []

            for i in range(len(col)):
                rows_by_trace_id = data.elems_by_trace_id_for_Aidxs( [i] )
                for trace_id, rows in rows_by_trace_id.iteritems():
                    x.append( rows['x'] )
                    y.append( rows['y'] )
                    z.append( rows['z'] )
                    c.append( [arr[i,j]]*len(rows) )
                    post_speed.append( [A[i,A_names.index('speed from closest post (m/sec)')]]*len(rows) )
                    post_angle_x.append( [A[i,A_names.index('angle to post (X)')]]*len(rows) )
                    post_angle_y.append( [A[i,A_names.index('angle to post (Y)')]]*len(rows) )
            x = np.hstack(x)
            y = np.hstack(y)
            z = np.hstack(z)
            c = np.hstack(c)
            post_speed = np.hstack(post_speed)
            post_angle_x = np.hstack(post_angle_x)
            post_angle_y = np.hstack(post_angle_y)

            post_angle = make_branch_cut_pi(np.arctan2( post_angle_y, post_angle_x ))

            #approaching = post_speed < 0
            D2R = np.pi/180.0
            #approaching = (post_speed < 0) & ( abs(post_angle) < 30*D2R ) # decreasing distance and heading with 30 degrees of post center
            approaching = abs(post_angle) < 30*D2R # heading with 30 degrees of post center

            fig = plt.figure()

            # X vs Z
            ax = fig.add_axes( (0.05, 0.65, .8, .3), sharex=ax)
            ax.set_title(col_name)
            ax.set_aspect('equal')
            collection = ax.scatter(x[~approaching],z[~approaching],c=c[~approaching],edgecolors='none',s=2)
            ax.scatter(x[approaching],z[approaching],c=c[approaching],edgecolors='none',s=10,
                       cmap=collection.cmap,
                       norm=collection.norm)
            stim_xml.plot_stim(ax, projection=xml_stimulus.SimpleOrthographicXZProjection() )

            # X vs Y
            ax = fig.add_axes( (0.05, 0.05, .8, .55),sharex=ax )
            ax.scatter(x[~approaching],y[~approaching],c=c[~approaching],edgecolors='none',s=2,
                       cmap=collection.cmap,
                       norm=collection.norm)
            ax.scatter(x[approaching],y[approaching],c=c[approaching],edgecolors='none',s=10,
                       cmap=collection.cmap,
                       norm=collection.norm)
            stim_xml.plot_stim(ax, projection=xml_stimulus.SimpleOrthographicXYProjection() )
            ax.set_aspect('equal')

            cax = fig.add_axes( (0.9, 0.05, .02, .9))
            cbar = fig.colorbar(collection, cax=cax, ax=ax )

#        plt.show()

    if 0:
        # plot dist vs velocity for post approach
        import matplotlib.pyplot as plt

        post_dist_column = A_names.index('distance to closest post (m)')
        post_speed_column = A_names.index('speed from closest post (m/sec)')
        post_accel_column = A_names.index('accel from closest post (m/sec/sec)')

        post_dist = A[:,post_dist_column]
        post_speed = A[:,post_speed_column]
        post_accel = A[:,post_accel_column]
        horiz_vel = A[:,A_names.index('horizontal velocity (m/sec)')]
        #post_angle = A[:,A_names.index('angle to post (rad)')]
        post_angle_x = A[:,A_names.index('angle to post (X)')]
        post_angle_y = A[:,A_names.index('angle to post (Y)')]
        post_angle = make_branch_cut_pi(np.arctan2( post_angle_y, post_angle_x ))

        speed = A[:,A_names.index('speed (m/sec)')]
        Arow = np.arange(len(A))

        #approaching = post_speed < 0
        approaching = abs(post_angle) < 30*D2R # heading with 30 degrees of post center
        this_Arow = Arow[approaching]
        contig_chunk_idxs = utils.get_contig_chunk_idxs( this_Arow )
        if 1:
            # only take sequences with > 1 observation
            contig_chunk_idxs = [ tup for tup in contig_chunk_idxs if (tup[1]-tup[0])>1 ]
        tup2line = {}

        fig = plt.figure()

        ax = fig.add_subplot(3,1,1)
        for tup in contig_chunk_idxs:
            this_start,this_stop = tup
            astart = this_Arow[this_start]
            astop = this_Arow[this_stop-1]+1
            line,=ax.plot( post_dist[astart:astop], post_speed[astart:astop], 'o-' )
            tup2line[tup]=line
        ax.set_xlabel('distance to closest post (m)')
        ax.set_ylabel('speed to closest post (m/sec)')

        ax = fig.add_subplot(3,1,2,sharex=ax)
        for tup in contig_chunk_idxs:
            line = tup2line[tup]
            this_start,this_stop = tup
            astart = this_Arow[this_start]
            astop = this_Arow[this_stop-1]+1
            ax.plot( post_dist[astart:astop], post_accel[astart:astop], 'o-', color=line.get_color()  )
        ax.set_xlabel('distance to closest post (m)')
        ax.set_ylabel('accel from closest post (m/sec/sec)')

        ax = fig.add_subplot(3,1,3,sharex=ax)
        for tup in contig_chunk_idxs:
            line = tup2line[tup]
            this_start,this_stop = tup
            astart = this_Arow[this_start]
            astop = this_Arow[this_stop-1]+1

            a_pred = (post_speed[astart:astop])**2/(2*post_dist[astart:astop])

            ax.plot( post_dist[astart:astop], a_pred, 'o-', color=line.get_color()  )
        ax.set_xlabel('distance to closest post (m)')
        ax.set_ylabel("Larry's model predicted accel (m/sec/sec)")


        if 0:
            ax = fig.add_subplot(4,1,4,sharex=ax)
            for tup in contig_chunk_idxs:
                line = tup2line[tup]
                this_start,this_stop = tup
                astart = this_Arow[this_start]
                astop = this_Arow[this_stop-1]+1
                ax.plot( post_dist[astart:astop], horiz_vel[astart:astop], 'o-', color=line.get_color() )
            ax.set_xlabel('distance to closest post (m)')
            ax.set_ylabel('horizontal velocity (m/sec)')

            ax = fig.add_subplot(4,1,5,sharex=ax)
            for tup in contig_chunk_idxs:
                line = tup2line[tup]
                this_start,this_stop = tup
                astart = this_Arow[this_start]
                astop = this_Arow[this_stop-1]+1
                ax.plot( post_dist[astart:astop], speed[astart:astop], 'o-', color=line.get_color() )
            ax.set_xlabel('distance to closest post (m)')
            ax.set_ylabel('speed (m/sec)')


        plt.show()

    if PICK:
        global line2data
        global ax_xy, ax_xz
        global all_figs

        ax_xy = {}
        ax_xz = {}

        import matplotlib.pyplot as plt

        for trace_id,kalman_rows in orig_data_by_trace_id.iteritems():

            fig=plt.figure()
            all_figs.append(fig)

            ax_xy[trace_id] = fig.add_subplot(2,1,1)
            ax_xy[trace_id].plot(kalman_rows['x'],kalman_rows['y'],'b.',ms=.5)
            ax_xy[trace_id].set_aspect('equal')

            ax_xz[trace_id] = fig.add_subplot(2,1,2)
            ax_xz[trace_id].plot(kalman_rows['x'],kalman_rows['z'],'b.',ms=.5)
            ax_xz[trace_id].set_aspect('equal')

        if 1:
            fig=plt.figure()
            all_figs.append(fig)

            ax = fig.add_subplot(2,2,1)
            line,=ax.plot( U[:,0], U[:,1], '.', picker=5)
            line2data[line]=data,ax
            ax.set_aspect('equal')
            plt.xlabel('U0')
            plt.ylabel('U1')

            ax = fig.add_subplot(2,2,2)
            line,=ax.plot( U[:,2], U[:,1], '.', picker=5)
            line2data[line]=data,ax
            ax.set_aspect('equal')
            plt.xlabel('U2')
            plt.ylabel('U1')

            ax = fig.add_subplot(2,2,3)
            line,=ax.plot( U[:,0], U[:,2], '.', picker=5)
            line2data[line]=data,ax
            ax.set_aspect('equal')
            plt.xlabel('U0')
            plt.ylabel('U2')
            fig.canvas.mpl_connect('pick_event', onpick1)
            print 'connected events'

        if 1:
            N = Vh.shape[1]
            fig=plt.figure()
            all_figs.append(fig)

            ind = np.arange(N)
            width = 0.2
            if 0:
                Vh = Vh.T
                print 'plotting columns of Vh'
            rects1 = plt.bar(ind, Vh[0,:], width, color='r')
            rects2 = plt.bar(ind+width, Vh[1,:], width, color='y')
            rects3 = plt.bar(ind+2*width, Vh[2,:], width, color='g')

            plt.ylabel('Factor')
            locs, barlabels = plt.xticks(ind+width, A_names )
            plt.setp(barlabels, 'rotation', 'vertical')
            plt.legend( (rects1[0], rects2[0], rects3[0]), ('Vh0 (s %.1f)'%s[0],
                                                            'Vh1 (s %.1f)'%s[1],
                                                            'Vh2 (s %.1f)'%s[2]) )

    if DO_KMEANS:

        # http://jfly.iam.u-tokyo.ac.jp/color/index.html
        NSFlabel2color = {None:(0,0,0,1), # black

                          0:(.8,.4,0,1), # vermillion ( very right )
                          1:(.35,.7,.9,1), # sky blue ( left )
                          2:(.9,.6,0,1), # orange (right )
                          3:(.95,.9,.25,1), # yellow ( hover )
                          4:(0,.45,.7,1), # blue ( very left )
                          5:(.8,.6,.7,1), # reddish purple (backward)
                          6:(0,.6,.6,1), # bluish green (forward)

                          7:(1,0,0,1), # red
                          8:(0,1,0,1), # green
                          9:(0,0,1,1), # blue
                          }

        if 1:
            # plot clusters in U space
            n_attribs = n_vals
            import matplotlib.pyplot as plt
            fig=plt.figure()

            ax_by_ij={}
            for i in range(n_attribs):
                for j in range(n_attribs):
                    if i>=j:
                        continue

                    ax = fig.add_subplot(n_attribs-1,n_attribs-1, (i*(n_attribs-1)) + j )
                    ax_by_ij[ (i,j) ] = ax

                    for this_cluster_label, centroid in enumerate(code_book):
                        U_picker = cluster_labels==this_cluster_label
                        line,=ax.plot( cluster_mat[U_picker,i], cluster_mat[U_picker,j], '.', color=NSFlabel2color[this_cluster_label] )

                        ax.set_aspect('equal')
                        if MOTOR_CLUSTER_ONLY:
                            ax.set_xlabel(A_names[i])
                            ax.set_ylabel(A_names[j])
                        else:
                            ax.set_xlabel('U%d'%i)
                            ax.set_ylabel('U%d'%j)

            # plot centroid
            for i in range(n_attribs):
                for j in range(n_attribs):
                    if i>=j:
                        continue
                    ax = ax_by_ij[ (i,j) ]

                    for centroid in code_book:
                        # centroid
                        if i>=len(centroid) or j>=len(centroid):
                            # can't plot centroid -- does not exist on this axis
                            continue
                        ax.plot( [centroid[i]], [centroid[j]], 'o',
                                 #color='k',
                                 mec='k',
                                 mfc='None',)
        if 1:
            # plot top and side views by cluster
            for trace_id,rows in orig_data_by_trace_id.iteritems():
                this_rowlabels = data.get_rowlabels_for_trace_id(trace_id)
                assert len(this_rowlabels) == len(rows)

                this_cluster_mat = mytake( cluster_mat, this_rowlabels )
                this_cluster_labels = mytake( cluster_labels, this_rowlabels )

                fig=plt.figure()
                ax = fig.add_subplot(1,1,1)

                for this_cluster_label in range(n_clusters):
                    color = NSFlabel2color[this_cluster_label]

                    cond = this_cluster_labels==this_cluster_label
                    ax.plot( rows['x'][cond], rows['y'][cond], '.', color=color )

                stim_xml.plot_stim(ax, projection=xml_stimulus.SimpleOrthographicXYProjection() )

                ax.plot( saccade_results['X'][:,0],
                         saccade_results['X'][:,1],
                         'rx', ms=10 )

                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_aspect('equal')

        if 1:
            # plot timeseries by cluster
            for trace_id,rows in orig_data_by_trace_id.iteritems():
                this_rowlabels = data.get_rowlabels_for_trace_id(trace_id)
                assert len(this_rowlabels) == len(rows)

                this_cluster_mat = mytake( cluster_mat, this_rowlabels ) # becomes as long as rows
                this_cluster_labels = mytake( cluster_labels, this_rowlabels ) # becomes as long as rows
                assert len(this_cluster_labels) == len(this_rowlabels)
                fig=plt.figure()

                ax = fig.add_subplot(3,1,1)
                if 1:
                    good_cond = ~np.isnan(this_cluster_labels)
                    good_labels = this_cluster_labels[good_cond]
                    c = [NSFlabel2color[i] for i in good_labels]
                    ax.scatter( rows['frame'][good_cond], this_cluster_labels[good_cond], c=c, edgecolors='none', s=3)
                else:
                    ax.plot( rows['frame'], this_cluster_labels, '.', color='k' )

                ax = fig.add_subplot(3,1,2,sharex=ax)
                for this_cluster_label in range(n_clusters):
                    color = NSFlabel2color[this_cluster_label]
                    cond = this_cluster_labels==this_cluster_label
                    ax.plot( rows['frame'][cond], rows['x'][cond], '.', color=color )
                for saccade_frame in saccade_results['frames']:
                    ax.axvline( saccade_frame )
                ax.set_ylabel('X (m)')

                ax = fig.add_subplot(3,1,3,sharex=ax)
                for this_cluster_label in range(n_clusters):
                    color = NSFlabel2color[this_cluster_label]
                    cond = this_cluster_labels==this_cluster_label
                    ax.plot( rows['frame'][cond], rows['z'][cond], '.', color=color )
                ax.set_ylabel('Z (m)')

    plt.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    analysis_options.add_common_options( parser )
    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit( options=options )

if __name__=='__main__':
    if 0:
        results = main()
    else:
        print 'testing'
        test_data_assoc_1()

