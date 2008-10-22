from __future__ import division
from __future__ import with_statement
import pkg_resources
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, time

import numpy
import numpy as np
from optparse import OptionParser
import flydra.a2.core_analysis as core_analysis
import flydra.a2.flypos
import flydra.a2.analysis_options as analysis_options

import flydra.a2.posts as posts

def angle_diff(ang1,ang2):
    return np.mod((ang1-ang2)+np.pi,2*np.pi)-np.pi

def test_angle_diff():
    ang1 = np.array([np.pi-0.001, -0.001,  0.001, np.pi+0.001])
    ang2 = np.array([np.pi+0.001,  0.001, -0.001, np.pi-0.001])
    actual = angle_diff(ang1,ang2)
    expected = np.array([-0.002, -0.002, 0.002, 0.002])
    #print 'actual',actual
    #print 'expected',expected

def get_horiz_turns( vx,vy, subsample_factor=20, frames_per_second=None):
    """return angular velocity of velocity direction in rad/sec"""
    N_observations = len(vx)//subsample_factor
    horiz_turns = []
    horiz_vel_angle = np.arctan2( vy, vx )
    d_angles = angle_diff(horiz_vel_angle[1:],horiz_vel_angle[:-1])
    for i in range( N_observations ):
        start = i*subsample_factor
        stop = (i+1)*subsample_factor
        total_angular_change = np.sum(  d_angles[start:stop] )
        whole_dt = 1.0/frames_per_second * subsample_factor
        vel_angular_rate = total_angular_change/whole_dt
        horiz_turns.append( vel_angular_rate ) # rad/sec
    horiz_turns = np.array( horiz_turns )
    return horiz_turns

def create_analysis_array( rec, subsample_factor=20, frames_per_second=None ):
    """from densely-spaced input data, create analysis array

    array has rows that are observations and columns that are features.

    Inputs
    ======
    rec - record array of densely-spaced input

    Outputs
    =======
    A - an MxN array of M observations over N features
    rowlabels - an array the length of rec labeling the corresponding row of A
    """
    rowlabels = np.empty( (len(rec),) );
    rowlabels.fill(np.nan)

    rowlabels_non_nan = np.arange( len(rec) // subsample_factor )
    rowlabels_non_nan = rowlabels_non_nan.repeat( subsample_factor )

    start_idx = 1 # skip nan at start
    stop_idx = len(rowlabels_non_nan)+1
    if stop_idx==len(rowlabels):
        stop_idx -= subsample_factor # skip nan at stop
    rowlabels[start_idx:stop_idx] = rowlabels_non_nan

    vx = rec['vel_x'][start_idx:stop_idx]
    vy = rec['vel_y'][start_idx:stop_idx]
    horizontal_angular_velocity = get_horiz_turns( vx,vy, subsample_factor=subsample_factor,
                                                   frames_per_second=frames_per_second)

    DEBUG=False
    if DEBUG:
        N_observations = len(horiz_turns)
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
                ax.text(xi[0],yi[0],'turn: %.1f'%(horiz_turns[i],))
                ax.set_aspect('equal')
                fig.savefig('debug%05d.png'%i)
                pylab.close(fig)

    closest_dist = np.ma.array(rec[ 'closest_dist' ],mask=rec[ 'closest_dist_mask' ])
    angle_of_closest_dist = np.ma.array(rec[ 'angle_of_closest_dist' ],mask=rec[ 'closest_dist_mask' ])
    post_angle = angle_of_closest_dist[start_idx:stop_idx]
    try:
        sin_post_angle = np.sin( post_angle )
    except:
        print 'post_angle'
        print post_angle
        raise

    def downsamp(arr):
        N_observations = len(arr)//subsample_factor
        x = np.ma.reshape(arr, (N_observations,subsample_factor))
        x = np.ma.mean(x,axis=1)
        return x

    A = [ horizontal_angular_velocity,
          downsamp(rec['vel_horiz'][start_idx:stop_idx]),
          downsamp(rec['vel_z'][start_idx:stop_idx]),
          downsamp(closest_dist[start_idx:stop_idx]), # NL func or binarize on this?
          downsamp(post_angle),
          downsamp(sin_post_angle),
          ]
    A_names = ['angular velocity about Z axis (rad/sec)',
               'horizontal velocity (m/sec)',
               'vertical velocity (m/sec)',
               'distance to closest post (m)',
               'angle to post (rad)',
               'sin(angle to post)',
               ]
    A = [np.array(ai) for ai in A]
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
    A = np.array(A).T # observations = rows, attributes = columns
    return A, rowlabels, A_names

def normalize_array(A):
    column_means = np.mean(A,axis=0)
    column_std = np.std(A,axis=0)
    norm_info = {'means':column_means,
                 'std':column_std}
    normA = (A - column_means)/column_std
    return normA,norm_info


def doit(options=None):
    if options.obj_only is not None:
        raise ValueError('obj_only is not a valid option for this function')

    kalman_rows, fps, stim_xml = posts.read_files_and_fuse_ids(options=options)
    results_recarray = posts.calc_retinal_coord_array(kalman_rows, fps, stim_xml)
    # pretend (for now) that we have lots of data files:
    results_recarrays = [results_recarray]
    all_A = []
    for results_recarray in results_recarrays:
        A, rowlabels, A_names = create_analysis_array( results_recarray,
                                                       frames_per_second=fps )
        all_A.append(A)

    # combine all observations into one giant array
    print np.set_printoptions(linewidth=150,suppress=True)
    A=np.vstack(all_A)
    normA,norm_info = normalize_array(A)
    print 'array size',normA.shape
    U,s,Vh = np.linalg.svd(normA,full_matrices=False)
    print 'U.shape',U.shape
    print 'U[:10]'
    print U[:10]
    print 's'
    print s
    print 'Vh.shape',Vh.shape
    print 'Vh'
    print Vh

    if 1:
        import matplotlib.pyplot as plt
        if 1:
            fig=plt.figure()

            ax = fig.add_subplot(2,2,1)
            ax.plot( U[:,0], U[:,1], '.')
            ax.set_aspect('equal')
            plt.xlabel('U0')
            plt.ylabel('U1')

            ax = fig.add_subplot(2,2,2)
            ax.plot( U[:,2], U[:,1], '.')
            ax.set_aspect('equal')
            plt.xlabel('U2')
            plt.ylabel('U1')

            ax = fig.add_subplot(2,2,3)
            ax.plot( U[:,0], U[:,2], '.')
            ax.set_aspect('equal')
            plt.xlabel('U0')
            plt.ylabel('U2')

        if 1:
            N = Vh.shape[1]
            plt.figure()
            ind = np.arange(N)
            width = 0.2
            if 1:
                Vh = Vh.T
                print 'plotting columns of Vh'
            rects1 = plt.bar(ind, Vh[0,:], width, color='r')
            rects2 = plt.bar(ind+width, Vh[1,:], width, color='y')
            rects3 = plt.bar(ind+2*width, Vh[2,:], width, color='g')

            plt.ylabel('Factor')
            locs, labels = plt.xticks(ind+width, A_names )
            plt.setp(labels, 'rotation', 'vertical')
            plt.legend( (rects1[0], rects2[0], rects3[0]), ('Vh0 (s %.1f)'%s[0],
                                                            'Vh1 (s %.1f)'%s[1],
                                                            'Vh2 (s %.1f)'%s[2]) )

        plt.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    analysis_options.add_common_options( parser )
    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit( options=options,
         )

if __name__=='__main__':
    main()

