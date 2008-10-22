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
import tables as PT
from optparse import OptionParser
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.core_analysis as core_analysis
import flydra.a2.analysis_options as analysis_options
import flydra.analysis.result_utils as result_utils
import flydra.a2.flypos
import flydra.geom as geom
import pylab

R2D = 180/numpy.pi

def calc_circle_tangent_points(radius,xy):
    """return endpoints of circle at origin as seen from points xy

The circle below has origin at (0,0) and radius r. It is distance h
from another point. Define d as the distance between the point and the
intersection of the tangent lines of the circle that meet the point. r
and d form a right triangle with angle a between h and r. Thus, cos(a)
== (r/h) and therefore a==acos(r/h).

This function returns the intersection of these tangent lines with the
circle, and thus are the endpoints of the visible region of the circle
from the location of the point.


                -------------
           ----/             \/------
         -/                  /    \- \-----
       -/                   /       \-     \-----
     -/                    /          \-         \------    d
    /                     |             \               \-----
   /                      / r            \                    \-----
  /                      /                \                         \------
 /                      /                  \                               \-----
 |                     / a                 |              h                      \-----
 |                    X--------------------+-------------------------------------------X---
 |                     \ a                 |                                     /-----
 \                      \                  /                               /-----
  \                     |                 /                          /-----
   \                     \  r            /                     /-----
    \                     \             /               /------
     -\                    \          /-          /-----    d
       -\                   \       /-      /-----
         -\                 |     /-  /-----
           ----\             X--/-----
                --------------\-


>>> xy = numpy.array([[  2,   0],
...        [  0,   2],
...        [ 10,   0],
...        [100,   2]])

>>> calc_circle_tangent_points(1.0,xy)
(array([[ 0.5       , -0.8660254 ],
       [ 0.8660254 ,  0.5       ],
       [ 0.1       , -0.99498744],
       [ 0.029991  , -0.99955017]]), array([[ 0.5       ,  0.8660254 ],
       [-0.8660254 ,  0.5       ],
       [ 0.1       ,  0.99498744],
       [-0.009999  ,  0.99995001]]))

    """
    r = radius
    xy = numpy.atleast_2d(xy)
    x = xy[:,0]
    y = xy[:,1]
    h = numpy.hypot(x,y)
    a = numpy.arccos( r/h ) # radians
    offset = numpy.angle( x+y*1j )

    # now calculate tangent points
    abs_angle_a = offset-a
    pt_a_x = r*numpy.cos(abs_angle_a)
    pt_a_y = r*numpy.sin(abs_angle_a)
    pt_a = numpy.hstack( (pt_a_x[:,numpy.newaxis],pt_a_y[:,numpy.newaxis]) )

    abs_angle_b = offset+a
    pt_b_x = r*numpy.cos(abs_angle_b)
    pt_b_y = r*numpy.sin(abs_angle_b)
    pt_b = numpy.hstack( (pt_b_x[:,numpy.newaxis],pt_b_y[:,numpy.newaxis]) )
    return pt_a, pt_b

def calc_closest_points(radius,xy):
    """return closest point of circle at origin as seen from points xy"""
    r = radius
    xy = numpy.atleast_2d(xy)
    x = xy[:,0]
    y = xy[:,1]
    xy_mag = numpy.hypot(x,y)
    xy_dir = xy / xy_mag[:,np.newaxis]
    pt_c = r*xy_dir
    return pt_c

def intersect_line_with_iso_z_plane( verts, z_array ):
    """find the points on a line with specified Z values

>>> va = numpy.array([1.0,0.0,0.0])

>>> vb = numpy.array([1.0,0.0,1.0])

>>> z_array = numpy.array([-1,-0.5,0, 0.33,1,10])

>>> intersect_line_with_iso_z_plane( [va,vb], z_array )
array([[  1.  ,   0.  ,  -1.  ],
       [  1.  ,   0.  ,  -0.5 ],
       [  1.  ,   0.  ,   0.  ],
       [  1.  ,   0.  ,   0.33],
       [  1.  ,   0.  ,   1.  ],
       [  1.  ,   0.  ,  10.  ]])

    """
    assert len(verts)==2
    fill_later = 0

    # See "parametric form" at:
    # http://en.wikipedia.org/w/index.php?title=Line-plane_intersection&oldid=202894088
    #
    # In that notation, we have a plane defined by three points:
    # p_0 = (0,0,z)
    p_0 = numpy.array([0,0,fill_later],dtype=float)
    # p_1 = (1,0,z)
    # p_2 = (0,1,z)
    #
    # and a line defined by two points:
    # l_a and l_b
    l_a, l_b = verts

    mat_to_invert = numpy.array([[fill_later, 1, 0],
                                 [fill_later, 0, 1],
                                 [fill_later, 0, 0]], dtype=float)
    mat_to_invert[:,0] = l_a-l_b
    mat_inverted = numpy.linalg.inv( mat_to_invert )

    results = []
    for i in range(len(z_array)):
        p_0[2] = z_array[i]
        vec = l_a-p_0
        tuv = numpy.dot( mat_inverted, vec )
        intersection = l_a + (l_b-l_a)*tuv[0]
        results.append(intersection)
    results = numpy.array(results)
    return results

def plot_angle_dist(subplot=None,options=None,bad=False):
    """

    bad - if false, show only points below max_post_z, if true, show
          only points above

    Note: this assumes that posts are perfectly vertical in the world
    coordinate system and that flies only look in a level plane.

    """
    assert options is not None
    assert options.stim_xml is not None, 'you must specify a stimulus .xml file'

    ca = core_analysis.get_global_CachingAnalyzer()

    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(options.kalman_filename)

    fps = result_utils.get_fps( data_file )

    if 1:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    if 1:
        file_timestamp = data_file.filename[4:19]
        fanout = xml_stimulus.xml_fanout_from_filename( options.stim_xml )
        include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp( timestamp_string=file_timestamp )
        walking_start_stops = fanout.get_walking_start_stops_for_timestamp( timestamp_string=file_timestamp )
        if include_obj_ids is not None:
            use_obj_ids = include_obj_ids
        if exclude_obj_ids is not None:
            use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )
        stim_xml = fanout.get_stimulus_for_timestamp(timestamp_string=file_timestamp)

    kalman_rows = flydra.a2.flypos.fuse_obj_ids(use_obj_ids, data_file,
                                                dynamic_model_name = dynamic_model,
                                                frames_per_second=fps)
    frame = kalman_rows['frame']
    if (options.start is not None) or (options.stop is not None):
        valid_cond = numpy.ones( frame.shape, dtype=numpy.bool )
        if options.start is not None:
            valid_cond &= (frame >= options.start)
        if options.stop is not None:
            valid_cond &= (frame <= options.stop)

        kalman_rows = kalman_rows[valid_cond]
    frame = kalman_rows['frame']
    X = numpy.array( [kalman_rows['x'],
                      kalman_rows['y'],
                      kalman_rows['z']]).T

    if 0:
        tmp_frame = frame - frame[0]
        # tmp print
        tmp_cond = (tmp_frame > (210*fps)) & (tmp_frame < (220*fps))
        print 'np.sum(tmp_cond)',np.sum(tmp_cond)
        print "kalman_rows['z'][tmp_cond]"
        print kalman_rows['z'][tmp_cond]
        sys.exit(0)

    Q = flydra.a2.flypos.pos2ori(X,force_pitch_0=True) # get orientation from velocity

    hz = fps
    dt = 1.0/hz

    # find fly horizontal velocity direction using central difference
    central_diff_2D = X[2:,:2] - X[:-2,:2]
    fly_direction_2D = numpy.angle( central_diff_2D[:,0] + central_diff_2D[:,1]*1j )

    all_pt_c_fly_retina_dist = []
    all_pt_c_fly_retina = []

    closest_all_pt_c_fly_retina_dist = []
    closest_all_pt_c_fly_retina = []
    closest_all_pt_c_fly_retina_mask = []

    for post in stim_xml.iterate_posts():
        intersections = intersect_line_with_iso_z_plane( post['verts'], X[:,2] )
        max_post_z = max( post['verts'][0][2], post['verts'][1][2] ) # max post height
        intersections = intersections[:,:2] # drop Z coord

        # put in post coords
        xy_offset_by_post_loc = X[:,:2] - intersections
        radius = post['diameter']/2.0
        pt_a, pt_b = calc_circle_tangent_points(radius,xy_offset_by_post_loc) # left, right edges of post
        pt_c = calc_closest_points(radius,xy_offset_by_post_loc) # closest point of post

        # return to world coords
        pt_a += intersections
        pt_b += intersections
        pt_c += intersections
        pt_d = intersections # center of post

        # put in fly cartesian coords
        pt_a_fly = pt_a - X[:,:2]
        pt_b_fly = pt_b - X[:,:2]
        pt_c_fly = pt_c - X[:,:2]
        pt_d_fly = pt_d - X[:,:2]

        # put in fly retinal coords (absolute angle)
        pt_a_fly_retina_abs = numpy.angle( pt_a_fly[:,0] + pt_a_fly[:,1]*1j )
        pt_b_fly_retina_abs = numpy.angle( pt_b_fly[:,0] + pt_b_fly[:,1]*1j )
        pt_c_fly_retina_abs = numpy.angle( pt_c_fly[:,0] + pt_c_fly[:,1]*1j )
        pt_d_fly_retina_abs = numpy.angle( pt_d_fly[:,0] + pt_d_fly[:,1]*1j )

        # put in fly retinal coords (angle relative to horiz. velocity direction)
        pt_a_fly_retina = numpy.mod(pt_a_fly_retina_abs[1:-1] - fly_direction_2D, 2*numpy.pi )
        pt_b_fly_retina = numpy.mod(pt_b_fly_retina_abs[1:-1] - fly_direction_2D, 2*numpy.pi )
        pt_c_fly_retina = numpy.mod(pt_c_fly_retina_abs[1:-1] - fly_direction_2D, 2*numpy.pi )
        pt_d_fly_retina = numpy.mod(pt_d_fly_retina_abs[1:-1] - fly_direction_2D, 2*numpy.pi )

        # put in fly retinal coords (distance)
        pt_a_fly_retina_dist = numpy.hypot( pt_a_fly[1:-1,0], pt_a_fly[1:-1,1] )
        pt_b_fly_retina_dist = numpy.hypot( pt_b_fly[1:-1,0], pt_b_fly[1:-1,1] )
        pt_c_fly_retina_dist = numpy.hypot( pt_c_fly[1:-1,0], pt_c_fly[1:-1,1] )
        pt_d_fly_retina_dist = numpy.hypot( pt_d_fly[1:-1,0], pt_d_fly[1:-1,1] )

        bad_cond = (X[1:-1,2] > max_post_z)
        if bad:
            bad_cond = ~bad_cond
        pt_a_fly_retina = numpy.ma.masked_where( bad_cond, pt_a_fly_retina )
        pt_b_fly_retina = numpy.ma.masked_where( bad_cond, pt_b_fly_retina )
        pt_c_fly_retina = numpy.ma.masked_where( bad_cond, pt_c_fly_retina )
        pt_d_fly_retina = numpy.ma.masked_where( bad_cond, pt_d_fly_retina )

        pt_a_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_a_fly_retina_dist )
        pt_b_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_b_fly_retina_dist )
        pt_c_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_c_fly_retina_dist )
        pt_d_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_d_fly_retina_dist )

        def plot_coords(arr):
            # wrap around 0 radians, convert to degrees
            arr += numpy.pi
            arr = numpy.mod( arr, 2*numpy.pi )
            arr -= numpy.pi
            arr = arr*R2D
            return arr

        time_seconds = numpy.arange( len(pt_a_fly_retina) )/float(fps)

        if 'angle' in subplot:
            # plot left and right edges of post
            ax = subplot['angle']
            line, = ax.plot( time_seconds, plot_coords(pt_a_fly_retina), '.' )
            ax.plot( time_seconds, plot_coords(pt_b_fly_retina), '.', color = line.get_color() )

        if 'dist' in subplot:
            # distance to closest point of post
            ax = subplot['dist']
            line, = ax.plot( time_seconds, pt_c_fly_retina_dist, '.' )
            ax.plot( time_seconds, pt_c_fly_retina_dist, '.', color = line.get_color() )

        if 'dist_vs_angle' in subplot:
            ax = subplot['dist_vs_angle']
            # relative to closest point of post
            line, = ax.plot( pt_c_fly_retina_dist, plot_coords(pt_c_fly_retina), '.' )

        if 'dist_vs_angle_hist' in subplot:
            # relative to closest point of post
            all_pt_c_fly_retina_dist.append( pt_c_fly_retina_dist.compressed() )
            all_pt_c_fly_retina.append( pt_c_fly_retina.compressed() )

        if ('closest_dist_vs_angle' in subplot or
            'closest_dist_vs_angle_hist' in subplot):
            closest_all_pt_c_fly_retina_dist.append( numpy.ma.getdata(pt_c_fly_retina_dist) )
            closest_all_pt_c_fly_retina.append( numpy.ma.getdata(pt_c_fly_retina) )
            closest_all_pt_c_fly_retina_mask.append( numpy.ma.getmask(pt_c_fly_retina_dist) )

    if 'dist_vs_angle_static' in subplot:
        x = numpy.linspace(0,1,100) # 1 meter
        y = numpy.zeros_like(x)
        xy = numpy.hstack( (x[:,numpy.newaxis], y[:,numpy.newaxis]) )

        radius = post['diameter']/2.0
        pt_a, pt_b = calc_circle_tangent_points(radius,xy)

        # fly eye coords
        fly_a = pt_a-xy
        fly_b = pt_b-xy

        angle_a = numpy.angle( fly_a[:,0] + fly_a[:,1]*1j )
        angle_b = numpy.angle( fly_b[:,0] + fly_b[:,1]*1j )

        angular_size = numpy.mod(angle_a - angle_b, 2*numpy.pi)

        ax = subplot['dist_vs_angle_static']
        ax.plot( x, angular_size*R2D )
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('retinal size (deg)')

    if 'angle' in subplot:
        ax = subplot['angle']
        ax.set_xlabel('time (s)')
        ax.set_ylabel('retinal position (deg)')

    if 'dist' in subplot:
        ax = subplot['dist']
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance (m)')

    if 'dist_vs_angle' in subplot:
        ax = subplot['dist_vs_angle']
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('retinal position (deg)')

    if 'dist_vs_angle_hist' in subplot:
        ax = subplot['dist_vs_angle_hist']

        # data is already compressed()
        all_pt_c_fly_retina_dist = numpy.hstack(all_pt_c_fly_retina_dist)
        all_pt_c_fly_retina = numpy.hstack(all_pt_c_fly_retina)
        if 1:
            # limit to close range
            cond = all_pt_c_fly_retina_dist<=0.5
            all_pt_c_fly_retina_dist = all_pt_c_fly_retina_dist[cond]
            all_pt_c_fly_retina = all_pt_c_fly_retina[cond]

        ax.hexbin( all_pt_c_fly_retina_dist, plot_coords(all_pt_c_fly_retina))
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('retinal position (deg)')

    if ('closest_dist_vs_angle' in subplot or
        'closest_dist_vs_angle_hist' in subplot):

        if 1:
            # choose closest post

            # stack each post as a row
            closest_all_pt_c_fly_retina_dist = np.ma.array(closest_all_pt_c_fly_retina_dist,mask=closest_all_pt_c_fly_retina_mask)
            closest_all_pt_c_fly_retina = np.ma.array(closest_all_pt_c_fly_retina,mask=closest_all_pt_c_fly_retina_mask)

            # find closest row
            taker = np.ma.argmin( closest_all_pt_c_fly_retina_dist, axis=0 )
            col_idx = numpy.arange(len(taker))

            closest_dist = closest_all_pt_c_fly_retina_dist[ taker, col_idx ]
            angle_of_closest_dist = closest_all_pt_c_fly_retina[ taker, col_idx ]

            if 'closest_dist_vs_angle' in subplot:
                ax = subplot['closest_dist_vs_angle']
                ax.plot( closest_dist, plot_coords(angle_of_closest_dist), '.')

                if '2nd_closest_dist_vs_angle' in subplot:
                    # make closest row the farthest row
                    closest_all_pt_c_fly_retina_dist.mask[taker, col_idx] = True

                    # find new closest row
                    taker = numpy.ma.argmin( closest_all_pt_c_fly_retina_dist, axis=0 )
                    col_idx = numpy.arange(len(taker))

                    closest_dist = closest_all_pt_c_fly_retina_dist[ taker, col_idx ]
                    angle_of_closest_dist = closest_all_pt_c_fly_retina[ taker, col_idx ]

                    ax = subplot['2nd_closest_dist_vs_angle']
                    ax.plot( closest_dist, plot_coords(angle_of_closest_dist), '.')

            if 'closest_dist_vs_angle_hist' in subplot:
                ax = subplot['closest_dist_vs_angle_hist']
                ax.hexbin( closest_dist, plot_coords(angle_of_closest_dist))

    if 'z' in subplot:
        ax = subplot['z']

        z = X[1:-1,2]
        time_seconds = numpy.arange( len(z) )/float(fps)

        ax.plot( time_seconds, z, '-' )
        ax.axhline(max_post_z)

        ax.set_xlabel('time (s)')
        ax.set_ylabel('z (m)')

def doit(options=None):
    subplot={}
    if 1:
        fig = pylab.figure()
        fig.text(0,0, options.kalman_filename )
        ax=fig.add_subplot(3,1,1)
        subplot['angle']=ax
        ax=fig.add_subplot(3,1,2,sharex=ax)
        subplot['dist']=ax
        ax=fig.add_subplot(3,1,3,sharex=ax)
        subplot['z']=ax

    if 1:
        fig2 = pylab.figure()
        ax=fig2.add_subplot(2,1,1)
        subplot['dist_vs_angle']=ax

        ax=fig2.add_subplot(2,1,2,sharex=ax,sharey=ax)
        subplot['dist_vs_angle_hist']=ax

    if 0:
        fig3 = pylab.figure()
        ax=fig3.add_subplot(1,1,1)
        subplot['dist_vs_angle_static']=ax

    if 1:
        fig4 = pylab.figure()
        ax=fig4.add_subplot(2,1,1)
        subplot['closest_dist_vs_angle']=ax
        ax.set_title('closest post only')

        ax=fig4.add_subplot(2,1,2,sharex=ax,sharey=ax)
        subplot['2nd_closest_dist_vs_angle']=ax
        ax.set_title('2nd closest post only')

    plot_angle_dist(subplot=subplot,options=options)
    if '2nd_closest_dist_vs_angle' in subplot:
        xlim = subplot['2nd_closest_dist_vs_angle'].get_xlim()
        subplot['2nd_closest_dist_vs_angle'].set_xlim((0,xlim[1]))

    if 0:
        fig4 = pylab.figure()
        ax=fig4.add_subplot(3,1,1)
        subplot['dist_vs_angle']=ax

        ax=fig4.add_subplot(3,1,2,sharex=ax,sharey=ax)
        subplot['dist_vs_angle_hist']=ax
        plot_angle_dist(subplot=subplot,options=options, bad=True)

        subplot['dist_vs_angle'].set_title('bad')

    pylab.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--save-envmap", action='store_true',
                      default=False)

    parser.add_option("--plot-receptors", action='store_true',
                      default=False)

    analysis_options.add_common_options( parser )
    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args):
        parser.print_help()
        return

    doit( options=options,
         )

if __name__=='__main__':
    main()

