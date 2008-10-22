from __future__ import division
from __future__ import with_statement
import pkg_resources
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, time, warnings

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

R2D = 180/numpy.pi

if 1:
    # http://jfly.iam.u-tokyo.ac.jp/color/index.html
    post_id_label2color = {None:(0,0,0,1), # black

                           0:(.8,.4,0,1), # vermillion ( very right )
                           1:(.35,.7,.9,1), # sky blue ( left )
                           2:(0,.6,.6,1), # bluish green (forward)
                           3:(.95,.9,.25,1), # yellow ( hover )
                           4:(0,.45,.7,1), # blue ( very left )
                           5:(.8,.6,.7,1), # reddish purple (backward)
                           6:(.9,.6,0,1), # orange (right )
                           }

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

def angle_diff(ang1,ang2):
    return np.mod((ang1-ang2)+np.pi,2*np.pi)-np.pi

def test_angle_diff():
    ang1 = np.array([np.pi-0.001, -0.001,  0.001, np.pi+0.001])
    ang2 = np.array([np.pi+0.001,  0.001, -0.001, np.pi-0.001])
    actual = angle_diff(ang1,ang2)
    expected = np.array([-0.002, -0.002, 0.002, 0.002])
    #print 'actual',actual
    #print 'expected',expected

def get_horiz_turns( vx,vy, subsample_factor=1, frames_per_second=None):
    """return angular velocity of velocity direction in rad/sec"""
    #warnings.filterwarnings( "error" )
    N_observations = len(vx)//subsample_factor
    horiz_turns = []
    horiz_vel_angle = np.arctan2( vy, vx )
    d_angles = angle_diff(horiz_vel_angle[1:],horiz_vel_angle[:-1])
    for i in range( N_observations ):
        start = i*subsample_factor
        stop = (i+1)*subsample_factor
        total_angular_change = np.ma.sum(  d_angles[start:stop] )
        n_samples = len(np.ma.array( d_angles[start:stop] ).compressed())
        if n_samples==0:
            horiz_turns.append( np.nan ) # rad/sec
        else:
            whole_dt = 1.0/frames_per_second * n_samples
            vel_angular_rate = total_angular_change/whole_dt
            horiz_turns.append( vel_angular_rate ) # rad/sec
    horiz_turns = np.array( horiz_turns )
    return horiz_turns

def read_files_and_fuse_ids(options=None):
    """

    Note: this assumes that posts are perfectly vertical in the world
    coordinate system and that flies only look in a level plane.

    """
    assert options is not None
    assert options.stim_xml is not None, 'you must specify a stimulus .xml file'

    ca = core_analysis.get_global_CachingAnalyzer()

    if options.kalman_filename is None:
        raise ValueError('options.kalman_filename must be specified')
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(options.kalman_filename)

    fps = result_utils.get_fps( data_file )

    if 1:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    if 1:
        file_timestamp = os.path.split(data_file.filename)[-1][4:19]
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

    # find saccades
    saccade_results = core_analysis.detect_saccades( kalman_rows,
                                                     frames_per_second=fps )
    return kalman_rows,fps,stim_xml, saccade_results

def calc_retinal_coord_array(kalman_rows,fps,stim_xml,
                             angular_velocity_method='linear_velocity', # tangent to direction of travel
                             extra_columns=None, # list of tuples of ('name',ndarray)
                             closest_method='closest angle', # also 'closest distance'
                             #closest_method='closest distance',
                             ):
    """return recarray with a row for each row of kalman_rows, but processed to include stimulus-relative columns"""
    orig_row_length = len(kalman_rows)
    result_col_arrays = []
    result_col_names = []

    frame = kalman_rows['frame']

    if extra_columns is not None:
        for (name, ndarray) in extra_columns:
            if len(ndarray) != len(frame):
                raise ValueError('ndarray is not length of kalman_rows')
            result_col_arrays.append( ndarray )
            result_col_names.append( name )

    result_col_arrays.append( frame )
    result_col_names.append( 'frame' )

    result_col_arrays.append( kalman_rows['orig_data_present'] )
    result_col_names.append( 'orig_data_present' )

    X = numpy.array( [kalman_rows['x'],
                      kalman_rows['y'],
                      kalman_rows['z']]).T

    result_col_arrays.append( X[:,0] )
    result_col_names.append( 'x' )
    result_col_arrays.append( X[:,1] )
    result_col_names.append( 'y' )
    result_col_arrays.append( X[:,2] )
    result_col_names.append( 'z' )

    dt = 1.0/fps

    fly_velocity = (X[2:,:] - X[:-2,:])/(2*dt)
    # pad with nan to keep same length
    fly_velocity = np.vstack(( [[np.nan,np.nan,np.nan]],
                               fly_velocity,
                               [[np.nan,np.nan,np.nan]] ))
    # find fly horizontal velocity direction using central difference
    fly_direction_2D = numpy.angle( fly_velocity[:,0] + fly_velocity[:,1]*1j )

    result_col_arrays.append( fly_velocity[:,0] )
    result_col_names.append( 'vel_x' )
    result_col_arrays.append( fly_velocity[:,1] )
    result_col_names.append( 'vel_y' )
    result_col_arrays.append( fly_velocity[:,2] )
    result_col_names.append( 'vel_z' )

    vel_horiz = np.sqrt(np.sum(fly_velocity[:,:2]**2,axis=1))
    result_col_arrays.append( vel_horiz )
    result_col_names.append( 'vel_horiz' )

    result_col_arrays.append( fly_direction_2D )
    result_col_names.append( 'fly_direction_2D' )

    closest_all_pt_c_fly_retina_dist = []
    closest_all_pt_c_fly_retina_dist_speed = []
    closest_all_pt_c_fly_retina_dist_accel = []
    closest_all_pt_c_fly_retina = []
    closest_all_pt_c_fly_retina_mask = []

    max_post_zs = []
    for post_num,post in enumerate(stim_xml.iterate_posts()):
        max_post_z = max( post['verts'][0][2], post['verts'][1][2] ) # max post height
        max_post_zs.append( max_post_z )
    if not len(max_post_zs):
        raise ValueError('post-based analysis requires (at least one) post')
    max_z = min( max_post_zs )

    for post_num,post in enumerate(stim_xml.iterate_posts()):
        post_name = 'post%d'%post_num

        intersections = intersect_line_with_iso_z_plane( post['verts'], X[:,2] )
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
        pt_a_fly_retina = numpy.mod(pt_a_fly_retina_abs - fly_direction_2D, 2*numpy.pi )
        pt_b_fly_retina = numpy.mod(pt_b_fly_retina_abs - fly_direction_2D, 2*numpy.pi )
        pt_c_fly_retina = numpy.mod(pt_c_fly_retina_abs - fly_direction_2D, 2*numpy.pi )
        pt_d_fly_retina = numpy.mod(pt_d_fly_retina_abs - fly_direction_2D, 2*numpy.pi )

        # center about 0 (shift branch cut from 0/2pi to -pi/pi)
        pt_a_fly_retina = np.mod(pt_a_fly_retina+np.pi,2*np.pi)-np.pi
        pt_b_fly_retina = np.mod(pt_b_fly_retina+np.pi,2*np.pi)-np.pi
        pt_c_fly_retina = np.mod(pt_c_fly_retina+np.pi,2*np.pi)-np.pi
        pt_d_fly_retina = np.mod(pt_d_fly_retina+np.pi,2*np.pi)-np.pi

        # put in fly retinal coords (distance)
        pt_a_fly_retina_dist = numpy.hypot( pt_a_fly[:,0], pt_a_fly[:,1] )
        pt_b_fly_retina_dist = numpy.hypot( pt_b_fly[:,0], pt_b_fly[:,1] )
        pt_c_fly_retina_dist = numpy.hypot( pt_c_fly[:,0], pt_c_fly[:,1] )
        pt_d_fly_retina_dist = numpy.hypot( pt_d_fly[:,0], pt_d_fly[:,1] )

        pt_c_fly_retina_dist_speed = (pt_c_fly_retina_dist[2:]-pt_c_fly_retina_dist[:-2])/(2*dt)
        pt_c_fly_retina_dist_speed = np.hstack(([pt_c_fly_retina_dist_speed[0]],
                                                pt_c_fly_retina_dist_speed,
                                                [pt_c_fly_retina_dist_speed[-1]])) # pad ends to maintain length

        pt_c_fly_retina_dist_accel = (pt_c_fly_retina_dist_speed[2:]-pt_c_fly_retina_dist_speed[:-2])/(2*dt)
        pt_c_fly_retina_dist_accel = np.hstack(([pt_c_fly_retina_dist_accel[0]],
                                                pt_c_fly_retina_dist_accel,
                                                [pt_c_fly_retina_dist_accel[-1]]))

        result_col_arrays.append( pt_a_fly_retina )
        result_col_names.append( post_name+'_pt_a_fly_retina' )
        result_col_arrays.append( pt_b_fly_retina )
        result_col_names.append( post_name+'_pt_b_fly_retina' )
        result_col_arrays.append( pt_c_fly_retina )
        result_col_names.append( post_name+'_pt_c_fly_retina' )
        result_col_arrays.append( pt_d_fly_retina )
        result_col_names.append( post_name+'_pt_d_fly_retina' )

        result_col_arrays.append( pt_a_fly_retina_dist )
        result_col_names.append( post_name+'_pt_a_fly_retina_dist' )
        result_col_arrays.append( pt_b_fly_retina_dist )
        result_col_names.append( post_name+'_pt_b_fly_retina_dist' )
        result_col_arrays.append( pt_c_fly_retina_dist )
        result_col_names.append( post_name+'_pt_c_fly_retina_dist' )
        result_col_arrays.append( pt_d_fly_retina_dist )
        result_col_names.append( post_name+'_pt_d_fly_retina_dist' )

        bad_cond = (X[:,2] > max_z)

        result_col_arrays.append( bad_cond )
        result_col_names.append( post_name+'_bad_cond' )

        # mask for finding closest point per-post:
        pt_a_fly_retina = numpy.ma.masked_where( bad_cond, pt_a_fly_retina )
        pt_b_fly_retina = numpy.ma.masked_where( bad_cond, pt_b_fly_retina )
        pt_c_fly_retina = numpy.ma.masked_where( bad_cond, pt_c_fly_retina )
        pt_d_fly_retina = numpy.ma.masked_where( bad_cond, pt_d_fly_retina )

        pt_a_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_a_fly_retina_dist )
        pt_b_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_b_fly_retina_dist )
        pt_c_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_c_fly_retina_dist )
        pt_d_fly_retina_dist = numpy.ma.masked_where( bad_cond, pt_d_fly_retina_dist )

        # accumulate per-post results:
        closest_all_pt_c_fly_retina_dist.append( numpy.ma.getdata(pt_c_fly_retina_dist) )
        closest_all_pt_c_fly_retina.append( numpy.ma.getdata(pt_c_fly_retina) )
        closest_all_pt_c_fly_retina_mask.append( numpy.ma.getmaskarray(pt_c_fly_retina_dist) )
        closest_all_pt_c_fly_retina_dist_speed.append( pt_c_fly_retina_dist_speed )
        closest_all_pt_c_fly_retina_dist_accel.append( pt_c_fly_retina_dist_accel )

    # stack each post as a row
    closest_all_pt_c_fly_retina_dist = np.ma.array(closest_all_pt_c_fly_retina_dist,mask=closest_all_pt_c_fly_retina_mask)
    closest_all_pt_c_fly_retina_dist_speed = np.ma.array(closest_all_pt_c_fly_retina_dist_speed,mask=closest_all_pt_c_fly_retina_mask)
    closest_all_pt_c_fly_retina_dist_accel = np.ma.array(closest_all_pt_c_fly_retina_dist_accel,mask=closest_all_pt_c_fly_retina_mask)
    closest_all_pt_c_fly_retina = np.ma.array(closest_all_pt_c_fly_retina,mask=closest_all_pt_c_fly_retina_mask)

    # find closest row
    if closest_method=='closest distance':
        taker = np.ma.argmin( closest_all_pt_c_fly_retina_dist, axis=0 )
    elif closest_method=='closest angle':
        taker = np.ma.argmin( abs(closest_all_pt_c_fly_retina), axis=0 )
    col_idx = numpy.arange(len(taker))

    closest_dist = closest_all_pt_c_fly_retina_dist[ taker, col_idx ]
    closest_dist_speed = closest_all_pt_c_fly_retina_dist_speed[ taker, col_idx ]
    closest_dist_accel = closest_all_pt_c_fly_retina_dist_accel[ taker, col_idx ]
    angle_of_closest_dist = closest_all_pt_c_fly_retina[ taker, col_idx ]
    result_col_arrays.append( closest_dist )
    result_col_names.append( 'closest_dist' )

    result_col_arrays.append( closest_dist_speed )
    result_col_names.append( 'closest_dist_speed' )

    result_col_arrays.append( closest_dist_accel )
    result_col_names.append( 'closest_dist_accel' )

    result_col_arrays.append( np.ma.getmaskarray(closest_dist) )
    result_col_names.append( 'closest_dist_mask' )

    result_col_arrays.append( angle_of_closest_dist )
    result_col_names.append( 'angle_of_closest_dist' )

    # calculate angular velocities
    if angular_velocity_method=='linear_velocity': # tangent to direction of travel
        horizontal_angular_velocity = get_horiz_turns( fly_velocity[:,0], fly_velocity[:,1],
                                                       frames_per_second=fps)

        result_col_arrays.append( horizontal_angular_velocity )
        result_col_names.append( 'horizontal_angular_velocity' )

        for delay in [50,100,150]: # msec
            n_steps_delay = int(np.round((delay/1000.0) /dt))
            if len(horizontal_angular_velocity) > n_steps_delay:
                delayed_hv = np.hstack((
                    horizontal_angular_velocity[n_steps_delay:],
                    [np.nan]*n_steps_delay ))
            else:
                delayed_hv =  np.array([np.nan]*len(horizontal_angular_velocity) )

            result_col_arrays.append( delayed_hv )
            result_col_names.append( 'horizontal_angular_velocity_%dmsec_delay'%delay )

        post_angle = angle_of_closest_dist
        post_angle_x = np.cos( post_angle ) # allow treating with linear distance operators
        post_angle_y = np.sin( post_angle )

        result_col_arrays.append( post_angle_x )
        result_col_names.append( 'closest_post_angle_x' )
        result_col_arrays.append( post_angle_y )
        result_col_names.append( 'closest_post_angle_y' )

        post_angular_velocity = -get_horiz_turns( post_angle_x, post_angle_y,
                                                  frames_per_second=fps)

        result_col_arrays.append( post_angular_velocity )
        result_col_names.append( 'closest_post_angular_velocity' )

    result = np.rec.fromarrays( result_col_arrays, names=result_col_names )
    final_row_length = len(result)
    assert final_row_length==orig_row_length
    return result

def plot_angle_dist(subplot=None,results_recarray=None,fps=None,saccade_results=None):

    def plot_coords(arr):
        arr = numpy.array(arr,copy=True) # don't modify original
        # wrap around 0 radians, convert to degrees
        arr += numpy.pi
        arr = numpy.mod( arr, 2*numpy.pi )
        arr -= numpy.pi
        arr = arr*R2D
        return arr

    all_pt_c_fly_retina_dist = []
    all_pt_c_fly_retina = []

    post_num = 0
    while True: # loop over all posts, we don't know how many yet

        post_name = 'post%d'%post_num
        if (post_name+'_bad_cond') not in results_recarray.dtype.fields:
            break # no more posts
        this_post_color = post_id_label2color[post_num]

        post_num += 1

        bad_cond = results_recarray[ post_name+'_bad_cond']

        pt_a_fly_retina = results_recarray[ post_name+'_pt_a_fly_retina']
        pt_a_fly_retina = np.ma.masked_where( bad_cond, pt_a_fly_retina )
        pt_a_fly_retina_dist = results_recarray[ post_name+'_pt_a_fly_retina_dist']
        pt_a_fly_retina_dist = np.ma.masked_where( bad_cond, pt_a_fly_retina_dist )

        pt_b_fly_retina = results_recarray[ post_name+'_pt_b_fly_retina']
        pt_b_fly_retina = np.ma.masked_where( bad_cond, pt_b_fly_retina )
        pt_b_fly_retina_dist = results_recarray[ post_name+'_pt_b_fly_retina_dist']
        pt_b_fly_retina_dist = np.ma.masked_where( bad_cond, pt_b_fly_retina_dist )

        pt_c_fly_retina = results_recarray[ post_name+'_pt_c_fly_retina']
        pt_c_fly_retina = np.ma.masked_where( bad_cond, pt_c_fly_retina )
        pt_c_fly_retina_dist = results_recarray[ post_name+'_pt_c_fly_retina_dist']
        pt_c_fly_retina_dist = np.ma.masked_where( bad_cond, pt_c_fly_retina_dist )

        pt_d_fly_retina = results_recarray[ post_name+'_pt_d_fly_retina']
        pt_d_fly_retina = np.ma.masked_where( bad_cond, pt_d_fly_retina )
        pt_d_fly_retina_dist = results_recarray[ post_name+'_pt_d_fly_retina_dist']
        pt_d_fly_retina_dist = np.ma.masked_where( bad_cond, pt_d_fly_retina_dist )

        #time_seconds = numpy.arange( len(pt_a_fly_retina) )/float(fps)
        time_seconds = (results_recarray['frame']-results_recarray['frame'][0])/float(fps)

        orig_post_kwargs = dict( marker='.',
                                 ms=2.5,
                                 linestyle='',
                                 color=this_post_color)
        if 'angle' in subplot:
            # plot left and right edges of post
            ax = subplot['angle']
            line, = ax.plot( time_seconds, plot_coords(pt_a_fly_retina), **orig_post_kwargs)
            ax.plot( time_seconds, plot_coords(pt_b_fly_retina), **orig_post_kwargs)

        if 'dist' in subplot:
            # distance to closest point of post
            ax = subplot['dist']
            #line, = ax.plot( time_seconds, pt_c_fly_retina_dist*100.0, '.' )
            ax.plot( time_seconds, pt_c_fly_retina_dist*100.0,**orig_post_kwargs)

        if 'dist_vs_angle' in subplot:
            ax = subplot['dist_vs_angle']
            # relative to closest point of post
            line, = ax.plot( pt_c_fly_retina_dist, plot_coords(pt_c_fly_retina), '.',color=this_post_color)

        if 'dist_vs_angle_hist' in subplot:
            # relative to closest point of post
            all_pt_c_fly_retina_dist.append( pt_c_fly_retina_dist.compressed() )
            all_pt_c_fly_retina.append( pt_c_fly_retina.compressed() )

    if 'dist_vs_angle_static' in subplot:
        dist_from = 'cylinder_edge'
        #dist_from = 'cylinder_center'
        x = numpy.linspace(0,.6,500) # 60 cm
        y = numpy.zeros_like(x)
        xy = numpy.hstack( (x[:,numpy.newaxis], y[:,numpy.newaxis]) )

        post_diameter = 0.05
        warnings.warn('WARNING: fixing post_diameter to %.1f'%post_diameter)


        radius = post_diameter/2.0
        pt_a, pt_b = calc_circle_tangent_points(radius,xy)

        # fly eye coords
        fly_a = pt_a-xy
        fly_b = pt_b-xy

        angle_a = numpy.angle( fly_a[:,0] + fly_a[:,1]*1j )
        angle_b = numpy.angle( fly_b[:,0] + fly_b[:,1]*1j )

        angular_size = numpy.mod(angle_a - angle_b, 2*numpy.pi)

        ax = subplot['dist_vs_angle_static']
        kwargs=dict(lw=3)
        if dist_from=='cylinder_center':
            ax.plot( x*100.0, angular_size*R2D, **kwargs)

        elif dist_from=='cylinder_edge':
            xnew = x-radius
            if 0:
                # pt_c[0] is nan, so this check fails
                pt_c = calc_closest_points(radius,xy)
                xnew2 = x-pt_c[:,0]
                assert np.allclose(xnew,xnew2)
            ax.plot( xnew*100.0, angular_size*R2D, **kwargs)
        ax.grid(True)
        ax.set_xlabel('distance (cm)')
        ax.set_ylabel('retinal size (deg)')

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
            # remove nans (hexbin doesn't like nans)
            good_cond = ~np.isnan(all_pt_c_fly_retina)
            all_pt_c_fly_retina_dist = all_pt_c_fly_retina_dist[good_cond]
            all_pt_c_fly_retina = all_pt_c_fly_retina[good_cond]

        if 1:
            # limit to close range
            cond = all_pt_c_fly_retina_dist<=0.5
            all_pt_c_fly_retina_dist = all_pt_c_fly_retina_dist[cond]
            all_pt_c_fly_retina = all_pt_c_fly_retina[cond]

        ax.hexbin( all_pt_c_fly_retina_dist, plot_coords(all_pt_c_fly_retina), gridsize = (40,20))
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('retinal position (deg)')

    # done looping over individual posts
    closest_dist = np.ma.array(results_recarray[ 'closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])
    angle_of_closest_dist = np.ma.array(results_recarray[ 'angle_of_closest_dist' ],mask=results_recarray[ 'closest_dist_mask' ])

    def fill_saccades(ax):
        for saccade_frame in saccade_results['frames']:
            t = (saccade_frame-results_recarray['frame'][0])/float(fps)
            t0 = t-0.01
            t1 = t+0.01
            ax.axvspan( t0, t1, facecolor='0.5', alpha=0.5, edgecolor='none' )

    closest_kwargs = dict(marker='.',linestyle='',color='k',ms = 8,zorder=-10)
    #closest_zorder = 10
    if 'angle' in subplot:
        ax = subplot['angle']
        if 1:
            # plot closest post
            line, = ax.plot( time_seconds, plot_coords(angle_of_closest_dist), **closest_kwargs)
            #line.set_zorder(closest_zorder)
        fill_saccades(ax)
        for fake_grid_line in [-180,-90,0,90,180]:
            ax.axhline( fake_grid_line, linestyle='--', color='0.8', alpha=0.5, zorder=-10)

        ax.set_xlabel('time (s)')
        ax.set_ylabel('post angle (deg)')

    if 'dist' in subplot:
        ax = subplot['dist']
        if 1:
            # plot closest post
            line, = ax.plot( time_seconds, closest_dist*100.0, **closest_kwargs)
            #line.set_zorder(closest_zorder)
        fill_saccades(ax)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance (cm)')

    if 'closest_dist_vs_angle' in subplot:
        ax = subplot['closest_dist_vs_angle']
        ax.plot( closest_dist, plot_coords(angle_of_closest_dist), '.')

    if 'closest_dist_vs_angle_hist' in subplot:
        ax = subplot['closest_dist_vs_angle_hist']
        ax.hexbin( closest_dist, plot_coords(angle_of_closest_dist), gridsize = (40,20))

    if 'z' in subplot:
        ax = subplot['z']

        z = results_recarray[ 'z' ]

        time_seconds = numpy.arange( len(z) )/float(fps)

        ax.plot( time_seconds, z, '-' )

        ax.set_xlabel('time (s)')
        ax.set_ylabel('z (m)')

    if 'angular_velocity' in subplot:
        ax = subplot['angular_velocity']

        horiz_ang_vel = results_recarray[ 'horizontal_angular_velocity' ]

        time_seconds = numpy.arange( len(horiz_ang_vel) )/float(fps)

        ax.plot( time_seconds, horiz_ang_vel*R2D, '-' )
        fill_saccades(ax)
        for fake_grid_line in [-1000,0,1000]:
            ax.axhline( fake_grid_line, linestyle='--', color='0.8', alpha=0.5, zorder=-10)

        ax.set_xlabel('time (s)')
        ax.set_ylabel('angular velocity (deg/s)')

        if 0:
            frame = results_recarray[ 'frame' ]
            prev_idx=0
            prev_mid_idx = 0
            sfs = list(saccade_results['frames']) # saccade frames
            sfs.sort()
            for saccde_num,saccade_frame in enumerate(sfs):
                t = (saccade_frame-frame[0])/float(fps)
                idx = np.nonzero(frame==saccade_frame)[0]

                if 1:
                    mid_idx = (idx+prev_idx)//2
                    start_idx = prev_mid_idx
                    stop_idx = mid_idx
                    prev_mid_idx = mid_idx
                else:
                    start_idx = prev_idx
                    stop_idx = idx

                angle_turned = np.sum( (horiz_ang_vel*R2D/fps)[start_idx:stop_idx] )

                # actually, this measures the angle of the previous
                # saccade, not the time point printed ehre...

                print 'saccade %d (time %.2f) - turned since last saccade: %.1f deg'%(
                    saccde_num, t, angle_turned)
                prev_idx = idx

def doit(options=None):
    if options.obj_only is not None:
        raise ValueError('obj_only is not a valid option for this function')

    kalman_rows, fps, stim_xml, saccade_results = read_files_and_fuse_ids(options=options)
    results_recarray = calc_retinal_coord_array(kalman_rows, fps, stim_xml)
    #import matplotlib
    #matplotlib.use('GTKAgg')
    import matplotlib.pyplot as plt
    import pylab

    PRETTY_NOTINTERACTIVE=True
    subplot={}
    if 1:
        fig1 = pylab.figure(figsize=(8,6))
        fig1.text(0,0, options.kalman_filename )
        ax=fig1.add_subplot(3,1,1)
        subplot['angle']=ax
        post_angle_axes = ax
        if PRETTY_NOTINTERACTIVE:
            ax = None # don't share axis -- disables plotting tick labels on all

        ax=fig1.add_subplot(3,1,2,sharex=ax)
        subplot['dist']=ax
        post_dist_axes = ax
        if PRETTY_NOTINTERACTIVE:
            ax = None # don't share axis -- disables plotting tick labels on all

        ax=fig1.add_subplot(3,1,3,sharex=ax)
        subplot['angular_velocity']=ax
        ang_vel_axes = ax

    if 0:
        fig2 = pylab.figure()
        ax=fig2.add_subplot(2,1,1)
        subplot['dist_vs_angle']=ax

        ax=fig2.add_subplot(2,1,2,sharex=ax,sharey=ax)
        subplot['dist_vs_angle_hist']=ax

    if 1:
        fig3 = pylab.figure(figsize=(4,4))
        static_angle_ax=fig3.add_subplot(1,1,1)
        subplot['dist_vs_angle_static']=static_angle_ax
        ax.set_ylim((0,72))
        ax.set_xlim((0,60))

    if 0:
        fig4 = pylab.figure()
        ax=fig4.add_subplot(2,1,1)
        subplot['closest_dist_vs_angle']=ax
        ax.set_title('closest post only')

    plot_angle_dist(subplot=subplot,
                    results_recarray=results_recarray,
                    fps=fps,
                    saccade_results=saccade_results)

    if 1:
        this_interesting_time = (49.5,56.5) # used with DATA20080619_174254.kh5
        ax = post_angle_axes
        ax.set_xlim(this_interesting_time)
        ax.set_yticks([-180,-90,0,90,180])
        ax.set_xticklabels([])
        ax.set_xlabel('')

        ax = post_dist_axes
        ax.set_xlim(this_interesting_time)
        ax.set_ylim((0,80))
        ax.set_yticks([0,40,80])
        ax.set_xticklabels([])
        ax.set_xlabel('')

        ax = ang_vel_axes
        ax.set_xlim(this_interesting_time)
        ax.set_ylim((-2500,2500))
        for ext in ['.png','.svg']:
            fname = 'post_angle_over_time'+ext
            fig1.savefig(fname,dpi=200)
            print 'saved',fname


        ax=static_angle_ax
        ax.set_ylim((0,72))
        ax.set_xlim((0,60))

        for ext in ['.png','.svg']:
        #for ext in ['.png','.pdf','.svg']:
            fname = 'post_angular_size_static'+ext
            fig3.savefig(fname)#,dpi=55)
            print 'saved',fname

    if 0:
        fig4 = pylab.figure()
        ax=fig4.add_subplot(3,1,1)
        subplot['dist_vs_angle']=ax

        ax=fig4.add_subplot(3,1,2,sharex=ax,sharey=ax)
        subplot['dist_vs_angle_hist']=ax
        plot_angle_dist(subplot=subplot,results_recarray=results_recarray, fps=fps, bad=True)

        subplot['dist_vs_angle'].set_title('bad')

    if 1:
        frame = results_recarray['frame']
        time = (frame-frame[0])/fps
        cond = (this_interesting_time[0] <= time) & (time <= this_interesting_time[1])

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

        orig_data_present = results_recarray['orig_data_present']
        x = results_recarray['x']
        y = results_recarray['y']
        if 0:
            ax1.plot( x[orig_data_present],y[orig_data_present],'.',color='0.5',ms=0.8)
        else:
            ax1.plot( x,y,'.',color='0.5',ms=0.8)

        x = results_recarray['x'][cond]
        y = results_recarray['y'][cond]
        ax1.plot( x,y,'k.',ms=2)
        #ax1.text( x[0],y[0], 'start')
        stim_xml.plot_stim( ax1,
                            projection=xml_stimulus.SimpleOrthographicXYProjection(),
                            post_colors=post_id_label2color,
                            draw_post_as_circle=True,
                            #show_post_num=True,
                            )
        ax1.set_frame_on(False)
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])

        for ext in ['.png','.svg']:
            fname = 'post_angle_over_time_top_view'+ext
            fig.savefig(fname,dpi=200)
            print 'saved',fname

    #print 'show() called'
    #pylab.show()

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

