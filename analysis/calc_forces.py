
from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from numarray.ieeespecial import nan, getnan
import math
import numarray as nx
import flydra.reconstruct as reconstruct
import cgtypes # tested with cgkit 1.2
import tables # pytables

from PQmath import *

# restore builtin functions which may have been overridden
min = __builtins__.min
max = __builtins__.max
sum = __builtins__.sum
round = __builtins__.round
abs = __builtins__.abs

def my_interp( A, B, frac ):
    return frac*(B-A)+A

def interpolate_P( results, start_frame, stop_frame, typ='best' ):
    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best
    fXl = [(row['frame'],
            row['x'],row['y'],row['z'],
            row['p0'],row['p1'],row['p2'],row['p3'],row['p4'],row['p5']) for row in
           data3d if start_frame <= row['frame'] <= stop_frame ] # XXX
#           data3d.where( start_frame <= data3d.cols.frame <= stop_frame )]
    assert len(fXl) == 2
    assert stop_frame > start_frame
    assert (stop_frame - start_frame) > 1

    fXl = nx.array(fXl)
    frame = fXl[:,0].astype(nx.Int32)
    P = fXl[:,1:4]

    print '  ',start_frame, P[0,:]
    
    dPdt = (P[1,:]-P[0,:])/float(frame[1]-frame[0])
    for frame_no in range(start_frame+1, stop_frame):
        frac = float(frame_no-start_frame)/float(stop_frame-start_frame)
        newP = P[0,:]+dPdt*frac

        print '  ',frame_no,newP,'<- new value'
        
        # now save to disk
        old_nrow = None
#        for row in data3d.where( data3d.cols.frame == frame_no ):
        for row in data3d:
            if row['frame'] != frame_no: # XXX
                continue 
            if old_nrow is not None:
                raise RuntimeError('more than row with frame number %d in data3d'%frame_no)
            old_nrow = row.nrow()

        # delete old row
        if old_nrow is not None:
            data3d.removeRows(start=old_nrow,stop=None)

        X = newP
        line3d = [nan]*6 # fill with nans
        cam_nos_used_str = ''
        new_row = data3d.row
        new_row['frame'] = frame_no
        new_row['x'] = X[0]
        new_row['y'] = X[1]
        new_row['z'] = X[2]
        new_row['p0'] = line3d[0]
        new_row['p1'] = line3d[1]
        new_row['p2'] = line3d[2]
        new_row['p3'] = line3d[3]
        new_row['p4'] = line3d[4]
        new_row['p5'] = line3d[5]
        new_row['timestamp']=0.0
        new_row['camns_used']=cam_nos_used_str
        new_row['mean_dist']=0.0
        new_row.append()
        data3d.flush()

    print '  ',stop_frame, P[1,:]
    
def sort_on_col0( a, b ):
    a0 = a[0]
    b0 = b[0]
    if a0 < b0: return -1
    elif a0 > b0: return 1
    else: return 0

def slerp_quats( Q, bad_idxs, allow_roll = True ):
    """replace quats in sequence with interpolated version"""
    for cur_idx in bad_idxs:
        
        pre_idx = cur_idx-1
        preQ = None
        while preQ is None:
            if pre_idx < 0:
                raise IndexError
            preQ = Q[pre_idx]
            if len(getnan(nx.array((preQ.w,preQ.x,preQ.y,preQ.z)))[0]):
                preQ = None
                pre_idx -= 1
                
        post_idx = cur_idx+1
        postQ = None
        while postQ is None:
            postQ = Q[post_idx]
            if len(getnan(nx.array((postQ.w,postQ.x,postQ.y,postQ.z)))[0]):
                postQ = None
                post_idx += 1

        frac = float(cur_idx-pre_idx)/float(post_idx-pre_idx)
        #print '  ',frac, cur_idx, pre_idx, post_idx
        new_quat = cgtypes.slerp(frac, preQ, postQ)
        if allow_roll:
            Q[cur_idx] = new_quat
        else:
            # convert back and forth from orientation to eliminate roll
            ori = quat_to_orient(new_quat)
            no_roll_quat = orientation_to_quat(ori)
            Q[cur_idx] = no_roll_quat
    
def do_it(results,Psmooth=None,Qsmooth=None, alpha=0.2, beta=20.0, lambda1=2e-9,
          interp_OK=False,
          do_smooth_position = False,
          do_smooth_quats = False,
          start_frame = 18629,
          stop_frame = 19032,
          
          plot_pos_and_vel = False,
          plot_pos_err_histogram = False,

          
          plot_xy = False, plot_xy_Qsmooth = False, plot_xy_Qraw = True,
          plot_xz = False,
          plot_xy_air = False,
    
          plot_hist_horiz_vel = False,
          plot_hist_vert_vel=False,
          plot_forward_vel_vs_pitch_angle = False,
          
          plot_accel = False,
          plot_smooth_pos_and_vel = False,
          
          plot_Q = False,
          plot_body_angular_vel = False,
          plot_error_angles = False,
          plot_body_ground_V = False,
          plot_body_air_V = False,
          plot_forces = False,
    
          had_post = True,
          show_grid = False,

          ):
    # get data from file

    if type(results) == tables.File:
        data3d = results.root.data3d_best
        fXl = [(row['frame'],
                row['x'],row['y'],row['z'],
                row['p0'],row['p1'],row['p2'],row['p3'],row['p4'],row['p5']) for row in
               data3d if start_frame <= row['frame'] <= stop_frame ] # XXX
##               data3d.where( start_frame <= data3d.cols.frame <= stop_frame )]
        fXl.sort( sort_on_col0 )
    else:
        print 'assuming results are numeric'
        fXl = results
    fXl = nx.array(fXl)
    frame = fXl[:,0].astype(nx.Int32)
    P = fXl[:,1:4]
    line3d = fXl[:,4:]

    t_P = (frame-frame[0])*1e-2 # put in seconds
    to_meters = 1e-3 # put in meters (from mm)
    P = nx.array(P)*to_meters 
    if Psmooth is not None:
        Psmooth = nx.array(Psmooth)*to_meters
    line3d = nx.array(line3d)

    # check timestamps
    delta_ts = t_P[1:]-t_P[:-1]
    frames_missing = False

    interpolated_xyz_frames = []
    for i,delta_t in enumerate(delta_ts):
        if not (0.009 < delta_t < 0.011):
            if interp_OK:
                fXl = list(fXl)

                first = frame[i]
                last = frame[i+1]

                N = last-first
                #print 'first', fXl[i][1:4]
                #print 'last', fXl[i+1][1:4]
                for ii,fno in enumerate(range(first,last)):
                    if ii == 0:
                        continue
                    frac = ii/float(N)

                    # do interpolation
                    new_x = my_interp( fXl[i][1], fXl[i+1][1], frac )
                    new_y = my_interp( fXl[i][2], fXl[i+1][2], frac )
                    new_z = my_interp( fXl[i][3], fXl[i+1][3], frac )
                    new_row = nx.array( [fno, new_x, new_y, new_z, nan, nan, nan, nan, nan, nan],
                                        type=fXl[0].type() )
                    fXl.append( new_row )
                    #print '  ',frac, new_row
                    print 'linear interpolation (frame %d)'%(fno,)
                    interpolated_xyz_frames.append( fno )
            else:
                frames_missing = True
                print 'are you missing frames between %d and %d?'%(frame[i], frame[i+1])
    if frames_missing:
        raise ValueError("results have missing frames")

    if len(interpolated_xyz_frames):
        # re-sort and partition results
        fXl.sort( sort_on_col0 )
        
        fXl = nx.array(fXl)
        frame = fXl[:,0].astype(nx.Int32)
        P = fXl[:,1:4]
        line3d = fXl[:,4:]

        t_P = (frame-frame[0])*1e-2 # put in seconds
        to_meters = 1e-3 # put in meters (from mm)
        P = nx.array(P)*to_meters 
        if Psmooth is not None:
            Psmooth = nx.array(Psmooth)*to_meters
        line3d = nx.array(line3d)

        frame_list = list(frame)
        no_distance_penalty_idxs = [ frame_list.index( fno ) for fno in interpolated_xyz_frames ]
    else:
        no_distance_penalty_idxs = []
            
    delta_t = delta_ts[0]

    # get angular position phi
    phi_with_nans = reconstruct.line_direction(line3d) # unit vector
    bad_idxs = getnan(phi_with_nans[:,0])[0]
    if bad_idxs[0] == 0:
        print 'WARNING: no orientation for first point'
    
    Q = QuatSeq([ orientation_to_quat(U) for U in phi_with_nans ])
    slerp_quats( Q, bad_idxs, allow_roll=False )
    for cur_idx in bad_idxs:
        print 'SLERPed missing quat at time %.2f (frame %d)'%(cur_idx*1e-2, frame[cur_idx])
    t_bad = t_P[bad_idxs]
    frame_bad = frame[bad_idxs]
    
    # first position derivative (velocity)
    dPdt = (P[2:]-P[:-2]) / (2*delta_t)
    t_dPdt = t_P[1:-1]

    # second position derivative (acceleration)
    d2Pdt2 = (P[2:] - 2*P[1:-1] + P[:-2]) / (delta_t**2)
    t_d2Pdt2 = t_P[1:-1]

    # first orientation derivative (angular velocity)
    omega = (Q[:-1].inverse()*Q[1:]).log()/delta_t
    t_omega = t_P[:-1]
    
    # second orientation derivative (angular acceleration)
    omega_dot = ((Q[1:-1].inverse()*Q[2:]).log() -
                 (Q[:-2].inverse()*Q[1:-1]).log()) / (delta_t**2)
    t_omega_dot = t_P[1:-1]
    
    xtitle = 'time'
    xtitle = None
    
    if had_post:
        post_top_center=array([ 130.85457512,  169.45421191,   50.53490689])
        post_radius=5 # mm
        post_height=10 # mm

    outputs = []
    if Psmooth is None and do_smooth_position:
        of = ObjectiveFunctionPosition(P, delta_t, alpha,
                                       no_distance_penalty_idxs=no_distance_penalty_idxs)
        epsilon1 = 150e6
        #epsilon1 = 1.0
        Psmooth = P.copy()
        last_err = None
        max_iter1 = 10000
        count = 0
        while count<max_iter1:
            count+=1
            start = time.time()
            del_F = of.get_del_F(Psmooth)
            stop = time.time()
            print 'P elapsed: % 4.2f secs,'%(stop-start,),
            err = nx.sum(nx.sum(del_F**2,axis=1))
            print 'sum( norm(del F)):',err
            if err < epsilon1:
                break
            elif last_err is not None:
                if err > last_err:
                    print 'ERROR: error is increasing, aborting'
                    break
            last_err = err
            Psmooth = Psmooth - lambda1*del_F
        outputs.append(Psmooth/to_meters)
    if Psmooth is not None:
        dPdt_smooth = (Psmooth[2:]-Psmooth[:-2]) / (2*delta_t)
        d2Pdt2_smooth = (Psmooth[2:] - 2*Psmooth[1:-1] + Psmooth[:-2]) / (delta_t**2)
        do_smooth_position = True # we did it (if cached or just now)
        
    if Qsmooth is None and do_smooth_quats:
        #gamma = 1000
        gamma = 0.0
        of = ObjectiveFunctionQuats(Q, delta_t, beta, gamma)
        epsilon2 = 200e6
        epsilon2 = 0
        percent_error_eps = 2
        #percent_error_eps = 9
        #lambda2 = 2e-9
        lambda2 = 1e-9
        lambda2 = 1e-11
        Q_k = Q[:] # make copy
        last_err = None
        max_iter2 = 2000
        count = 0
        while count<max_iter2:
            count += 1
            start = time.time()
            del_G = of.get_del_G(Q_k)
            D = of._getDistance(Q_k)
            E = of._getEnergy(Q_k)
            R = of._getRoll(Q_k)
            print '  G = %s + %s*%s + %s*%s'%(str(D),str(beta),str(E),str(gamma),str(R))
            stop = time.time()
            err = math.sqrt(nx.sum(nx.array(abs(del_G))**2))
            if err < epsilon2:
                print 'reached epsilon2'
                break
            elif last_err is not None:
                pct_err = (last_err-err)/last_err*100.0
                print 'Q elapsed: % 6.2f secs,'%(stop-start,),
                print 'current gradient:',err,
                print '   (%3.1f%%)'%(pct_err,)
                
                if err > last_err:
                    print 'ERROR: error is increasing, aborting'
                    break
                if pct_err < percent_error_eps:
                    print 'reached percent_error_eps'
                    break
            else:
                print 'Q elapsed: % 6.2f secs,'%(stop-start,),
                print 'current gradient:',err
            last_err = err
            Q_k = Q_k*(del_G*-lambda2).exp()
        if count>=max_iter2:
            print 'reached max_iter2'
        Qsmooth = Q_k
        outputs.append(Qsmooth)
    if Qsmooth is not None:
        omega_smooth = (Qsmooth[:-1].inverse()*Qsmooth[1:]).log()/delta_t

        omega_dot_smooth = ((Qsmooth[1:-1].inverse()*Qsmooth[2:]).log() -
                            (Qsmooth[:-2].inverse()*Qsmooth[1:-1]).log()) / (delta_t**2)
        do_smooth_quats = True # we've got 'em now, one way or another

    # body-centric groundspeed (using quaternion rotation)
    body_ground_V = rotate_velocity_by_orientation( dPdt, Q[1:-1])
    if Qsmooth is not None:
        body_ground_V_smooth = rotate_velocity_by_orientation( dPdt_smooth, Qsmooth[1:-1])

    airspeed = nx.array((-.4,0,0))
    dPdt_air = dPdt - airspeed # world centric airspeed
    if Psmooth is not None:
        dPdt_smooth_air = dPdt_smooth - airspeed # world centric airspeed
    # body-centric airspeed (using quaternion rotation)
    body_air_V = rotate_velocity_by_orientation(dPdt_air,Q[1:-1])
    if Qsmooth is not None:
        body_air_V_smooth = rotate_velocity_by_orientation(dPdt_smooth_air,Qsmooth[1:-1])

    if 1: # compute body-centric angular velocity
        omega_body = rotate_velocity_by_orientation( omega, Q[:-1])
        if Qsmooth is not None:
            omega_smooth_body = rotate_velocity_by_orientation( omega_smooth, Qsmooth[:-1])
        t_omega_body = t_P[:-1]

    if Qsmooth is not None: # compute forces (for now, smooth data only)
        rad2deg = 180/math.pi
        
        # vector for current orientation (use only indices with velocity info)
        orient_parallel = quat_to_orient(Qsmooth)[1:-1]

        # vector for current velocity
        Vair_orient = dPdt_air/nx.sqrt(nx.sum(dPdt_air**2,axis=1)[:,nx.NewAxis])
        # compute alpha == angle of attack
        aattack = nx.arccos( [nx.dot(v,p) for v,p in zip(Vair_orient,orient_parallel)])
        #print aattack*rad2deg

        # find vector for normal force
        tmp_out_of_plane = [cross(v,p) for v,p in zip(Vair_orient,orient_parallel)]
        orient_normal = [cross(p,t) for t,p in zip(tmp_out_of_plane,orient_parallel)]

        cyl_diam = 0.5 #mm
        cyl_diam = cyl_diam / 1e3 # meters
        cyl_height = 1.75 #mm
        cyl_height = cyl_height / 1e3 # meters
        A = cyl_diam*cyl_height
        
        rho = 1.25 # kg/m^3
        
        V2 = body_air_V.x**2 + body_air_V.y**2 + body_air_V.z**2

        C_P=0.16664033221423064*nx.cos(aattack)+0.33552465566450407*nx.cos(aattack)**3
        C_N=0.75332031249999987*nx.sin(aattack)
        
        F_P = 0.5*rho*A*C_P*V2
        F_N = 0.5*rho*A*C_N*V2
        t_forces = t_dPdt

        # force required to stay aloft
        fly_mass = 1e-6 # guesstimate (1 milligram)
        G = 9.81 # meters / second
        aloft_force = fly_mass*G

    if 1: # compute error angles
        make_norm = reconstruct.norm_vec
        rad2deg = 180/math.pi

        # raw
        
        flat_heading = dPdt_air.copy()
        flat_heading[:,2] = 0 # no Z component
        flat_heading = [ make_norm(f) for f in flat_heading ]
        flat_heading_angle = nx.array([ math.atan2(f[1],f[0])*rad2deg for f in flat_heading ])

        vel_air_dir = [ make_norm(v) for v in dPdt_air ]
        vel_pitch_angle = nx.array([ math.asin(v[2])*rad2deg for v in vel_air_dir ])
        
        flat_orientation = quat_to_orient(Q)
        flat_orientation[:,2]=0
        flat_orientation = [ make_norm(f) for f in flat_orientation ]
        flat_orientation_angle = nx.array([ math.atan2(f[1],f[0])*rad2deg for f in flat_orientation ])

        orient_pitch_angle = [quat_to_euler(q)[1]*rad2deg for q in Q]

        heading_err = flat_orientation_angle[1:-1]-flat_heading_angle
        t_heading_err = t_dPdt
        pitch_body_err = orient_pitch_angle[1:-1]-vel_pitch_angle
        t_pitch_body_err = t_dPdt
        #pitch_body_err = [ math.asin(p[2])*rad2deg for p in quat_to_orient(Q) ]
        #t_pitch_body_err = t_P

        #   derivs
        d_heading_err_dt = (heading_err[2:]-heading_err[:-2]) / (2*delta_t)
        t_d_heading_err_dt = t_heading_err[1:-1]
        d_pitch_body_err_dt = (pitch_body_err[2:]-pitch_body_err[:-2]) / (2*delta_t)
        t_d_pitch_body_err_dt =t_pitch_body_err[1:-1]
        
        # smooth

        if Psmooth is not None and Qsmooth is not None:
            flat_heading_smooth = dPdt_smooth_air.copy()
            flat_heading_smooth[:,2] = 0 # no Z component
            flat_heading_smooth = [ make_norm(f) for f in flat_heading_smooth ]
            flat_heading_angle_smooth = nx.array([ math.atan2(f[1],f[0])*rad2deg for f in flat_heading_smooth ])

            vel_smooth_air_dir = [ make_norm(v) for v in dPdt_smooth_air ]
            vel_smooth_pitch_angle = nx.array([ math.asin(v[2])*rad2deg for v in vel_smooth_air_dir ])

            flat_orientation_smooth = quat_to_orient(Qsmooth)
            flat_orientation_smooth[:,2]=0
            flat_orientation_smooth = [ make_norm(f) for f in flat_orientation_smooth ]
            flat_orientation_angle_smooth = nx.array([ math.atan2(f[1],f[0])*rad2deg for f in flat_orientation_smooth ])

            orient_pitch_angle_smooth = [quat_to_euler(q)[1]*rad2deg for q in Qsmooth]

            heading_err_smooth = flat_orientation_angle_smooth[1:-1]-flat_heading_angle_smooth
            t_heading_err_smooth = t_dPdt
            pitch_body_err_smooth = orient_pitch_angle_smooth[1:-1]-vel_smooth_pitch_angle
            t_pitch_body_err_smooth = t_dPdt
            #pitch_body_err_smooth = nx.array([ math.asin(p[2])*rad2deg for p in quat_to_orient(Qsmooth) ])
            #t_pitch_body_err_smooth = t_P

            #   derivs
            d_heading_err_smooth_dt = (heading_err_smooth[2:]-heading_err_smooth[:-2]) / (2*delta_t)
            t_d_heading_err_smooth_dt = t_heading_err_smooth[1:-1]
            d_pitch_body_err_smooth_dt = (pitch_body_err_smooth[2:]-pitch_body_err_smooth[:-2]) / (2*delta_t)
            t_d_pitch_body_err_smooth_dt =t_pitch_body_err_smooth[1:-1]

    if 1: # compute horizontal velocity
        xvel, yvel = dPdt[:,0], dPdt[:,1]
        horiz_vel = nx.sqrt( xvel**2 + yvel**2)
        vert_vel = dPdt[:,2]

    if plot_hist_horiz_vel:
        hist( horiz_vel, bins=30 )
        xlabel( 'horizontal velocity (m/sec)')
        ylabel( 'count' )
        
    if plot_hist_vert_vel:
        hist( vert_vel, bins=30 )
        xlabel( 'vertical velocity (m/sec)')
        ylabel( 'count' )
        
    if plot_forward_vel_vs_pitch_angle:
        vert_vel_limit = 0.1 # meter/sec
        hvels = []
        pitches = []
        for i in range(len(vert_vel)):
            if abs( vert_vel[i] ) < vert_vel_limit:
                hvels.append( horiz_vel[i] )
                pitch = (orient_pitch_angle[i] + orient_pitch_angle[i+1])/2.0
                pitches.append( pitch )
        plot( hvels, pitches, 'k.' )
        xlabel( 'horizontal velocity (m/sec)' )
        ylabel( 'pitch angle (degrees)' )
        
    if plot_pos_and_vel:
        linewidth = 1.5
        subplot(3,1,1)
        title('ground speed, global reference frame')
        plot( t_P, P[:,0], 'rx', t_P, P[:,1], 'gx', t_P, P[:,2], 'bx' )
        smooth_lines = plot( t_P, Psmooth[:,0], 'r-', t_P, Psmooth[:,1], 'g-', t_P, Psmooth[:,2], 'b-' )
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Position\n(m)')
        grid()
        
        subplot(3,1,2)
        plot( t_dPdt, dPdt[:,0], 'rx', t_dPdt, dPdt[:,1], 'gx', t_dPdt, dPdt[:,2], 'bx', t_dPdt, nx.sqrt(nx.sum(dPdt**2,axis=1)), 'kx')
        smooth_lines = plot( t_dPdt, dPdt_smooth[:,0], 'r-', t_dPdt, dPdt_smooth[:,1], 'g-', t_dPdt, dPdt_smooth[:,2], 'b-', t_dPdt, nx.sqrt(nx.sum(dPdt_smooth**2,axis=1)), 'k-')
        legend(smooth_lines,('x','y','z','mag'))
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Velocity\n(m/sec)')
        grid()
        
        subplot(3,1,3)
        plot( t_d2Pdt2, d2Pdt2[:,0], 'r-', t_d2Pdt2, d2Pdt2[:,1], 'g-', t_d2Pdt2, d2Pdt2[:,2], 'b-' )
        smooth_lines = plot( t_d2Pdt2, d2Pdt2_smooth[:,0], 'r-', t_d2Pdt2, d2Pdt2_smooth[:,1], 'g-', t_d2Pdt2, d2Pdt2_smooth[:,2], 'b-' )
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Acceleration\n(m/sec/sec)')
        xlabel('Time (sec)')
        grid()

    elif plot_pos_err_histogram:

        #subplot(2,1,1)
        axes([.075,.575,.85,.375])
        x_err = list((Psmooth[:,0] - P[:,0])*1000.0)
        y_err = list((Psmooth[:,1] - P[:,1])*1000.0)
        z_err = list((Psmooth[:,2] - P[:,2])*1000.0)

        xlim = -.2,.2
        ylim = 0,20
        color_alpha = 0.5
        
        xlines = hist(x_err, bins = 17)[2]
##        ylabel('counts')
##        set(gca(),'ylim',ylim)
##        set(gca(),'xlim',xlim)
        set(xlines,'alpha',color_alpha)
        set(xlines,'facecolor',(1,0,0))

        ylines = hist(y_err, bins = 19)[2]
##        ylabel('counts')
##        set(gca(),'xlim',xlim)
##        set(gca(),'ylim',ylim)
        set(ylines,'alpha',color_alpha)
        set(ylines,'facecolor',(0,1,0))

        zlines = hist(z_err, bins = 50)[2]
        legend((xlines[0],ylines[0],zlines[0]),['X','Y','Z'])
        ylabel('counts')
        set(gca(),'ylim',ylim)
        set(gca(),'xlim',xlim)
        set(zlines,'alpha',color_alpha)
        set(zlines,'facecolor',(0,0,1))

        grid()
        xlabel('distance from smoothed data (mm)')

        #subplot(2,1,2)
        axes([.075,.0975,.85,.375])
        rad2deg = 180/math.pi
        euler_smooth = nx.array([quat_to_euler(q) for q in Qsmooth])*rad2deg
        euler = nx.array([quat_to_euler(q) for q in Q])*rad2deg
        
        yaw_err = list(euler_smooth[:,0] - euler[:,0])
        pitch_err = list(euler_smooth[:,1] - euler[:,1])
        roll_err = list(euler_smooth[:,2] - euler[:,2])

        xlim = -60,60
        ylim = 0,33
##        color_alpha = 0.6
        
        yawlines = hist(yaw_err, bins = 25)[2]
        ylabel('counts')
##        set(gca(),'ylim',ylim)
##        set(gca(),'xlim',xlim)
        set(yawlines,'alpha',color_alpha)
        set(yawlines,'facecolor',(1,0,0))

        pitchlines = hist(pitch_err, bins = 50)[2]
##        ylabel('counts')
        set(gca(),'xlim',xlim)
        set(gca(),'ylim',ylim)
        set(pitchlines,'alpha',color_alpha)
        set(pitchlines,'facecolor',(0,1,0))

        legend([yawlines[0],pitchlines[0]],['yaw','pitch'])

##        rolllines = hist(roll_err, bins = 5)[2]
##        legend((xlines[0],ylines[0],rolllines[0]),['yaw','pitch','roll'])
##        ylabel('counts')
##        set(gca(),'ylim',ylim)
##        set(gca(),'xlim',xlim)
##        set(rolllines,'alpha',color_alpha)
##        set(rolllines,'facecolor',(0,0,1))

        grid()
        xlabel('distance from smoothed data (deg)')

    elif plot_xy:
        axes([.1,.1,.8,.8])
        title('top view')

        if had_post:
            theta = linspace(0,2*math.pi,30)[:-1]
            postxs = post_radius*nx.cos(theta) + post_top_center[0]
            postys = post_radius*nx.sin(theta) + post_top_center[1]
            fill( postxs, postys )
        
##        title('top view (ground frame)')
        plot(P[:,0]*1000,P[:,1]*1000,'ko',mfc=(1,1,1),markersize=2)
##        plot(P[:,0]*1000,P[:,1]*1000,'ko',mfc=(1,1,1),markersize=4)
        
        for idx in range(len(t_P)):
            if idx%10==0:
                if xtitle == 'time':
                    text(P[idx,0]*1000,P[idx,1]*1000, str(t_P[idx]) )
                elif xtitle == 'frame':
                    text(P[idx,0]*1000,P[idx,1]*1000, str(frame[idx]) )
                
        #if do_smooth_position: # smoothed
        #    plot(Psmooth[:,0]*1000,Psmooth[:,1]*1000,'b-')

        for use_it, data, color in [[plot_xy_Qsmooth,Qsmooth,  (0,0,1,1)],
                                    [plot_xy_Qraw,   Q, (0,0,0,1)]]:
            if use_it:
                segments = []
                for i in range(len(P)):
                    pi = P[i]
                    qi = data[i]
                    Pqi = quat_to_orient(qi)
                    segment = ( (pi[0]*1000,  # x1
                                 pi[1]*1000), # y1
                                (pi[0]*1000-Pqi[0]*2,   # x2
                                 pi[1]*1000-Pqi[1]*2) ) # y2
                    segments.append( segment )

                collection = LineCollection(segments,
                                            colors=[color]*len(segments))
                gca().add_collection(collection)
        xlabel('x (mm)')
        ylabel('y (mm)')
        #t=text( 0.6, .2, '<- wind (0.4 m/sec)', transform = gca().transAxes)

        if show_grid:
            grid()

    elif plot_xz:
        axes([.1,.1,.8,.8])
        title('side view')
        #title('side view (ground frame)')

        if had_post:
            postxs = [post_top_center[0] + post_radius,
                      post_top_center[0] + post_radius,
                      post_top_center[0] - post_radius,
                      post_top_center[0] - post_radius]
            postzs = [post_top_center[2],
                      post_top_center[2] - post_height,
                      post_top_center[2] - post_height,
                      post_top_center[2]]
            fill( postxs, postzs )
        
        #plot(P[:,0]*1000,P[:,2]*1000,'ko',mfc=(1,1,1),markersize=4)
        plot(P[:,0]*1000,P[:,2]*1000,'ko',mfc=(1,1,1),markersize=2)
        
        for idx in range(len(t_P)):
            if idx%10==0:
                if xtitle == 'time':
                    text(P[idx,0]*1000,P[idx,2]*1000, str(t_P[idx]) )
                elif xtitle == 'frame':
                    text(P[idx,0]*1000,P[idx,2]*1000, str(frame[idx]) )
                
        if 0:
        #if do_smooth_position: # smoothed
            plot(Psmooth[:,0]*1000,Psmooth[:,2]*1000,'b-')

        for use_it, data, color in [[plot_xy_Qsmooth,Qsmooth,  (0,0,1,1)],
                                    [plot_xy_Qraw,   Q, (0,0,0,1)]]:
            if use_it:
                segments = []
                for i in range(len(P)):
                    pi = P[i]
                    qi = data[i]
                    Pqi = quat_to_orient(qi)
                    segment = ( (pi[0]*1000,  # x1
                                 pi[2]*1000), # y1
                                (pi[0]*1000-Pqi[0]*2,   # x2
                                 pi[2]*1000-Pqi[2]*2) ) # y2
                    segments.append( segment )

                collection = LineCollection(segments,
                                            colors=[color]*len(segments))
                gca().add_collection(collection)
        xlabel('x (mm)')
        ylabel('z (mm)')
##        t=text( 0, 1.0, '<- wind (0.4 m/sec)',
###        t=text( 0.6, .2, '<- wind (0.4 m/sec)',
##                transform = gca().transAxes,
##                horizontalalignment = 'left',
##                verticalalignment = 'top',
##                )

        if show_grid:
            grid()

    elif plot_xy_air:
        axes([.1,.1,.8,.8])
        title('position (wind frame)')
        
        xairvel = 0.4 # m/sec
        xairvel = xairvel / 100.0 # 100 positions/sec
        
        Pair = P.copy()
        for i in range(len(Pair)):
            Pair[i,0] = P[i,0]+xairvel*i
        plot(Pair[:,0]*1000,Pair[:,1]*1000,'ko',mfc=(1,1,1),markersize=2)

        Psmooth_air = Psmooth.copy()
        for i in range(len(Psmooth_air)):
            Psmooth_air[i,0] = Psmooth[i,0]+xairvel*i
        
        for idx in range(len(t_P)):
            if idx%10==0:
                if xtitle == 'time':
                    text(P[idx,0]*1000,P[idx,1]*1000, str(t_P[idx]) )
                elif xtitle == 'frame':
                    text(P[idx,0]*1000,P[idx,1]*1000, str(frame[idx]) )
            
##        if do_smooth_position: # smoothed
##            plot(Psmooth_air[:,0]*1000,Psmooth_air[:,1]*1000,'b-')

        for use_it, data, color in [[plot_xy_Qsmooth,Qsmooth,  (0,0,1,1)],
                                    [plot_xy_Qraw,   Q, (0,0,0,1)]]:
            if use_it:
                segments = []
                for i in range(len(Pair)):
                    pi = Pair[i]
                    qi = data[i]
                    Pqi = quat_to_orient(qi)
                    segment = ( (pi[0]*1000,  # x1
                                 pi[1]*1000), # y1
                                (pi[0]*1000-Pqi[0]*2,   # x2
                                 pi[1]*1000-Pqi[1]*2) ) # y2
                    segments.append( segment )

                collection = LineCollection(segments,
                                            colors=[color]*len(segments))
                gca().add_collection(collection)
        xlabel('x (mm)')
        ylabel('y (mm)')
        grid()

    elif plot_accel:
        subplot(3,1,1)
        plot( t_d2Pdt2, d2Pdt2[:,0], 'r-' )
        grid()
        
        subplot(3,1,2)
        plot( t_d2Pdt2, d2Pdt2[:,1], 'g-' )
        grid()
        
        subplot(3,1,3)
        plot( t_d2Pdt2, d2Pdt2[:,2], 'b-' )
        grid()
    elif plot_smooth_pos_and_vel:
        linewidth = 1.5
        subplot(3,1,1)
        title('Global reference frame, ground speed')
        raw_lines = plot( t_P, P[:,0], 'rx', t_P, P[:,1], 'gx', t_P, P[:,2], 'bx' )
        smooth_lines = plot( t_P, Psmooth[:,0], 'r-', t_P, Psmooth[:,1], 'g-', t_P, Psmooth[:,2], 'b-' )
        legend(smooth_lines,('X','Y','Z'),2)
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Position (m)')
        grid()
        
        subplot(3,1,2)
        raw_lines = plot( t_dPdt, dPdt[:,0], 'rx', t_dPdt, dPdt[:,1], 'gx', t_dPdt, dPdt[:,2], 'bx' )
        smooth_lines = plot( t_dPdt, dPdt_smooth[:,0], 'r-', t_dPdt, dPdt_smooth[:,1], 'g-', t_dPdt, dPdt_smooth[:,2], 'b-' )
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Velocity (m/sec)')
        grid()

        subplot(3,1,3)
        raw_lines = plot( t_d2Pdt2, d2Pdt2[:,0], 'rx', t_d2Pdt2, d2Pdt2[:,1], 'gx', t_d2Pdt2, d2Pdt2[:,2], 'bx' )
        smooth_lines = plot( t_d2Pdt2, d2Pdt2_smooth[:,0], 'r-', t_d2Pdt2, d2Pdt2_smooth[:,1], 'g-', t_d2Pdt2, d2Pdt2_smooth[:,2], 'b-' )
        set(smooth_lines,'linewidth',linewidth)
        ylabel('Acceleration (m/sec/sec)')
        xlabel('Time (sec)')
        grid()
    elif plot_Q:
        linewidth = 1.5
        subplot(3,1,1)
        title('quaternions in R4')
        raw_lines = plot( t_P, Q.w, 'kx', t_P, Q.x, 'rx',
                          t_P, Q.y, 'gx', t_P, Q.z, 'bx')
        if do_smooth_quats:
            smooth_lines = plot( t_P, Qsmooth.w, 'k', t_P, Qsmooth.x, 'r',
                                 t_P, Qsmooth.y, 'g', t_P, Qsmooth.z, 'b')
            set(smooth_lines,'linewidth',linewidth)
            legend(smooth_lines,['w','x','y','z'])
        else:
            legend(raw_lines,['w','x','y','z'])
        ylabel('orientation')
        grid()

        subplot(3,1,2)
        if 0:
            # only plot raw if no smooth (derivatives of raw data are very noisy)
            raw_lines = plot( t_omega, omega.w, 'kx', t_omega, omega.x, 'rx',
                              t_omega, omega.y, 'gx', t_omega, omega.z, 'bx')
        if do_smooth_quats:
            rad2deg = 180/math.pi
            mag_omega = nx.array([ abs(q) for q in omega_smooth ])*rad2deg
##            print 't_omega.shape',t_omega.shape
##            print 'mag_omega.shape',mag_omega.shape
##            smooth_lines = plot( t_omega, mag_omega)
##            smooth_lines = plot( t_omega, nx.arctan2(omega_smooth.y, omega_smooth.x)*rad2deg, 'k-')
            smooth_lines = plot( t_omega, omega_smooth.w, 'k', t_omega, omega_smooth.x, 'r',
                                 t_omega, omega_smooth.y, 'g', t_omega, omega_smooth.z, 'b')
            set(smooth_lines,'linewidth',linewidth)
            legend(smooth_lines,['w','x','y','z'])
        else:
            legend(raw_lines,['w','x','y','z'])
        ylabel('omega')
        xlabel('Time (sec)')
        grid()
        
        subplot(3,1,3)
        if 0:
            # only plot raw if no smooth (derivatives of raw data are very noisy)
            raw_lines = plot( t_omega_dot, omega_dot.w, 'kx', t_omega_dot, omega_dot.x, 'rx',
                              t_omega_dot, omega_dot.y, 'gx', t_omega_dot, omega_dot.z, 'bx')
        if do_smooth_quats:
            smooth_lines = plot( t_omega_dot, omega_dot_smooth.w, 'k', t_omega_dot, omega_dot_smooth.x, 'r',
                                 t_omega_dot, omega_dot_smooth.y, 'g', t_omega_dot, omega_dot_smooth.z, 'b')
            set(smooth_lines,'linewidth',linewidth)
            legend(smooth_lines,['w','x','y','z'])
        else:
            legend(raw_lines,['w','x','y','z'])
        ylabel('omega dot')
        xlabel('Time (sec)')
        grid()

    elif plot_body_angular_vel:
        rad2deg = 180/math.pi
        linewidth = 1.5
        smooth = 1
        rad2deg = 180/math.pi
        fontsize = 10

        ax1 = subplot(3,1,1)
        title('angles and angular velocities')
        euler_smooth = nx.array([quat_to_euler(q) for q in Qsmooth])*rad2deg
        euler = nx.array([quat_to_euler(q) for q in Q])*rad2deg
        yaw = euler[:,0]; pitch = euler[:,1]; roll = euler[:,2]
        yaw_smooth = euler_smooth[:,0]; pitch_smooth = euler_smooth[:,1]; roll_smooth = euler_smooth[:,2]
        if xtitle == 'time':
            xdata = t_P
        elif xtitle == 'frame':
            xdata = t_P*100 + start_frame
        lines = plot(xdata, yaw, 'r-', xdata, pitch, 'g-', xdata, roll, 'b-')
        lines_smooth = plot(xdata, yaw_smooth, 'r-', xdata, pitch_smooth, 'g-', xdata, roll_smooth, 'b-')
        if xtitle == 'time':
            plot(t_bad,[0.0]*len(t_bad),'ko')
        elif xtitle == 'frame':
            plot(frame_bad,[0.0]*len(frame_bad),'ko')
        set(lines_smooth,'lw',linewidth)
        legend(lines,['heading','pitch (body)','roll'])
        ylabel('angular position (global)\n(deg)')
        set(gca().yaxis.label,'size',fontsize)
        set(gca(),'ylim',(-15,75))
        grid()

        plot_mag = False
        plot_roll = False
        subplot(3,1,2, sharex=ax1)
        if xtitle == 'time':
            xdata = t_omega
        elif xtitle == 'frame':
            xdata = t_omega*100 + start_frame
        if plot_mag:
            mag_omega = nx.array([ abs(q) for q in omega ])*rad2deg
            args = [ xdata, mag_omega, 'k-']
            line_titles = ['mag']
        else:
            args = []
            line_titles = []
        args.extend( [xdata, omega.z*rad2deg, 'r-', xdata, omega.y*rad2deg, 'g-'] )
        line_titles.extend( ['heading','pitch (body)'] )
        if plot_roll:
            args.extend( [ xdata, omega.x*rad2deg, 'b-'] )
            line_titles.extend( ['roll'] )
        lines=plot( *args )
        if plot_mag:
            mag_omega = nx.array([ abs(q) for q in omega_smooth ])*rad2deg
            args = [xdata, mag_omega, 'k-' ]
        else:
            args = []
        args.extend( [xdata, omega_smooth.z*rad2deg, 'r-', xdata, omega_smooth.y*rad2deg, 'g-'] )
        if plot_roll:
            args.extend( [xdata, omega_smooth.x*rad2deg, 'b-'] )
        lines_smooth=plot( *args )
        set(lines_smooth,'lw',linewidth)
        legend(lines,line_titles)
        ylabel('angular velocity\nglobal frame (deg/sec)')
        set(gca().yaxis.label,'size',fontsize)
        set(gca(),'ylim',[-750,600])
        grid()

        subplot(3,1,3, sharex=ax1)
        if plot_mag:
            mag_omega_body = nx.array([ abs(q) for q in omega_body ])*rad2deg
            args = [ xdata, mag_omega_body, 'k-' ]
            line_titles = ['mag']
        else:
            args = []
            line_titles = []
        args.extend([xdata, omega_body.z*rad2deg, 'r-', xdata, omega_body.y*rad2deg, 'g-'])
        line_titles.extend( ['yaw','pitch'])
        if plot_roll:
            args.extend([xdata, omega_body.x*rad2deg, 'b-'])
            line_titles.extend( ['roll'])
        lines = plot(*args)
        legend(lines,line_titles)

        if plot_mag:
            mag_omega_body = nx.array([ abs(q) for q in omega_smooth_body ])*rad2deg
            args = [ xdata, mag_omega_body, 'k-' ]
        else:
            args = []
        args.extend( [ xdata, omega_smooth_body.z*rad2deg, 'r-', xdata, omega_smooth_body.y*rad2deg, 'g-'] )
        if plot_roll:
            args.extend( [ xdata, omega_smooth_body.x*rad2deg, 'b-' ])
        lines_smooth=plot( *args)
        set(lines_smooth,'lw',linewidth)
        ylabel('angular velocity\nbody frame (deg/sec)')
        set(gca().yaxis.label,'size',fontsize)
        set(gca(),'ylim',[-500,500])
        if xtitle == 'time':
            xlabel('time (sec)')
        elif xtitle == 'frame':
            xlabel('frame')
        grid()

    elif plot_error_angles:
        # plot
        linewidth=1.5
        subplot(2,1,1)
        title('orientation - course direction = error in earlier work')
        plot(t_heading_err,heading_err,'b-')
        heading_lines = plot(t_heading_err_smooth,heading_err_smooth,'b-',lw=linewidth)
        plot(t_pitch_body_err,pitch_body_err,'r-')
        pitch_lines = plot(t_pitch_body_err_smooth,pitch_body_err_smooth,'r-',lw=linewidth)
        legend((heading_lines[0],pitch_lines[0]),('heading','pitch (body)'))
        ylabel('angle (deg)')
        grid()

        subplot(2,1,2)
        plot( t_d_heading_err_dt, d_heading_err_dt,'b-')
        plot( t_d_heading_err_smooth_dt, d_heading_err_smooth_dt,'b-',lw=linewidth)
        plot(t_d_pitch_body_err_dt,d_pitch_body_err_dt,'r-')
        plot(t_d_pitch_body_err_smooth_dt,d_pitch_body_err_smooth_dt,'r-',lw=linewidth)
        set(gca(),'ylim',[-2000,2000])
        ylabel('anglular velocity (deg/sec)')
        xlabel('time (sec)')
        grid()
    elif plot_body_ground_V:
        linewidth = 1.5
        subplot(4,1,1)
        title('groundspeed (body frame)')
        plot(t_dPdt,body_ground_V.x,'kx')
        plot(t_dPdt,body_ground_V_smooth.x,'b-',lw=linewidth)
        ylabel('forward\n(m/sec)')
        grid()
        subplot(4,1,2)
        plot(t_dPdt,body_ground_V.y,'kx')
        plot(t_dPdt,body_ground_V_smooth.y,'b-',lw=linewidth)
        ylabel('sideways\n(m/sec)')
        grid()
        subplot(4,1,3)
        plot(t_dPdt,body_ground_V.z,'kx')
        plot(t_dPdt,body_ground_V_smooth.z,'b-',lw=linewidth)
        ylabel('upward\n(m/sec)')
        grid()
        subplot(4,1,4)
        raw_norm = nx.sqrt(body_ground_V.x**2 + body_ground_V.y**2 + body_ground_V.z**2)
        plot(t_dPdt,raw_norm,'kx')
        smooth_norm = nx.sqrt(body_ground_V_smooth.x**2 + body_ground_V_smooth.y**2 + body_ground_V_smooth.z**2)
        plot(t_dPdt,smooth_norm,'b-',lw=linewidth)
        ylabel('|V|\n(m/sec)')
        xlabel('time (sec)')
        grid()
    elif plot_body_air_V:
        linewidth = 1.5
        subplot(4,1,1)
        title('airspeed (body frame)')
        plot(t_dPdt,body_air_V.x,'kx')
        plot(t_dPdt,body_air_V_smooth.x,'b-',lw=linewidth)
        ylabel('forward\n(m/sec)')
        subplot(4,1,2)
        plot(t_dPdt,body_air_V.y,'kx')
        plot(t_dPdt,body_air_V_smooth.y,'b-',lw=linewidth)
        ylabel('sideways\n(m/sec)')
        subplot(4,1,3)
        plot(t_dPdt,body_air_V.z,'kx')
        plot(t_dPdt,body_air_V_smooth.z,'b-',lw=linewidth)
        ylabel('upward\n(m/sec)')
        subplot(4,1,4)
        raw_norm = nx.sqrt(body_air_V.x**2 + body_air_V.y**2 + body_air_V.z**2)
        plot(t_dPdt,raw_norm,'kx')
        smooth_norm = nx.sqrt(body_air_V_smooth.x**2 + body_air_V_smooth.y**2 + body_air_V_smooth.z**2)
        plot(t_dPdt,smooth_norm,'b-',lw=linewidth)
        ylabel('|V|\n(m/sec)')
        xlabel('time (sec)')
    elif plot_forces:

        subplot(2,1,1)
        title('predicted aerodynamic forces on body')
        lines = plot(t_forces,F_P,'b-', t_forces, F_N,'r-',lw=1.5)
        ylabel('force (N)')
        legend(lines,['parallel','normal'])
        ylim = get(gca(),'ylim')
        set(gca(),'ylim',[0,ylim[1]])
        text( .1, .9, 'Force to keep 1 mg aloft: %.1e'%aloft_force,
              transform = gca().transAxes)

        subplot(2,1,2)
        aattack_lines = plot(t_dPdt,aattack*rad2deg,lw=1.5)
        ylabel('alpha (deg)')
        gca().yaxis.tick_left()
        
        subplot(2,1,2,frameon=False)
        vel_lines = plot(t_dPdt,nx.sqrt(nx.sum(dPdt_smooth_air**2,axis=1)),'k',lw=1.5)
        gca().yaxis.tick_right()

##        if 1:
##            print 'gca().dataLim',gca().dataLim
##            outputs.append( gca().dataLim )
##            print 'gca().viewLim',gca().viewLim
##            outputs.append( gca().viewLim )
            

        legend((aattack_lines[0],vel_lines[0]),('alpha','|V|'))
        xlabel('time (sec)')
              
    return outputs
