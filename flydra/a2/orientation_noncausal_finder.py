from __future__ import division
from __future__ import with_statement

import flydra.analysis.result_utils as result_utils
import flydra.a2.core_analysis as core_analysis
import tables
import numpy as np
import flydra.reconstruct as reconstruct
import collections

import matplotlib.pyplot as plt

from image_based_orientation import openFileSafe
import flydra.kalman.ekf as kalman_ekf
import flydra.analysis.PQmath as PQmath
import cgtypes # cgkit 1.x
import sympy
import sys

ca = core_analysis.get_global_CachingAnalyzer()

slope2modpi = np.arctan # assign function name
D2R = 180.0/np.pi
gate_angle_threshold_radians = 20.0*D2R

if 1:
    a,b,c,d=sympy.symbols('abcd')
    R = sympy.Matrix([[d**2+a**2-b**2-c**2, 2*(a*b-c*d), 2*(a*c+b*d)],
                      [2*(a*b+c*d), d**2+b**2-a**2-c**2, 2*(b*c-a*d)],
                      [2*(a*c-b*d), 2*(b*c+a*d), d**2+c**2-b**2-a**2]])
    u = sympy.Matrix([[1],[0],[0]])
    U=R*u

    P00 = sympy.Symbol('P00')
    P01 = sympy.Symbol('P01')
    P02 = sympy.Symbol('P02')
    P03 = sympy.Symbol('P03')

    P10 = sympy.Symbol('P10')
    P11 = sympy.Symbol('P11')
    P12 = sympy.Symbol('P12')
    P13 = sympy.Symbol('P13')

    P20 = sympy.Symbol('P20')
    P21 = sympy.Symbol('P21')
    P22 = sympy.Symbol('P22')
    P23 = sympy.Symbol('P23')

    Ax = sympy.Symbol('Ax')
    Ay = sympy.Symbol('Ay')
    Az = sympy.Symbol('Az')
    A = sympy.Matrix([[Ax],[Ay],[Az]])
    hA = sympy.Matrix([[Ax],[Ay],[Az],[1]])

    P = sympy.Matrix([[P00,P01,P02,P03],
                      [P10,P11,P12,P13],
                      [P20,P21,P22,P23]])

    B = A+U
    hB = sympy.Matrix([[B[0]],[B[1]],[B[2]],[1]])

    ha = P*hA
    hb = P*hB

    a2 = sympy.Matrix([[ha[0]/ha[2]],[ha[1]/ha[2]]])
    b2 = sympy.Matrix([[hb[0]/hb[2]],[hb[1]/hb[2]]])

    vec = b2-a2
    dy = vec[1]
    dx = vec[0]

    theta = sympy.atan(dy/dx)

    sympy.pprint(hB)
    #sympy.pprint(R)
    #sympy.pprint(u)
    sympy.pprint(U)
    sympy.pprint(a)
    sympy.pprint(b)
    sympy.pprint(theta)

    print
    print sympy.latex(R)
    print
    print sympy.latex(R*u)
    print
    print sympy.latex(P)
    print
    print sympy.latex(theta)

    sys.exit(0)

def statespace2cgtypes_quat( x ):
    return cgtypes.quat( x[6], x[3], x[4], x[5] )

def get_point_on_line( x, A, mu=1.0 ):
    """get a point on a line through A specified by state space vector x
    """
    # x is our state space vector
    q = statespace2cgtypes_quat(x)
    return mu*np.asarray(PQmath.quat_to_orient(q))+A

def find_theta_mod_pi_between_points(a,b):
    diff = a-b
    dx,dy=diff
    if dx==0.0:
        return np.pi/2
    return slope2modpi(dy/dx)

if 1:
    from sympy import Symbol, Matrix, sqrt, latex

    # This formulation is from Marins, Yun, Bachmann, McGhee, and Zyda
    # (2001). An Extended Kalman Filter for Quaternion-Based
    # Orientation Estimation Using MARG Sensors. Proceedings of the
    # 2001 IEEE/RSJ International Conference on Intelligent Robots and
    # Systems.

    tau_rx = Symbol('tau_rx')
    tau_ry = Symbol('tau_ry')
    tau_rz = Symbol('tau_rz')

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4') # quat x
    x5 = Symbol('x5') # quat y
    x6 = Symbol('x6') # quat z
    x7 = Symbol('x7') # quat w (scalar)

    # eqns 9-15
    f1 = -1/tau_rx*x1
    f2 = -1/tau_ry*x2
    f3 = -1/tau_rz*x3
    scale = 2*sqrt(x4**2 + x5**2 + x6**2 + x7**2)
    f4 = 1/scale * ( x3*x5 - x2*x6 + x1*x7 )
    f5 = 1/scale * (-x3*x4 + x1*x6 + x2*x7 )
    f6 = 1/scale * ( x2*x4 - x1*x5 + x3*x7 )
    f7 = 1/scale * (-x1*x4 - x2*x5 + x3*x6 )

    derivative_x = (f1,f2,f3,f4,f5,f6,f7)
    derivative_x = Matrix(derivative_x).T

    #print derivative_x
    x = (x1,x2,x3,x4,x5,x6,x7)

    dx_symbolic = derivative_x.jacobian(x)
    #print A

    ## print latex(derivative_x)
    ## print
    ## print latex(dx_symbolic)

    if 0:
        sdict = {x1:1.,
                 x2:2.,
                 x3:3.,
                 x4:4.,
                 x5:5.,
                 x6:6.,
                 x7:7.,

                 tau_rx : 1,
                 tau_ry : 1,
                 tau_rz : 1,
                 }


        A_evaluated = dx_symbolic.evalf( subs=sdict )
        import sympy
        #sympy.pngview(A_evaluated)
        sympy.pprint(A_evaluated)
        A_evaluated = np.asarray(A_evaluated)
        print A_evaluated
        #print latex(A_evaluated)

def eval_deriv(x):
    """evaluate derivative f(x) at x"""
    sdict = {x1:float(x[0]),
             x2:float(x[1]),
             x3:float(x[2]),
             x4:float(x[3]),
             x5:float(x[4]),
             x6:float(x[5]),
             x7:float(x[6]),

             tau_rx : 1,
             tau_ry : 1,
             tau_rz : 1,
             }
    return np.asarray(dx_symbolic.evalf( subs=sdict ))

if 1:
    start = stop = None
    use_obj_ids = [19]

    with openFileSafe( 'DATA20080915_153202.image-based-re2d.kalmanized.h5',
                       mode='r') as kh5:
        with openFileSafe( 'DATA20080915_153202.image-based-re2d.h5',
                           mode='r') as h5:
            reconst = reconstruct.Reconstructor(kh5)

            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
            fps = result_utils.get_fps(h5)
            dt = 1.0/fps

            # associate framenumbers with timestamps using 2d .h5 file
            data2d = h5.root.data2d_distorted[:] # load to RAM
            data2d_idxs = np.arange(len(data2d))
            h5_framenumbers = data2d['frame']
            h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

            kalman_observations_2d_idxs = (
                kh5.root.kalman_observations_2d_idxs[:])

            for obj_id_enum,obj_id in enumerate(use_obj_ids):
            # Use data association step from kalmanization to load potentially
            # relevant 2D orientations, but discard previous 3D orientation.

                obj_3d_rows = ca.load_dynamics_free_MLE_position( obj_id, kh5)

                smoothed_3d_rows = ca.load_data(
                    obj_id, kh5,
                    frames_per_second=fps,
                    dynamic_model_name='mamarama, units: mm')
                smoothed_frame_qfi = result_utils.QuickFrameIndexer(
                    smoothed_3d_rows['frame'])

                slopes_by_camn_by_frame = collections.defaultdict(dict)
                min_frame = np.inf
                max_frame = -np.inf
                for this_3d_row in obj_3d_rows:
                    # iterate over each sample in the current camera
                    framenumber = this_3d_row['frame']
                    if framenumber < min_frame:
                        min_frame = framenumber
                    if framenumber > max_frame:
                        max_frame = framenumber

                    if start is not None:
                        if not framenumber >= start:
                            continue
                    if stop is not None:
                        if not framenumber <= stop:
                            continue
                    h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)

                    frame2d = data2d[h5_2d_row_idxs]
                    frame2d_idxs = data2d_idxs[h5_2d_row_idxs]

                    obs_2d_idx = this_3d_row['obs_2d_idx']
                    kobs_2d_data = kalman_observations_2d_idxs[int(obs_2d_idx)]

                    # Parse VLArray.
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]

                    # Now, for each camera viewing this object at this
                    # frame, extract images.
                    for camn, camn_pt_no in zip(this_camns, this_camn_idxs):
                        # find 2D point corresponding to object
                        cam_id = camn2cam_id[camn]

                        cond = ((frame2d['camn']==camn) &
                                (frame2d['frame_pt_idx']==camn_pt_no))
                        idxs = np.nonzero(cond)[0]
                        assert len(idxs)==1
                        idx = idxs[0]

                        orig_data2d_rownum = frame2d_idxs[idx]
                        frame_timestamp = frame2d[idx]['timestamp']

                        row = frame2d[idx]
                        assert framenumber==row['frame']
                        slopes_by_camn_by_frame[camn][framenumber] = (
                            row['slope'])

                # now collect in a numpy array for all cam

                frame_range = np.arange(min_frame,max_frame+1)
                camn_list = slopes_by_camn_by_frame.keys()
                camn_list.sort()
                cam_id_list = [camn2cam_id[camn] for camn in camn_list]
                n_cams = len(camn_list)
                n_frames = len(frame_range)

                # NxM array with rows being frames and cols being cameras
                slopes = np.ones( (n_frames,n_cams), dtype=np.float)
                for j,camn in enumerate(camn_list):

                    slopes_by_frame = slopes_by_camn_by_frame[camn]

                    for frame_idx,absolute_frame_number in enumerate(
                        frame_range):

                        slopes[frame_idx,j] = slopes_by_frame.get(
                            absolute_frame_number,np.nan)

                    ## plt.plot(frame_range,slope2modpi(slopes[:,j]),'.',
                    ##          label=camn2cam_id[camn])

                ## plt.legend()
                ## plt.savefig('fig_ori.png')

                if 1:
                    # guesstimate initial orientation (XXX not done)
                    up_vec = 0,0,1
                    q0 = PQmath.orientation_to_quat( up_vec )
                    w0 = 0,0,0 # no angular rate

                    xhatminus = w0[0],w0[1],w0[2], q0.x, q0.y, q0.z, q0.w
                    print 'xhatminus',xhatminus

                    Pminus = np.zeros((7,7))

                    # angular rate part of state variance is .5
                    for i in range(0,3):
                        Pminus[i,i] = .5

                    # quaternion part of state variance is 1
                    for i in range(3,7):
                        Pminus[i,i] = 1

                if 1:
                    # setup of noise estimates
                    Q = np.zeros((7,7))

                    # angular rate part of state variance is .5
                    for i in range(0,3):
                        Q[i,i] = .5

                    # quaternion part of state variance is 1
                    for i in range(3,7):
                        Q[i,i] = 1

                preA = np.eye(7)

                ekf = kalman_ekf.EKF( xhatminus, Pminus )
                for frame_idx, absolute_frame_number in enumerate(frame_range):

                    # Evaluate the Jacobian of the process update
                    # using previous frame's posterior estimate. (This
                    # is not quite the same as this frame's prior
                    # estimate.)

                    this_dx = eval_deriv( xhatminus )
                    A = preA + this_dx*dt
                    print
                    print 'frame',absolute_frame_number
                    print 'past posterior',xhatminus
                    print 'A'
                    print A
                    xhatminus, Pminus=ekf.step1__calculate_a_priori(A,Q)
                    print 'new prior',xhatminus

                    # 1. Predict per-camera orientations for cameras
                    # with ori info.

                    # 2. Gate per-camera orientations (on Mahalanobis
                    # distance?).

                    this_frame_slopes = slopes[frame_idx,:]
                    print 'this_frame_slopes',this_frame_slopes

                    missing_data = False
                    gate_vector=None
                    y=None
                    C=None
                    R=None
                    cams_without_data = np.isnan( this_frame_slopes )
                    if np.all(cams_without_data):
                        missing_data = True

                    if not missing_data:
                        smoothed_pos_idxs = smoothed_frame_qfi.get_frame_idxs(
                            absolute_frame_number)
                        assert len(smoothed_pos_idxs)==1
                        smoothed_pos_idx = smoothed_pos_idxs[0]
                        smooth_row = smoothed_3d_rows[smoothed_pos_idx]
                        assert smooth_row['frame'] == absolute_frame_number
                        center_position = np.array((smooth_row['x'],
                                                    smooth_row['y'],
                                                    smooth_row['z']))

                        print 'center_position',center_position
                        other_position = get_point_on_line(xhatminus,
                                                           center_position)
                        print 'other_position',other_position
                        cams_with_data = ~cams_without_data
                        possible_cam_idxs = np.nonzero(cams_with_data)[0]
                        print 'possible_cam_idxs',possible_cam_idxs
                        gate_vector = np.zeros( (n_cams,), dtype=np.bool)
                        flip_vector = np.zeros( (n_cams,), dtype=np.bool)
                        for camn_idx in possible_cam_idxs:
                            cam_id = cam_id_list[camn_idx]

                            # This ignores distortion. To incorporate
                            # distortion, this could would require
                            # appropriate scaling of orientation
                            # vector.

                            a = reconst.find2d( cam_id, center_position)
                            b = reconst.find2d( cam_id, other_position)

                            print 'cam_id',cam_id
                            theta_expected=find_theta_mod_pi_between_points(a,b)
                            theta_measured=slope2modpi(
                                this_frame_slopes[camn_idx])
                            print 'theta_expected,theta_measured',theta_expected,theta_measured
                            if reconstruct.angles_near(
                                theta_expected,theta_measured,
                                gate_angle_threshold_radians,
                                mod_pi=True):
                                gate_vector[camn_idx]=1
                                print 'good'

                                if not reconstruct.angles_near(
                                    theta_expected,theta_measured+np.pi,
                                    gate_angle_threshold_radians,
                                    mod_pi=False):
                                    flip_vector[camn_idx]=1
                                    print 'flipped'
                        print 'gate_vector',gate_vector
                        print 'flip_vector',flip_vector
                        missing_data = not bool(np.sum(gate_vector))

                    # 3. Construct observations model using all
                    # gated-in camera orientations.

                    if not missing_data:
                        print gate_vector

                    print 'missing_data',missing_data
                    ekf.step2__calculate_a_posteriori(xhatminus, Pminus, y=y,
                                                      C=C,R=R,
                                                      missing_data=missing_data)

