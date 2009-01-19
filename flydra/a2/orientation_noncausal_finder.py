from __future__ import division
from __future__ import with_statement

import flydra.analysis.result_utils as result_utils
import flydra.a2.core_analysis as core_analysis
import tables
import numpy as np
import flydra.reconstruct as reconstruct
import collections

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from image_based_orientation import openFileSafe,clear_col
import flydra.kalman.ekf as kalman_ekf
import flydra.analysis.PQmath as PQmath
import flydra.geom as geom
import cgtypes # cgkit 1.x
import sympy
import sys
from sympy import Symbol, Matrix, sqrt, latex, lambdify
import pickle

D2R = np.pi/180.0
R2D = 180.0/np.pi

# configuration and constants (important stuff)
expected_orientation_method = 'trust_prior'

Q_scalar_rate = 0.1
Q_scalar_quat = 0.1
R_scalar = 10

gate_angle_threshold_radians = 100.0*D2R # everything?
area_threshold_for_orientation = 500

# everything else
np.set_printoptions(linewidth=130,suppress=True)

ca = core_analysis.get_global_CachingAnalyzer()
slope2modpi = np.arctan # assign function name

def statespace2cgtypes_quat( x ):
    return cgtypes.quat( x[6], x[3], x[4], x[5] )

def state_to_ori(x):
    q = statespace2cgtypes_quat( x )
    return PQmath.quat_to_orient(q)

def state_to_hzline(x,A):
    Ax,Ay,Az = A[0], A[1], A[2]
    Ux,Uy,Uz = state_to_ori(x)
    line = geom.line_from_points( geom.ThreeTuple((Ax,Ay,Az)),
                                  geom.ThreeTuple((Ax+Ux,Ay+Uy,Az+Uz)) )
    return line.to_hz()

## state = (0,0,0, 0,1,0,0)
## A = (0,0,0)
## hz = state_to_hzline(state,A)
## print hz
## 1/0
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

class drop_dims(object):
    def __init__(self,func):
        self.func = func
    def __call__(self,*args,**kwargs):
        mat = self.func(*args,**kwargs)
        arr2d = np.array(mat)
        assert len(arr2d.shape)==2
        if arr2d.shape[0]==1:
            arr1d = arr2d[0,:]
            return arr1d
        return arr2d


def get_theta_symbolic(x):

    # This formulation partly from Marins, Yun, Bachmann, McGhee, and
    # Zyda (2001). An Extended Kalman Filter for Quaternion-Based
    # Orientation Estimation Using MARG Sensors. Proceedings of the
    # 2001 IEEE/RSJ International Conference on Intelligent Robots and
    # Systems.

    # nomenclature match
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]

    # Observation model per camera

    a = x4
    b = x5
    c = x6
    d = x7

    # rotation matrix (eqn 6)
    R = Matrix([[d**2+a**2-b**2-c**2, 2*(a*b-c*d), 2*(a*c+b*d)],
                [2*(a*b+c*d), d**2+b**2-a**2-c**2, 2*(b*c-a*d)],
                [2*(a*c-b*d), 2*(b*c+a*d), d**2+c**2-b**2-a**2]])
    # default orientation with no rotation
    u = Matrix([[1],[0],[0]])

    # rotated orientation
    U=R*u

    # point in space
    Ax = sympy.Symbol('Ax')
    Ay = sympy.Symbol('Ay')
    Az = sympy.Symbol('Az')
    A = sympy.Matrix([[Ax],[Ay],[Az]]) # make vector
    hA = sympy.Matrix([[Ax],[Ay],[Az],[1]]) # homogenous

    # second point in space, the two of which define line
    B = A+U
    hB = sympy.Matrix([[B[0]],[B[1]],[B[2]],[1]]) # homogenous

    # camera matrix
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
    P = sympy.Matrix([[P00,P01,P02,P03],
                      [P10,P11,P12,P13],
                      [P20,P21,P22,P23]])

    # the image projection of points on line
    ha = P*hA
    hb = P*hB

    # de homogenize
    a2 = sympy.Matrix([[ha[0]/ha[2]],[ha[1]/ha[2]]])
    b2 = sympy.Matrix([[hb[0]/hb[2]],[hb[1]/hb[2]]])

    # direction in image
    vec = b2-a2

    # rise and run
    dy = vec[1]
    dx = vec[0]

    # prefer atan over atan2 because observations are mod pi.
    theta = sympy.atan(dy/dx)

    arg_tuple = (P00, P01, P02, P03,
                 P10, P11, P12, P13,
                 P20, P21, P22, P23,
                 Ax, Ay, Az,
                 x)

    # Process model

    if 0:
        tau_rx = Symbol('tau_rx')
        tau_ry = Symbol('tau_ry')
        tau_rz = Symbol('tau_rz')
    else:
        tau_rx = 0.01
        tau_ry = 0.01
        tau_rz = 0.01

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

    dx_symbolic = derivative_x.jacobian((x1,x2,x3,x4,x5,x6,x7))

    return theta, dx_symbolic, arg_tuple

if 1:
    x = sympy.DeferredVector('x')
    theta, dx_symbolic, arg_tuple_x = get_theta_symbolic(x)

    ## sympy.pprint(hB)
    ## #sympy.pprint(R)
    ## #sympy.pprint(u)
    ## sympy.pprint(U)
    ## sympy.pprint(a)
    ## sympy.pprint(b)
    if 1:
        print 'theta'
        sympy.pprint(theta)
        print

    theta_linearized = [ theta.diff(x[i]) for i in range(7)]
    if 0:
        print 'theta_linearized'
        for i in range(len(theta_linearized)):
            sympy.pprint(theta_linearized[i])
        print

eval_G = lambdify(arg_tuple_x, theta, 'numpy')
eval_linG=lambdify(arg_tuple_x, theta_linearized, 'numpy')

if 1:
    if 0:
        print 'dx_symbolic'
        sympy.pprint(dx_symbolic)
        print

eval_dAdt = drop_dims( lambdify( x,dx_symbolic,'numpy'))

if 1:
    start = stop = None
    use_obj_ids = [19]
    debug_level = 0

    with openFileSafe( 'DATA20080915_153202.image-based-re2d.kalmanized.h5',
                       mode='r+') as kh5:
        with openFileSafe( 'DATA20080915_153202.image-based-re2d.h5',
                           mode='r') as h5:
            xhat_results = {}
            print 'clearing columns'
            clear_col( kh5.root.kalman_observations,'hz_line0')
            clear_col( kh5.root.kalman_observations,'hz_line1')
            clear_col( kh5.root.kalman_observations,'hz_line2')
            clear_col( kh5.root.kalman_observations,'hz_line3')
            clear_col( kh5.root.kalman_observations,'hz_line4')
            clear_col( kh5.root.kalman_observations,'hz_line5')
            print 'done clearing columns'

            fig1=plt.figure()
            ax1=fig1.add_subplot(511)
            ax2=fig1.add_subplot(512,sharex=ax1)
            ax3=fig1.add_subplot(513,sharex=ax1)
            ax4=fig1.add_subplot(514,sharex=ax1)
            ax5=fig1.add_subplot(515,sharex=ax1)
            ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

            reconst = reconstruct.Reconstructor(kh5)

            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
            fps = result_utils.get_fps(h5)
            dt = 1.0/fps

            used_camn_dict = {}

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
                all_xhats = []
                all_ori = []
                xhat_results[ obj_id ] = collections.defaultdict(dict)

                obj_3d_rows = ca.load_dynamics_free_MLE_position( obj_id, kh5)

                smoothed_3d_rows = ca.load_data(
                    obj_id, kh5,
                    frames_per_second=fps,
                    dynamic_model_name='mamarama, units: mm')
                smoothed_frame_qfi = result_utils.QuickFrameIndexer(
                    smoothed_3d_rows['frame'])

                slopes_by_camn_by_frame = collections.defaultdict(dict)
                pt_idx_by_camn_by_frame = collections.defaultdict(dict)
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
                        if ((row['eccentricity']<reconst.minimum_eccentricity)or
                            (row['area'] < area_threshold_for_orientation)):
                            slopes_by_camn_by_frame[camn][framenumber]=np.nan
                            pt_idx_by_camn_by_frame[camn][framenumber]=camn_pt_no
                        else:
                            slopes_by_camn_by_frame[camn][framenumber]=row['slope']
                            pt_idx_by_camn_by_frame[camn][framenumber]=camn_pt_no

                # now collect in a numpy array for all cam

                assert int(min_frame)==min_frame
                assert int(max_frame+1)==max_frame+1
                frame_range = np.arange(int(min_frame),int(max_frame+1))
                if debug_level >= 1:
                    print 'frame range %d-%d'%(frame_range[0],frame_range[-1])
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

                        ## if absolute_frame_number==329236:
                        ##     print '**camn',camn
                        slopes[frame_idx,j] = slopes_by_frame.get(
                            absolute_frame_number,np.nan)

                    ax1.plot(frame_range,slope2modpi(slopes[:,j]),'.',
                             label=camn2cam_id[camn])
                ## print 'found all points','*'*80

                ax1.legend()
                ## plt.savefig('fig_ori.png')

                ## # find all possible hypotheses on any given frame
                ## all_hypotheses = [h for h in reconstruct.setOfSubsets(
                ##     range(len(camn_list))) if len(h) >= 2]
                ## for i,h in enumerate(all_hypotheses):
                ##     print i,h




################################################################################
################################################################################
################################################################################
################################################################################

                # EKF method here





                if 1:
                    # guesstimate initial orientation (XXX not done)
                    up_vec = 0,0,1
                    q0 = PQmath.orientation_to_quat( up_vec )
                    w0 = 0,0,0 # no angular rate
                    init_x = np.array([w0[0],w0[1],w0[2],
                                          q0.x, q0.y, q0.z, q0.w])
                    print 'init_x',init_x

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

                    # angular rate part of state variance
                    for i in range(0,3):
                        Q[i,i] = Q_scalar_rate

                    # quaternion part of state variance
                    for i in range(3,7):
                        Q[i,i] = Q_scalar_quat

                preA = np.eye(7)

                ekf = kalman_ekf.EKF( init_x, Pminus )
                previous_posterior_x = init_x
                _save_plot_rows = []
                for frame_idx, absolute_frame_number in enumerate(frame_range):
                    # Evaluate the Jacobian of the process update
                    # using previous frame's posterior estimate. (This
                    # is not quite the same as this frame's prior
                    # estimate. The difference this frame's prior
                    # estimate is _after_ the process update
                    # model. Which we need to get doing this.)

                    _save_plot_rows.append( np.nan*np.ones( (n_cams,) ))

                    this_dx = eval_dAdt( previous_posterior_x )
                    A = preA + this_dx*dt
                    if debug_level >= 1:
                        print
                        print 'frame',absolute_frame_number,'-'*40
                        print 'previous posterior',previous_posterior_x
                        if debug_level > 6:
                            print 'A'
                            print A

                    xhatminus, Pminus=ekf.step1__calculate_a_priori(A,Q)
                    if debug_level >= 1:
                        print 'new prior',xhatminus

                    # 1. Gate per-camera orientations.

                    this_frame_slopes = slopes[frame_idx,:]
                    if debug_level >= 5:
                        print 'this_frame_slopes',this_frame_slopes

                    missing_data = False
                    gate_vector=None

                    y=[] # observation (per camera)
                    hx=[] # expected observation (per camera)
                    C=[] # linearized observation model (per camera)
                    N_obs_this_frame = 0
                    cams_without_data = np.isnan( this_frame_slopes )
                    if np.all(cams_without_data):
                        missing_data = True

                    smoothed_pos_idxs = smoothed_frame_qfi.get_frame_idxs(
                        absolute_frame_number)
                    assert len(smoothed_pos_idxs)==1
                    smoothed_pos_idx = smoothed_pos_idxs[0]
                    smooth_row = smoothed_3d_rows[smoothed_pos_idx]
                    assert smooth_row['frame'] == absolute_frame_number
                    center_position = np.array((smooth_row['x'],
                                                smooth_row['y'],
                                                smooth_row['z']))
                    if debug_level >= 2:
                        print 'center_position',center_position

                    if not missing_data:
                        if expected_orientation_method == 'trust_prior':
                            other_position = get_point_on_line(xhatminus,
                                                               center_position)
                            if debug_level >= 6:
                                print 'other_position',other_position
                        else:
                            # e.g. based on SVD fit of camera plances
                            raise NotImplementedError('')

                        cams_with_data = ~cams_without_data
                        possible_cam_idxs = np.nonzero(cams_with_data)[0]
                        if debug_level >= 6:
                            print 'possible_cam_idxs',possible_cam_idxs
                        gate_vector = np.zeros( (n_cams,), dtype=np.bool)
                        flip_vector = np.zeros( (n_cams,), dtype=np.bool)
                        for camn_idx in possible_cam_idxs:
                            cam_id = cam_id_list[camn_idx]
                            camn = camn_list[camn_idx]

                            # This ignores distortion. To incorporate
                            # distortion, this would require
                            # appropriate scaling of orientation
                            # vector, which would require knowing
                            # target's size. In which case we should
                            # track head and tail separately and not
                            # use this whole quaternion mess.

                            a = reconst.find2d( cam_id, center_position)
                            b = reconst.find2d( cam_id, other_position)

                            if debug_level >= 6:
                                print 'cam_id',cam_id
                            theta_expected=find_theta_mod_pi_between_points(a,b)
                            theta_measured=slope2modpi(
                                this_frame_slopes[camn_idx])
                            if debug_level >= 6:
                                print '  theta_expected,theta_measured',theta_expected*R2D,theta_measured*R2D
                            if reconstruct.angles_near(
                                theta_expected,theta_measured,
                                gate_angle_threshold_radians,
                                mod_pi=True):
                                gate_vector[camn_idx]=1
                                if debug_level >= 6:
                                    print '    good'

                                if not reconstruct.angles_near(
                                    theta_expected,theta_measured+np.pi,
                                    gate_angle_threshold_radians,
                                    mod_pi=False):
                                    flip_vector[camn_idx]=1
                                    if debug_level >= 6:
                                        print '      flipped'

                                y.append(theta_measured)

                                P = reconst.get_pmat( cam_id )
                                args =  (P[0,0], P[0,1], P[0,2], P[0,3],
                                         P[1,0], P[1,1], P[1,2], P[1,3],
                                         P[2,0], P[2,1], P[2,2], P[2,3],
                                         center_position[0],
                                         center_position[1],
                                         center_position[2],
                                         xhatminus)
                                hx.append(eval_G(*args))
                                _save_plot_rows[-1][camn_idx] = (y[-1] - hx[-1])

                                C.append(eval_linG(*args))
                                N_obs_this_frame += 1
                                # save which orientations were used
                                if absolute_frame_number not in used_camn_dict:
                                    used_camn_dict[absolute_frame_number]=[]

                                ## if absolute_frame_number>=329235:
                                ##     print 'cam_id',cam_id
                                ##     print 'camn',camn
                                ##     if absolute_frame_number>=329238:
                                ##         1/0
                                camn_pt_no = (
                                    pt_idx_by_camn_by_frame[camn][
                                    absolute_frame_number])
                                used_camn_dict[absolute_frame_number].append(
                                    (camn,camn_pt_no))
                            else:
                                if debug_level >= 6:
                                    print '    bad'

                        if debug_level >= 1:
                            print 'gate_vector',gate_vector
                            print 'flip_vector',flip_vector
                        missing_data = not bool(np.sum(gate_vector))

                    # 3. Construct observations model using all
                    # gated-in camera orientations.

                    if missing_data:
                        C = None
                        R = None
                        hx = None
                    else:
                        C = np.array(C)
                        R=R_scalar*np.eye(N_obs_this_frame)
                        hx = np.array(hx)
                        if debug_level >= 6:
                            print 'full values'
                            print 'C',C
                            print 'hx',hx
                            print 'R',R

                    if debug_level >= 1:
                        print 'missing_data',missing_data
                    xhat,P = ekf.step2__calculate_a_posteriori(xhatminus, Pminus, y=y, hx=hx,
                                                               C=C,R=R,
                                                               missing_data=missing_data)
                    if debug_level >= 1:
                        print 'xhat',xhat
                    previous_posterior_x = xhat
                    xhat_results[ obj_id ][absolute_frame_number ] = (xhat,center_position)
                    all_xhats.append(xhat)
                    all_ori.append( state_to_ori(xhat) )

                all_xhats = np.array( all_xhats )
                all_ori = np.array( all_ori )
                _save_plot_rows = np.array( _save_plot_rows )

                ax2.plot(frame_range,all_xhats[:,0],'.',label='p')
                ax2.plot(frame_range,all_xhats[:,1],'.',label='q')
                ax2.plot(frame_range,all_xhats[:,2],'.',label='r')
                ax2.legend()

                ax3.plot(frame_range,all_xhats[:,3],'.',label='a')
                ax3.plot(frame_range,all_xhats[:,4],'.',label='b')
                ax3.plot(frame_range,all_xhats[:,5],'.',label='c')
                ax3.plot(frame_range,all_xhats[:,6],'.',label='d')
                ax3.legend()

                ax4.plot(frame_range,all_ori[:,0],'.',label='x')
                ax4.plot(frame_range,all_ori[:,1],'.',label='y')
                ax4.plot(frame_range,all_ori[:,2],'.',label='z')
                ax4.legend()

                for i in range(n_cams):
                    ax5.plot(frame_range,_save_plot_rows[:,i], '.',
                             label=cam_id_list[i])
                ax5.legend()
        print 'saving results'
        result = xhat_results
        for obj_id in result.keys():
            table = kh5.root.kalman_observations
            rows = table[:]
            cond1 = (rows['obj_id']==obj_id)
            for absolute_frame_number in result[obj_id].keys():
                cond2 = rows['frame']==absolute_frame_number
                cond3 = cond1 & cond2
                idxs = np.nonzero(cond3)[0]
                if len(idxs):
                    xhat,pos = result[obj_id][absolute_frame_number]
                    hz = state_to_hzline(xhat,pos)
                    assert len(idxs)==1
                    idx = idxs[0]
                    for row in table.iterrows(start=idx,stop=(idx+1)):
                        assert row['obj_id']==obj_id
                        assert row['frame']==absolute_frame_number
                        for i in range(6):
                            row['hz_line%d'%i] = hz[i]
                        row.update()

    if debug_level >= 1:
        debug_fname = 'temp_results.pkl'
        print 'saving debug results to file',
        fd = open(debug_fname,mode='w')
        pickle.dump(used_camn_dict,fd)
        fd.close()
    plt.show()
