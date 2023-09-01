from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import

import flydra_analysis.analysis.result_utils as result_utils
import flydra_analysis.a2.core_analysis as core_analysis
import tables
import numpy as np
import flydra_core.reconstruct as reconstruct
import collections, time, sys, os
from optparse import OptionParser

from .tables_tools import clear_col, open_file_safe
import flydra_core.kalman.ekf as kalman_ekf
import flydra_analysis.analysis.PQmath as PQmath
import flydra_core.geom as geom
import cgtypes  # cgkit 1.x
import sympy
from sympy import Symbol, Matrix, sqrt, latex, lambdify
import pickle
import warnings
from flydra_core.kalman.ori_smooth import ori_smooth

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# configuration and constants (important stuff)
expected_orientation_method = "trust_prior"
# expected_orientation_method = 'SVD_line_fits'

Q_scalar_rate = 0.1
Q_scalar_quat = 0.1
R_scalar = 10

# everything else

slope2modpi = np.arctan  # assign function name


def angle_diff(a, b, mod_pi=False):
    """return difference between two angles in range [-pi,pi]"""
    if mod_pi:
        a = 2 * a
        b = 2 * b
    result = np.mod((a - b) + np.pi, 2 * np.pi) - np.pi
    if mod_pi:
        result = 0.5 * result
    return result


def test_angle_diff():
    for a in [-5, -1, 0, 1, 2, 5, 45, 89, 89.9, 90, 90.1, 179, 180, 181, 359]:
        ar = a * D2R
        assert abs(angle_diff(ar, ar + 2 * np.pi)) < 0.0001
        assert abs(angle_diff(ar, ar + np.pi, mod_pi=True)) < 0.0001
        assert abs(angle_diff(ar, ar + np.pi)) > (np.pi - 0.0001)

    for (a, b, expected) in [
        (82 * D2R, -37 * D2R, -61 * D2R),
    ]:
        actual = angle_diff(a, b, mod_pi=True)
        assert actual < (expected + 1e-5)
        assert actual > (expected - 1e-5)


def statespace2cgtypes_quat(x):
    return cgtypes.quat(x[6], x[3], x[4], x[5])


def cgtypes_quat2statespace(q, Rp=0, Rq=0, Rr=0):
    return (Rp, Rq, Rr, q.x, q.y, q.z, q.w)


def state_to_ori(x):
    q = statespace2cgtypes_quat(x)
    return PQmath.quat_to_orient(q)


def state_to_hzline(x, A):
    Ax, Ay, Az = A[0], A[1], A[2]
    Ux, Uy, Uz = state_to_ori(x)
    line = geom.line_from_points(
        geom.ThreeTuple((Ax, Ay, Az)), geom.ThreeTuple((Ax + Ux, Ay + Uy, Az + Uz))
    )
    return line.to_hz()


def get_point_on_line(x, A, mu=1.0):
    """get a point on a line through A specified by state space vector x
    """
    # x is our state space vector
    q = statespace2cgtypes_quat(x)
    return mu * np.asarray(PQmath.quat_to_orient(q)) + A


def find_theta_mod_pi_between_points(a, b):
    diff = a - b
    dx, dy = diff
    if dx == 0.0:
        return np.pi / 2
    return slope2modpi(dy / dx)


class drop_dims(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        mat = self.func(*args, **kwargs)
        arr2d = np.array(mat)
        assert len(arr2d.shape) == 2
        if arr2d.shape[0] == 1:
            arr1d = arr2d[0, :]
            return arr1d
        return arr2d


class SymobolicModels:
    def __init__(self):
        # camera matrix
        self.P00 = sympy.Symbol("P00")
        self.P01 = sympy.Symbol("P01")
        self.P02 = sympy.Symbol("P02")
        self.P03 = sympy.Symbol("P03")

        self.P10 = sympy.Symbol("P10")
        self.P11 = sympy.Symbol("P11")
        self.P12 = sympy.Symbol("P12")
        self.P13 = sympy.Symbol("P13")

        self.P20 = sympy.Symbol("P20")
        self.P21 = sympy.Symbol("P21")
        self.P22 = sympy.Symbol("P22")
        self.P23 = sympy.Symbol("P23")

        # center about point
        self.Ax = sympy.Symbol("Ax")
        self.Ay = sympy.Symbol("Ay")
        self.Az = sympy.Symbol("Az")

    def get_process_model(self, x):

        # This formulation partly from Marins, Yun, Bachmann, McGhee, and
        # Zyda (2001). An Extended Kalman Filter for Quaternion-Based
        # Orientation Estimation Using MARG Sensors. Proceedings of the
        # 2001 IEEE/RSJ International Conference on Intelligent Robots and
        # Systems.

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        if 0:
            tau_rx = Symbol("tau_rx")
            tau_ry = Symbol("tau_ry")
            tau_rz = Symbol("tau_rz")
        else:
            tau_rx = 0.1
            tau_ry = 0.1
            tau_rz = 0.1

        # eqns 9-15
        f1 = -1 / tau_rx * x1
        f2 = -1 / tau_ry * x2
        f3 = -1 / tau_rz * x3
        scale = 2 * sqrt(x4 ** 2 + x5 ** 2 + x6 ** 2 + x7 ** 2)
        f4 = 1 / scale * (x3 * x5 - x2 * x6 + x1 * x7)
        f5 = 1 / scale * (-x3 * x4 + x1 * x6 + x2 * x7)
        f6 = 1 / scale * (x2 * x4 - x1 * x5 + x3 * x7)
        f7 = 1 / scale * (-x1 * x4 - x2 * x5 + x3 * x6)

        derivative_x = (f1, f2, f3, f4, f5, f6, f7)
        derivative_x = Matrix(derivative_x).T

        dx_symbolic = derivative_x.jacobian((x1, x2, x3, x4, x5, x6, x7))
        return dx_symbolic

    def get_observation_model(self, x):

        # Make Nomenclature match with Marins, Yun, Bachmann, McGhee,
        # and Zyda (2001).

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        a = x4
        b = x5
        c = x6
        d = x7

        # rotation matrix (eqn 6)
        R = Matrix(
            [
                [
                    d ** 2 + a ** 2 - b ** 2 - c ** 2,
                    2 * (a * b - c * d),
                    2 * (a * c + b * d),
                ],
                [
                    2 * (a * b + c * d),
                    d ** 2 + b ** 2 - a ** 2 - c ** 2,
                    2 * (b * c - a * d),
                ],
                [
                    2 * (a * c - b * d),
                    2 * (b * c + a * d),
                    d ** 2 + c ** 2 - b ** 2 - a ** 2,
                ],
            ]
        )
        # default orientation with no rotation
        u = Matrix([[1], [0], [0]])

        # rotated orientation
        U = R * u

        # point in space
        A = sympy.Matrix([[self.Ax], [self.Ay], [self.Az]])  # make vector
        hA = sympy.Matrix([[self.Ax], [self.Ay], [self.Az], [1]])  # homogenous

        # second point in space, the two of which define line
        B = A + U
        hB = sympy.Matrix([[B[0]], [B[1]], [B[2]], [1]])  # homogenous

        P = sympy.Matrix(
            [
                [self.P00, self.P01, self.P02, self.P03],
                [self.P10, self.P11, self.P12, self.P13],
                [self.P20, self.P21, self.P22, self.P23],
            ]
        )

        # the image projection of points on line
        ha = P * hA
        hb = P * hB

        # de homogenize
        a2 = sympy.Matrix([[ha[0] / ha[2]], [ha[1] / ha[2]]])
        b2 = sympy.Matrix([[hb[0] / hb[2]], [hb[1] / hb[2]]])

        # direction in image
        vec = b2 - a2

        # rise and run
        dy = vec[1]
        dx = vec[0]

        # prefer atan over atan2 because observations are mod pi.
        theta = sympy.atan(dy / dx)
        return theta


def doit(
    output_h5_filename=None,
    kalman_filename=None,
    data2d_filename=None,
    start=None,
    stop=None,
    gate_angle_threshold_degrees=40.0,
    area_threshold_for_orientation=0.0,
    obj_only=None,
    options=None,
):
    gate_angle_threshold_radians = gate_angle_threshold_degrees * D2R

    if options.show:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

    M = SymobolicModels()
    x = sympy.DeferredVector("x")
    G_symbolic = M.get_observation_model(x)
    dx_symbolic = M.get_process_model(x)

    if 0:
        print("G_symbolic")
        sympy.pprint(G_symbolic)
        print()

    G_linearized = [G_symbolic.diff(x[i]) for i in range(7)]
    if 0:
        print("G_linearized")
        for i in range(len(G_linearized)):
            sympy.pprint(G_linearized[i])
        print()

    arg_tuple_x = (
        M.P00,
        M.P01,
        M.P02,
        M.P03,
        M.P10,
        M.P11,
        M.P12,
        M.P13,
        M.P20,
        M.P21,
        M.P22,
        M.P23,
        M.Ax,
        M.Ay,
        M.Az,
        x,
    )

    xm = sympy.DeferredVector("xm")
    arg_tuple_x_xm = (
        M.P00,
        M.P01,
        M.P02,
        M.P03,
        M.P10,
        M.P11,
        M.P12,
        M.P13,
        M.P20,
        M.P21,
        M.P22,
        M.P23,
        M.Ax,
        M.Ay,
        M.Az,
        x,
        xm,
    )

    eval_G = lambdify(arg_tuple_x, G_symbolic, "numpy")
    eval_linG = lambdify(arg_tuple_x, G_linearized, "numpy")

    # coord shift of observation model
    phi_symbolic = M.get_observation_model(xm)

    # H = G - phi
    H_symbolic = G_symbolic - phi_symbolic

    # We still take derivative wrt x (not xm).
    H_linearized = [H_symbolic.diff(x[i]) for i in range(7)]

    eval_phi = lambdify(arg_tuple_x_xm, phi_symbolic, "numpy")
    eval_H = lambdify(arg_tuple_x_xm, H_symbolic, "numpy")
    eval_linH = lambdify(arg_tuple_x_xm, H_linearized, "numpy")

    if 0:
        print("dx_symbolic")
        sympy.pprint(dx_symbolic)
        print()

    eval_dAdt = drop_dims(lambdify(x, dx_symbolic, "numpy"))

    debug_level = 0
    if debug_level:
        np.set_printoptions(linewidth=130, suppress=True)

    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)

    ca = core_analysis.get_global_CachingAnalyzer()
    with open_file_safe(output_h5_filename, mode="w") as output_h5:

        with open_file_safe(kalman_filename, mode="r") as kh5:
            with open_file_safe(data2d_filename, mode="r") as h5:
                for input_node in kh5.root._f_iter_nodes():
                    # copy everything from source to dest
                    input_node._f_copy(output_h5.root, recursive=True)

                try:
                    dest_table = output_h5.root.ML_estimates
                except tables.exceptions.NoSuchNodeError as err1:
                    # backwards compatibility
                    try:
                        dest_table = output_h5.root.kalman_observations
                    except tables.exceptions.NoSuchNodeError as err2:
                        raise err1
                for colname in ["hz_line%d" % i for i in range(6)]:
                    clear_col(dest_table, colname)
                dest_table.flush()

                if options.show:
                    fig1 = plt.figure()
                    ax1 = fig1.add_subplot(511)
                    ax2 = fig1.add_subplot(512, sharex=ax1)
                    ax3 = fig1.add_subplot(513, sharex=ax1)
                    ax4 = fig1.add_subplot(514, sharex=ax1)
                    ax5 = fig1.add_subplot(515, sharex=ax1)
                    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

                    min_frame_range = np.inf
                    max_frame_range = -np.inf

                reconst = reconstruct.Reconstructor(kh5)

                camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
                fps = result_utils.get_fps(h5)
                dt = 1.0 / fps

                used_camn_dict = {}

                # associate framenumbers with timestamps using 2d .h5 file
                data2d = h5.root.data2d_distorted[:]  # load to RAM
                if start is not None:
                    data2d = data2d[data2d["frame"] >= start]
                if stop is not None:
                    data2d = data2d[data2d["frame"] <= stop]
                data2d_idxs = np.arange(len(data2d))
                h5_framenumbers = data2d["frame"]
                h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

                ML_estimates_2d_idxs = kh5.root.ML_estimates_2d_idxs[:]

                all_kobs_obj_ids = dest_table.read(field="obj_id")
                all_kobs_frames = dest_table.read(field="frame")
                use_obj_ids = np.unique(all_kobs_obj_ids)
                if obj_only is not None:
                    use_obj_ids = obj_only

                if hasattr(kh5.root.kalman_estimates.attrs, "dynamic_model_name"):
                    dynamic_model = kh5.root.kalman_estimates.attrs.dynamic_model_name
                    if dynamic_model.startswith("EKF "):
                        dynamic_model = dynamic_model[4:]
                else:
                    dynamic_model = "mamarama, units: mm"
                    warnings.warn(
                        'could not determine dynamic model name, using "%s"'
                        % dynamic_model
                    )

                for obj_id_enum, obj_id in enumerate(use_obj_ids):
                    # Use data association step from kalmanization to load potentially
                    # relevant 2D orientations, but discard previous 3D orientation.
                    if obj_id_enum % 100 == 0:
                        print(
                            "obj_id %d (%d of %d)"
                            % (obj_id, obj_id_enum, len(use_obj_ids))
                        )
                    if options.show:
                        all_xhats = []
                        all_ori = []

                    output_row_obj_id_cond = all_kobs_obj_ids == obj_id

                    obj_3d_rows = ca.load_dynamics_free_MLE_position(obj_id, kh5)
                    if start is not None:
                        obj_3d_rows = obj_3d_rows[obj_3d_rows["frame"] >= start]
                    if stop is not None:
                        obj_3d_rows = obj_3d_rows[obj_3d_rows["frame"] <= stop]

                    try:
                        smoothed_3d_rows = ca.load_data(
                            obj_id,
                            kh5,
                            use_kalman_smoothing=True,
                            frames_per_second=fps,
                            dynamic_model_name=dynamic_model,
                        )
                    except core_analysis.NotEnoughDataToSmoothError:
                        continue

                    smoothed_frame_qfi = result_utils.QuickFrameIndexer(
                        smoothed_3d_rows["frame"]
                    )

                    slopes_by_camn_by_frame = collections.defaultdict(dict)
                    x0d_by_camn_by_frame = collections.defaultdict(dict)
                    y0d_by_camn_by_frame = collections.defaultdict(dict)
                    pt_idx_by_camn_by_frame = collections.defaultdict(dict)
                    min_frame = np.inf
                    max_frame = -np.inf

                    start_idx = None
                    for this_idx, this_3d_row in enumerate(obj_3d_rows):
                        # iterate over each sample in the current camera
                        framenumber = this_3d_row["frame"]

                        if not np.isnan(this_3d_row["hz_line0"]):
                            # We have a valid initial 3d orientation guess.
                            if framenumber < min_frame:
                                min_frame = framenumber
                                assert start_idx is None, "frames out of order?"
                                start_idx = this_idx

                        max_frame = max(max_frame, framenumber)
                        h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)

                        frame2d = data2d[h5_2d_row_idxs]
                        frame2d_idxs = data2d_idxs[h5_2d_row_idxs]

                        obs_2d_idx = this_3d_row["obs_2d_idx"]
                        kobs_2d_data = ML_estimates_2d_idxs[int(obs_2d_idx)]

                        # Parse VLArray.
                        this_camns = kobs_2d_data[0::2]
                        this_camn_idxs = kobs_2d_data[1::2]

                        # Now, for each camera viewing this object at this
                        # frame, extract images.
                        for camn, camn_pt_no in zip(this_camns, this_camn_idxs):
                            # find 2D point corresponding to object
                            cam_id = camn2cam_id[camn]

                            cond = (frame2d["camn"] == camn) & (
                                frame2d["frame_pt_idx"] == camn_pt_no
                            )
                            idxs = np.nonzero(cond)[0]
                            if len(idxs) == 0:
                                continue
                            assert len(idxs) == 1
                            ## if len(idxs)!=1:
                            ##     raise ValueError('expected one (and only one) frame, got %d'%len(idxs))
                            idx = idxs[0]

                            orig_data2d_rownum = frame2d_idxs[idx]
                            frame_timestamp = frame2d[idx]["timestamp"]

                            row = frame2d[idx]
                            assert framenumber == row["frame"]
                            if (row["eccentricity"] < reconst.minimum_eccentricity) or (
                                row["area"] < area_threshold_for_orientation
                            ):
                                slopes_by_camn_by_frame[camn][framenumber] = np.nan
                                x0d_by_camn_by_frame[camn][framenumber] = np.nan
                                y0d_by_camn_by_frame[camn][framenumber] = np.nan
                                pt_idx_by_camn_by_frame[camn][framenumber] = camn_pt_no
                            else:
                                slopes_by_camn_by_frame[camn][framenumber] = row[
                                    "slope"
                                ]
                                x0d_by_camn_by_frame[camn][framenumber] = row["x"]
                                y0d_by_camn_by_frame[camn][framenumber] = row["y"]
                                pt_idx_by_camn_by_frame[camn][framenumber] = camn_pt_no

                    if start_idx is None:
                        warnings.warn(
                            "skipping obj_id %d: "
                            "could not find valid start frame" % obj_id
                        )
                        continue

                    obj_3d_rows = obj_3d_rows[start_idx:]

                    # now collect in a numpy array for all cam

                    assert int(min_frame) == min_frame
                    assert int(max_frame + 1) == max_frame + 1
                    frame_range = np.arange(int(min_frame), int(max_frame + 1))
                    if debug_level >= 1:
                        print("frame range %d-%d" % (frame_range[0], frame_range[-1]))
                    camn_list = slopes_by_camn_by_frame.keys()
                    camn_list.sort()
                    cam_id_list = [camn2cam_id[camn] for camn in camn_list]
                    n_cams = len(camn_list)
                    n_frames = len(frame_range)

                    save_cols = {}
                    save_cols["frame"] = []
                    for camn in camn_list:
                        save_cols["dist%d" % camn] = []
                        save_cols["used%d" % camn] = []
                        save_cols["theta%d" % camn] = []

                    # NxM array with rows being frames and cols being cameras
                    slopes = np.ones((n_frames, n_cams), dtype=np.float64)
                    x0ds = np.ones((n_frames, n_cams), dtype=np.float64)
                    y0ds = np.ones((n_frames, n_cams), dtype=np.float64)
                    for j, camn in enumerate(camn_list):

                        slopes_by_frame = slopes_by_camn_by_frame[camn]
                        x0d_by_frame = x0d_by_camn_by_frame[camn]
                        y0d_by_frame = y0d_by_camn_by_frame[camn]

                        for frame_idx, absolute_frame_number in enumerate(frame_range):

                            slopes[frame_idx, j] = slopes_by_frame.get(
                                absolute_frame_number, np.nan
                            )
                            x0ds[frame_idx, j] = x0d_by_frame.get(
                                absolute_frame_number, np.nan
                            )
                            y0ds[frame_idx, j] = y0d_by_frame.get(
                                absolute_frame_number, np.nan
                            )

                        if options.show:
                            frf = np.array(frame_range, dtype=np.float64)
                            min_frame_range = min(np.min(frf), min_frame_range)
                            max_frame_range = max(np.max(frf), max_frame_range)

                            ax1.plot(
                                frame_range,
                                slope2modpi(slopes[:, j]),
                                ".",
                                label=camn2cam_id[camn],
                            )

                    if options.show:
                        ax1.legend()

                    if 1:
                        # estimate orientation of initial frame
                        row0 = obj_3d_rows[
                            :1
                        ]  # take only first row but keep as 1d array
                        hzlines = np.array(
                            [
                                row0["hz_line0"],
                                row0["hz_line1"],
                                row0["hz_line2"],
                                row0["hz_line3"],
                                row0["hz_line4"],
                                row0["hz_line5"],
                            ]
                        ).T
                        directions = reconstruct.line_direction(hzlines)
                        q0 = PQmath.orientation_to_quat(directions[0])
                        assert not np.isnan(
                            q0.x
                        ), "cannot start with missing orientation"
                        w0 = 0, 0, 0  # no angular rate
                        init_x = np.array([w0[0], w0[1], w0[2], q0.x, q0.y, q0.z, q0.w])

                        Pminus = np.zeros((7, 7))

                        # angular rate part of state variance is .5
                        for i in range(0, 3):
                            Pminus[i, i] = 0.5

                        # quaternion part of state variance is 1
                        for i in range(3, 7):
                            Pminus[i, i] = 1

                    if 1:
                        # setup of noise estimates
                        Q = np.zeros((7, 7))

                        # angular rate part of state variance
                        for i in range(0, 3):
                            Q[i, i] = Q_scalar_rate

                        # quaternion part of state variance
                        for i in range(3, 7):
                            Q[i, i] = Q_scalar_quat

                    preA = np.eye(7)

                    ekf = kalman_ekf.EKF(init_x, Pminus)
                    previous_posterior_x = init_x
                    if options.show:
                        _save_plot_rows = []
                        _save_plot_rows_used = []
                    for frame_idx, absolute_frame_number in enumerate(frame_range):
                        # Evaluate the Jacobian of the process update
                        # using previous frame's posterior estimate. (This
                        # is not quite the same as this frame's prior
                        # estimate. The difference this frame's prior
                        # estimate is _after_ the process update
                        # model. Which we need to get doing this.)

                        if options.show:
                            _save_plot_rows.append(np.nan * np.ones((n_cams,)))
                            _save_plot_rows_used.append(np.nan * np.ones((n_cams,)))

                        this_dx = eval_dAdt(previous_posterior_x)
                        A = preA + this_dx * dt
                        if debug_level >= 1:
                            print()
                            print("frame", absolute_frame_number, "-" * 40)
                            print("previous posterior", previous_posterior_x)
                            if debug_level > 6:
                                print("A")
                                print(A)

                        xhatminus, Pminus = ekf.step1__calculate_a_priori(A, Q)
                        if debug_level >= 1:
                            print("new prior", xhatminus)

                        # 1. Gate per-camera orientations.

                        this_frame_slopes = slopes[frame_idx, :]
                        this_frame_theta_measured = slope2modpi(this_frame_slopes)
                        this_frame_x0d = x0ds[frame_idx, :]
                        this_frame_y0d = y0ds[frame_idx, :]
                        if debug_level >= 5:
                            print("this_frame_slopes", this_frame_slopes)

                        save_cols["frame"].append(absolute_frame_number)
                        for j, camn in enumerate(camn_list):
                            # default to no detection, change below
                            save_cols["dist%d" % camn].append(np.nan)
                            save_cols["used%d" % camn].append(0)
                            save_cols["theta%d" % camn].append(
                                this_frame_theta_measured[j]
                            )

                        all_data_this_frame_missing = False
                        gate_vector = None

                        y = []  # observation (per camera)
                        hx = []  # expected observation (per camera)
                        C = []  # linearized observation model (per camera)
                        N_obs_this_frame = 0
                        cams_without_data = np.isnan(this_frame_slopes)
                        if np.all(cams_without_data):
                            all_data_this_frame_missing = True

                        smoothed_pos_idxs = smoothed_frame_qfi.get_frame_idxs(
                            absolute_frame_number
                        )
                        if len(smoothed_pos_idxs) == 0:
                            all_data_this_frame_missing = True
                            smoothed_pos_idx = None
                            smooth_row = None
                            center_position = None
                        else:
                            try:
                                assert len(smoothed_pos_idxs) == 1
                            except:
                                print("obj_id", obj_id)
                                print("absolute_frame_number", absolute_frame_number)
                                if len(frame_range):
                                    print(
                                        "frame_range[0],frame_rang[-1]",
                                        frame_range[0],
                                        frame_range[-1],
                                    )
                                else:
                                    print("no frame range")
                                print("len(smoothed_pos_idxs)", len(smoothed_pos_idxs))
                                raise
                            smoothed_pos_idx = smoothed_pos_idxs[0]
                            smooth_row = smoothed_3d_rows[smoothed_pos_idx]
                            assert smooth_row["frame"] == absolute_frame_number
                            center_position = np.array(
                                (smooth_row["x"], smooth_row["y"], smooth_row["z"])
                            )
                            if debug_level >= 2:
                                print("center_position", center_position)

                        if not all_data_this_frame_missing:
                            if expected_orientation_method == "trust_prior":
                                state_for_phi = xhatminus  # use a priori
                            elif expected_orientation_method == "SVD_line_fits":
                                # construct matrix of planes
                                P = []
                                for camn_idx in range(n_cams):
                                    this_x0d = this_frame_x0d[camn_idx]
                                    this_y0d = this_frame_y0d[camn_idx]
                                    slope = this_frame_slopes[camn_idx]
                                    plane, ray = reconst.get_3D_plane_and_ray(
                                        cam_id, this_x0d, this_y0d, slope
                                    )
                                    if np.isnan(plane[0]):
                                        continue
                                    P.append(plane)
                                if len(P) < 2:
                                    # not enough data to do SVD... fallback to prior
                                    state_for_phi = xhatminus  # use a priori
                                else:
                                    Lco = reconstruct.intersect_planes_to_find_line(P)
                                    q = PQmath.pluecker_to_quat(Lco)
                                    state_for_phi = cgtypes_quat2statespace(q)

                            cams_with_data = ~cams_without_data
                            possible_cam_idxs = np.nonzero(cams_with_data)[0]
                            if debug_level >= 6:
                                print("possible_cam_idxs", possible_cam_idxs)
                            gate_vector = np.zeros((n_cams,), dtype=np.bool_)
                            ## flip_vector = np.zeros( (n_cams,), dtype=np.bool_)
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

                                ## theta_measured=slope2modpi(
                                ##     this_frame_slopes[camn_idx])
                                theta_measured = this_frame_theta_measured[camn_idx]
                                if debug_level >= 6:
                                    print("cam_id %s, camn %d" % (cam_id, camn))
                                if debug_level >= 3:
                                    a = reconst.find2d(cam_id, center_position)
                                    other_position = get_point_on_line(
                                        xhatminus, center_position
                                    )
                                    b = reconst.find2d(cam_id, other_position)
                                    theta_expected = find_theta_mod_pi_between_points(
                                        a, b
                                    )
                                    print(
                                        (
                                            "  theta_expected,theta_measured",
                                            theta_expected * R2D,
                                            theta_measured * R2D,
                                        )
                                    )

                                P = reconst.get_pmat(cam_id)
                                if 0:
                                    args_x = (
                                        P[0, 0],
                                        P[0, 1],
                                        P[0, 2],
                                        P[0, 3],
                                        P[1, 0],
                                        P[1, 1],
                                        P[1, 2],
                                        P[1, 3],
                                        P[2, 0],
                                        P[2, 1],
                                        P[2, 2],
                                        P[2, 3],
                                        center_position[0],
                                        center_position[1],
                                        center_position[2],
                                        xhatminus,
                                    )
                                    this_y = theta_measured
                                    this_hx = eval_G(*args_x)
                                    this_C = eval_linG(*args_x)
                                else:
                                    args_x_xm = (
                                        P[0, 0],
                                        P[0, 1],
                                        P[0, 2],
                                        P[0, 3],
                                        P[1, 0],
                                        P[1, 1],
                                        P[1, 2],
                                        P[1, 3],
                                        P[2, 0],
                                        P[2, 1],
                                        P[2, 2],
                                        P[2, 3],
                                        center_position[0],
                                        center_position[1],
                                        center_position[2],
                                        xhatminus,
                                        state_for_phi,
                                    )
                                    this_phi = eval_phi(*args_x_xm)
                                    this_y = angle_diff(
                                        theta_measured, this_phi, mod_pi=True
                                    )
                                    this_hx = eval_H(*args_x_xm)
                                    this_C = eval_linH(*args_x_xm)
                                    if debug_level >= 3:
                                        print(
                                            (
                                                "  this_phi,this_y",
                                                this_phi * R2D,
                                                this_y * R2D,
                                            )
                                        )

                                save_cols["dist%d" % camn][-1] = this_y  # save

                                # gate
                                if abs(this_y) < gate_angle_threshold_radians:
                                    save_cols["used%d" % camn][-1] = 1
                                    gate_vector[camn_idx] = 1
                                    if debug_level >= 3:
                                        print("    good")
                                    if options.show:
                                        _save_plot_rows_used[-1][camn_idx] = this_y
                                    y.append(this_y)
                                    hx.append(this_hx)
                                    C.append(this_C)
                                    N_obs_this_frame += 1

                                    # Save which camn and camn_pt_no was used.
                                    if absolute_frame_number not in used_camn_dict:
                                        used_camn_dict[absolute_frame_number] = []
                                    camn_pt_no = pt_idx_by_camn_by_frame[camn][
                                        absolute_frame_number
                                    ]
                                    used_camn_dict[absolute_frame_number].append(
                                        (camn, camn_pt_no)
                                    )
                                else:
                                    if options.show:
                                        _save_plot_rows[-1][camn_idx] = this_y
                                    if debug_level >= 6:
                                        print("    bad")
                            if debug_level >= 1:
                                print("gate_vector", gate_vector)
                                # print 'flip_vector',flip_vector
                            all_data_this_frame_missing = not bool(np.sum(gate_vector))

                        # 3. Construct observations model using all
                        # gated-in camera orientations.

                        if all_data_this_frame_missing:
                            C = None
                            R = None
                            hx = None
                        else:
                            C = np.array(C)
                            R = R_scalar * np.eye(N_obs_this_frame)
                            hx = np.array(hx)
                            if 0:
                                # crazy observation error scaling
                                for i in range(N_obs_this_frame):
                                    beyond = abs(y[i]) - 10 * D2R
                                    beyond = max(0, beyond)  # clip at zero
                                    R[i:i] = R_scalar * (1 + 10 * beyond)
                            if debug_level >= 6:
                                print("full values")
                                print("C", C)
                                print("hx", hx)
                                print("y", y)
                                print("R", R)

                        if debug_level >= 1:
                            print(
                                "all_data_this_frame_missing",
                                all_data_this_frame_missing,
                            )
                        xhat, P = ekf.step2__calculate_a_posteriori(
                            xhatminus,
                            Pminus,
                            y=y,
                            hx=hx,
                            C=C,
                            R=R,
                            missing_data=all_data_this_frame_missing,
                        )
                        if debug_level >= 1:
                            print("xhat", xhat)
                        previous_posterior_x = xhat
                        if center_position is not None:
                            # save
                            output_row_frame_cond = (
                                all_kobs_frames == absolute_frame_number
                            )
                            output_row_cond = (
                                output_row_frame_cond & output_row_obj_id_cond
                            )
                            output_idxs = np.nonzero(output_row_cond)[0]
                            if len(output_idxs) == 0:
                                pass
                            else:
                                assert len(output_idxs) == 1
                                idx = output_idxs[0]
                                hz = state_to_hzline(xhat, center_position)
                                for row in dest_table.iterrows(
                                    start=idx, stop=(idx + 1)
                                ):
                                    for i in range(6):
                                        row["hz_line%d" % i] = hz[i]
                                    row.update()
                        ## xhat_results[ obj_id ][absolute_frame_number ] = (
                        ##     xhat,center_position)
                        if options.show:
                            all_xhats.append(xhat)
                            all_ori.append(state_to_ori(xhat))

                    # save to H5 file
                    names = [colname for colname in save_cols]
                    names.sort()
                    arrays = []
                    for name in names:
                        if name == "frame":
                            dtype = np.int64
                        elif name.startswith("dist"):
                            dtype = np.float32
                        elif name.startswith("used"):
                            dtype = np.bool
                        elif name.startswith("theta"):
                            dtype = np.float32
                        else:
                            raise NameError("unknown name %s" % name)
                        arr = np.array(save_cols[name], dtype=dtype)
                        arrays.append(arr)
                    save_recarray = np.rec.fromarrays(arrays, names=names)
                    h5group = core_analysis.get_group_for_obj(
                        obj_id, output_h5, writeable=True
                    )
                    output_h5.create_table(
                        h5group,
                        "obj%d" % obj_id,
                        save_recarray,
                        filters=tables.Filters(1, complib="zlib"),
                    )

                    if options.show:
                        all_xhats = np.array(all_xhats)
                        all_ori = np.array(all_ori)
                        _save_plot_rows = np.array(_save_plot_rows)
                        _save_plot_rows_used = np.array(_save_plot_rows_used)

                        ax2.plot(frame_range, all_xhats[:, 0], ".", label="p")
                        ax2.plot(frame_range, all_xhats[:, 1], ".", label="q")
                        ax2.plot(frame_range, all_xhats[:, 2], ".", label="r")
                        ax2.legend()

                        ax3.plot(frame_range, all_xhats[:, 3], ".", label="a")
                        ax3.plot(frame_range, all_xhats[:, 4], ".", label="b")
                        ax3.plot(frame_range, all_xhats[:, 5], ".", label="c")
                        ax3.plot(frame_range, all_xhats[:, 6], ".", label="d")
                        ax3.legend()

                        ax4.plot(frame_range, all_ori[:, 0], ".", label="x")
                        ax4.plot(frame_range, all_ori[:, 1], ".", label="y")
                        ax4.plot(frame_range, all_ori[:, 2], ".", label="z")
                        ax4.legend()

                        colors = []
                        for i in range(n_cams):
                            (line,) = ax5.plot(
                                frame_range,
                                _save_plot_rows_used[:, i] * R2D,
                                "o",
                                label=cam_id_list[i],
                            )
                            colors.append(line.get_color())
                        for i in range(n_cams):
                            # loop again to get normal MPL color cycling
                            ax5.plot(
                                frame_range,
                                _save_plot_rows[:, i] * R2D,
                                "o",
                                mec=colors[i],
                                ms=1.0,
                            )
                        ax5.set_ylabel("observation (deg)")
                        ax5.legend()

        # record that we did this...
        output_h5.root.ML_estimates.attrs.ori_ekf_time = time.time()

    if 0:
        debug_fname = "temp_results.pkl"
        print("saving debug results to file", end=" ")
        fd = open(debug_fname, mode="w")
        pickle.dump(used_camn_dict, fd)
        fd.close()

    if options.show:
        ax1.set_xlim(min_frame_range, max_frame_range)
        plt.show()


def is_orientation_fit(filename):
    with open_file_safe(filename, mode="r") as h5:
        if hasattr(h5.root.ML_estimates.attrs, "ori_ekf_time"):
            ori_ekf_time = h5.root.ML_estimates.attrs.ori_ekf_time
            return True
        else:
            return False


def is_orientation_fit_sysexit():
    usage = "%prog FILE"

    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    filename = args[0]
    if is_orientation_fit(filename):
        sys.exit(0)
    else:
        sys.exit(1)


def smooth(x, window_len=10, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    # copied from http://www.scipy.org/Cookbook/SignalSmooth

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


def compute_ori_quality(h5_context, orig_frames, obj_id, smooth_len=10):
    """compute quality of orientation estimate
    """
    ca = core_analysis.get_global_CachingAnalyzer()
    group = h5_context.get_or_make_group_for_obj(obj_id)
    try:
        table = getattr(group, "obj%d" % obj_id)
    except:
        sys.stderr.write(
            "ERROR while getting EKF fit data for obj_id %d in file opening %s\n"
            % (obj_id, h5_context.filename)
        )
        sys.stderr.write(
            "Hint: re-run orientation fitting for this file (for this obj_id).\n"
        )
        raise
    table_ram = table[:]
    frames = table_ram["frame"]

    camns = []
    for colname in table.colnames:
        if colname.startswith("dist"):
            camn = int(colname[4:])
            camns.append(camn)
    camns.sort()
    ncams = len(camns)

    # start at zero quality
    results = np.zeros((len(orig_frames),))
    for origi, frame in enumerate(orig_frames):
        cond = frames == frame
        idxs = np.nonzero(cond)[0]
        if len(idxs) == 0:
            results[origi] = np.nan
            continue

        assert len(idxs) == 1
        idx = idxs[0]
        this_row = table_ram[idx]
        used_this_row = np.array([this_row["used%d" % camn] for camn in camns])
        n_used = np.sum(used_this_row)
        if 1:
            results[origi] = n_used
        else:
            theta_this_row = np.array([this_row["theta%d" % camn] for camn in camns])
            data_this_row = ~np.isnan(theta_this_row)
            n_data = np.sum(data_this_row)
            n_rejected = n_data - n_used
            if n_rejected == 0:
                if n_used == 0:
                    results[origi] = 0.0
                else:
                    results[origi] = ncams
            else:
                results[origi] = n_used / n_rejected
    if smooth_len:
        if len(results) > smooth_len:
            results = smooth(results, window_len=smooth_len)
    return results


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "--h5", type="string", help=".h5 file with data2d_distorted (REQUIRED)"
    )

    parser.add_option(
        "-k",
        "--kalman-file",
        dest="kalman_filename",
        type="string",
        help=".h5 file with kalman data and 3D reconstructor",
    )

    parser.add_option(
        "--output-h5",
        type="string",
        help="filename for output .h5 file with data2d_distorted",
    )

    parser.add_option(
        "--gate-angle-threshold-degrees",
        type="float",
        default=40.0,
        help="maximum angle (in degrees) to include 2D orientation",
    )

    parser.add_option(
        "--area-threshold-for-orientation",
        type="float",
        default=0.0,
        help="minimum area required to use 2D feature for 3D orientation",
    )

    parser.add_option("--show", action="store_true", default=False)

    parser.add_option(
        "--start", type="int", default=None, help="frame number to begin analysis on"
    )

    parser.add_option(
        "--stop", type="int", default=None, help="frame number to end analysis on"
    )

    parser.add_option("--obj-only", type="string")

    (options, args) = parser.parse_args()

    if options.h5 is None:
        raise ValueError("--h5 option must be specified")

    if options.output_h5 is None:
        raise ValueError("--output-h5 option must be specified")

    if options.kalman_filename is None:
        raise ValueError("--kalman-file option must be specified")

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    doit(
        kalman_filename=options.kalman_filename,
        data2d_filename=options.h5,
        area_threshold_for_orientation=options.area_threshold_for_orientation,
        gate_angle_threshold_degrees=options.gate_angle_threshold_degrees,
        start=options.start,
        stop=options.stop,
        output_h5_filename=options.output_h5,
        obj_only=options.obj_only,
        options=options,
    )


if __name__ == "__main__":
    main()
