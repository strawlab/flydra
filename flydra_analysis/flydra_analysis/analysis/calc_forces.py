from __future__ import print_function
from __future__ import absolute_import
from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from numpy import nan
import math, time

import numpy as nx
import numpy as np

import flydra_core.reconstruct as reconstruct
import cgtypes  # cgkit 1.x
import tables  # pytables
import scipy.signal
import scipy.io

from .PQmath import *

### restore builtin functions which may have been overridden
##min = __builtins__.min
##max = __builtins__.max
##sum = __builtins__.sum
##round = __builtins__.round
##abs = __builtins__.abs


def getnan(x):
    return numpy.nonzero(numpy.isnan(x))


def my_interp(A, B, frac):
    return frac * (B - A) + A


def interpolate_P(results, start_frame, stop_frame, typ="best"):
    if typ == "fast":
        data3d = results.root.data3d_fast
    elif typ == "best":
        data3d = results.root.data3d_best
    fXl = [
        (
            row["frame"],
            row["x"],
            row["y"],
            row["z"],
            row["p0"],
            row["p1"],
            row["p2"],
            row["p3"],
            row["p4"],
            row["p5"],
        )
        for row in data3d
        if start_frame <= row["frame"] <= stop_frame
    ]  # XXX
    #           data3d.where( start_frame <= data3d.cols.frame <= stop_frame )]
    assert len(fXl) == 2
    assert stop_frame > start_frame
    assert (stop_frame - start_frame) > 1

    fXl = nx.array(fXl)
    frame = fXl[:, 0].astype(nx.int64)
    P = fXl[:, 1:4]

    print("  ", start_frame, P[0, :])

    dPdt = (P[1, :] - P[0, :]) / float(frame[1] - frame[0])
    for frame_no in range(start_frame + 1, stop_frame):
        frac = float(frame_no - start_frame) / float(stop_frame - start_frame)
        newP = P[0, :] + dPdt * frac

        print("  ", frame_no, newP, "<- new value")

        # now save to disk
        old_nrow = None
        #        for row in data3d.where( data3d.cols.frame == frame_no ):
        for row in data3d:
            if row["frame"] != frame_no:  # XXX
                continue
            if old_nrow is not None:
                raise RuntimeError(
                    "more than row with frame number %d in data3d" % frame_no
                )
            old_nrow = row.nrow()

        # delete old row
        if old_nrow is not None:
            data3d.remove_rows(start=old_nrow, stop=None)

        X = newP
        line3d = [nan] * 6  # fill with nans
        cam_nos_used_str = ""
        new_row = data3d.row
        new_row["frame"] = frame_no
        new_row["x"] = X[0]
        new_row["y"] = X[1]
        new_row["z"] = X[2]
        new_row["p0"] = line3d[0]
        new_row["p1"] = line3d[1]
        new_row["p2"] = line3d[2]
        new_row["p3"] = line3d[3]
        new_row["p4"] = line3d[4]
        new_row["p5"] = line3d[5]
        new_row["timestamp"] = 0.0
        new_row["camns_used"] = cam_nos_used_str
        new_row["mean_dist"] = 0.0
        new_row.append()
        data3d.flush()

    print("  ", stop_frame, P[1, :])


def sort_on_col0(a, b):
    a0 = a[0]
    b0 = b[0]
    if a0 < b0:
        return -1
    elif a0 > b0:
        return 1
    else:
        return 0


def slerp_quats(Q, bad_idxs, allow_roll=True):
    """replace quats in sequence with interpolated version"""
    for cur_idx in bad_idxs:

        pre_idx = cur_idx - 1
        preQ = None
        while preQ is None:
            if pre_idx < 0:
                raise IndexError
            preQ = Q[pre_idx]
            if len(getnan(nx.array((preQ.w, preQ.x, preQ.y, preQ.z)))[0]):
                preQ = None
                pre_idx -= 1

        post_idx = cur_idx + 1
        postQ = None
        while postQ is None:
            try:
                postQ = Q[post_idx]
            except IndexError:
                raise RuntimeError(
                    "attempted to interpolate orientation with no final orientation value (reduce stop frame)"
                )
            if len(getnan(nx.array((postQ.w, postQ.x, postQ.y, postQ.z)))[0]):
                postQ = None
                post_idx += 1

        frac = float(cur_idx - pre_idx) / float(post_idx - pre_idx)
        # print '  ',frac, cur_idx, pre_idx, post_idx
        new_quat = cgtypes.slerp(frac, preQ, postQ)
        if allow_roll:
            Q[cur_idx] = new_quat
        else:
            # convert back and forth from orientation to eliminate roll
            ori = quat_to_orient(new_quat)
            no_roll_quat = orientation_to_quat(ori)
            Q[cur_idx] = no_roll_quat


def do_it(
    results,
    start_frame=None,
    stop_frame=None,
    Psmooth=None,
    Qsmooth=None,
    alpha=0.2,
    beta=20.0,
    lambda1=2e-9,
    lambda2=1e-11,
    gamma=0.0,
    percent_error_eps_quats=9,
    interp_OK=False,
    return_err_tol=False,
    force_err_tol=None,
    return_frame_numbers=False,
    return_resultant_forces=False,
    return_roll_qsmooth=False,
    return_coronal_dir=False,
    do_smooth_position=False,
    return_smooth_position=False,
    do_smooth_quats=False,
    return_smooth_quats=False,
    plot_pos_and_vel=False,
    plot_ffts=False,
    plot_pos_err_histogram=False,
    plot_vel_vs_accel=False,
    return_vel_vs_pitch_info=False,
    plot_xy=False,
    plot_xy_Qsmooth=False,
    plot_xy_Qraw=True,
    plot_xy_Psmooth=False,
    plot_xz=False,
    plot_xy_air=False,
    plot_force_angle_info=False,
    plot_hists=False,
    plot_hist_horiz_vel=False,
    plot_hist_vert_vel=False,
    plot_forward_vel_vs_pitch_angle=False,
    plot_accel=False,
    plot_smooth_pos_and_vel=False,
    plot_Q=False,
    plot_body_angular_vel=False,
    plot_body_angular_vel2=False,
    plot_error_angles=False,
    plot_body_ground_V=False,
    plot_body_air_V=False,
    plot_forces=False,
    plot_srini_landing_fig=False,
    had_post=True,
    show_grid=False,
    xtitle="time",
    force_scaling=1e7,
    drag_model_for_roll="linear",
    return_drag_force=False,
    return_thrust_force=False,
    fps=100.0,
):

    rad2deg = 180 / math.pi
    deg2rad = 1 / rad2deg
    fps = float(fps)

    #############################################################
    # get position data, make sure there are no holes

    # get data from file
    if isinstance(results, tables.File):
        data3d = results.root.data3d_best
        fXl = [
            (
                row["frame"],
                row["x"],
                row["y"],
                row["z"],
                row["p0"],
                row["p1"],
                row["p2"],
                row["p3"],
                row["p4"],
                row["p5"],
            )
            for row in data3d
            if start_frame <= row["frame"] <= stop_frame
        ]  # XXX
        # data3d.where( start_frame <= data3d.cols.frame <= stop_frame )]
        fXl.sort(sort_on_col0)
    else:
        print("assuming results are numeric")
        fXl = results
    fXl = nx.asarray(fXl)
    frame = fXl[:, 0].astype(nx.int64)

    if start_frame is None:
        start_frame = frame.min()
    else:
        valid_cond = frame >= start_frame
        fXl = fXl[valid_cond]
        frame = fXl[:, 0].astype(nx.int64)
    if stop_frame is None:
        stop_frame = frame.max()
    else:
        valid_cond = frame <= stop_frame
        fXl = fXl[valid_cond]
        frame = fXl[:, 0].astype(nx.int64)

    print("frame[:5]", frame[:5])

    P = fXl[:, 1:4]
    line3d = fXl[:, 4:]

    print("P[:5]", P[:5])
    print("line3d[:5]", line3d[:5])

    # reality check on data to ensure no big jumps -- drops frames
    framediff = frame[1:] - frame[:-1]
    Pdiff = P[1:, :] - P[:-1, :]
    Pdiff_dist = nx.sqrt(nx.sum(Pdiff ** 2, axis=1))
    mean_Pdiff_dist = np.mean(Pdiff_dist)
    std_Pdiff_dist = np.std(Pdiff_dist)
    newframe = [frame[0]]
    newP = [P[0, :]]
    newline3d = [line3d[0, :]]
    cur_ptr = 0
    n_sigma = 5
    if force_err_tol is None:
        err_tol = n_sigma * std_Pdiff_dist
        if err_tol < 30:
            err_tol = 30
            print("at lower limit", end=" ")  # 30 mm/IFI = 3 meters/sec
        else:
            print("calculated", end=" ")
    else:
        err_tol = force_err_tol
        print("given", end=" ")
    print("err_tol", err_tol)

    outputs = []

    if return_err_tol:
        outputs.append(err_tol)

    while (cur_ptr + 1) < frame.shape[0]:
        cur_ptr += 1
        tmpP1 = newP[-1]
        tmpP2 = P[cur_ptr]
        # Pdiff_dist = math.sqrt(nx.sum((newP[-1] - P[cur_ptr])**2))
        Pdiff_dist = math.sqrt(nx.sum((tmpP2 - tmpP1) ** 2))
        if abs(Pdiff_dist - mean_Pdiff_dist) > err_tol:
            print(
                "WARNING: frame %d position difference exceeded %d sigma, ignoring data"
                % (frame[cur_ptr], n_sigma)
            )
            continue
        newframe.append(frame[cur_ptr])
        newP.append(P[cur_ptr])
        newline3d.append(line3d[cur_ptr])
    frame = nx.array(newframe)
    P = nx.array(newP)
    line3d = nx.array(newline3d)

    fXl = nx.concatenate((frame[:, nx.newaxis], P, line3d), axis=1)

    IFI = 1.0 / fps
    t_P = (frame - frame[0]) * IFI  # put in seconds

    to_meters = 1e-3  # put in meters (from mm)
    P = nx.array(P) * to_meters

    line3d = nx.array(line3d)

    # check timestamps
    delta_ts = t_P[1:] - t_P[:-1]
    frames_missing = False
    # interpolate to get fake data where missing
    interpolated_xyz_frames = []
    for i, delta_t in enumerate(delta_ts):
        if not (0.009 < delta_t < 0.011):
            if interp_OK:
                fXl = list(fXl)

                first = frame[i]
                last = frame[i + 1]

                N = last - first
                for ii, fno in enumerate(range(first, last)):
                    if ii == 0:
                        continue
                    frac = ii / float(N)

                    # do interpolation
                    new_x = my_interp(fXl[i][1], fXl[i + 1][1], frac)
                    new_y = my_interp(fXl[i][2], fXl[i + 1][2], frac)
                    new_z = my_interp(fXl[i][3], fXl[i + 1][3], frac)
                    new_row = nx.array(
                        [fno, new_x, new_y, new_z, nan, nan, nan, nan, nan, nan],
                        dtype=fXl[0].dtype,
                    )
                    fXl.append(new_row)
                    print(
                        "  linear interpolation at time %0.2f (frame %d)"
                        % ((fno - start_frame) * 0.01, fno,)
                    )

                    interpolated_xyz_frames.append(fno)
            else:
                frames_missing = True
                print(
                    "are you missing frames between %d and %d?"
                    % (frame[i], frame[i + 1])
                )
    if frames_missing:
        raise ValueError("results have missing frames (hint: interp_OK=True)")

    if len(interpolated_xyz_frames):
        # re-sort and partition results
        fXl.sort(sort_on_col0)

        fXl = nx.array(fXl)
        frame = fXl[:, 0]
        P = fXl[:, 1:4]
        line3d = fXl[:, 4:]

        t_P = (frame - frame[0]) * IFI  # put in seconds
        to_meters = 1e-3  # put in meters (from mm)
        P = nx.array(P) * to_meters
        line3d = nx.array(line3d)

        frame_list = list(frame)
        interped_p_idxs = [frame_list.index(fno) for fno in interpolated_xyz_frames]
    else:
        interped_p_idxs = []

    delta_t = delta_ts[0]

    ################################################################

    if return_frame_numbers:
        outputs.append(frame)

    # get angular position phi
    phi_with_nans = reconstruct.line_direction(line3d)  # unit vector
    slerped_q_idxs = getnan(phi_with_nans[:, 0])[0]
    if len(slerped_q_idxs) and slerped_q_idxs[0] == 0:
        raise ValueError("no orientation for first point")

    Q = QuatSeq([orientation_to_quat(U) for U in phi_with_nans])
    slerp_quats(Q, slerped_q_idxs, allow_roll=False)
    for cur_idx in slerped_q_idxs:
        print(
            "  SLERPed missing quat at time %.2f (frame %d)"
            % (cur_idx * IFI, frame[cur_idx])
        )
    t_bad = nx.take(t_P, slerped_q_idxs)
    # frame_bad = frame[slerped_q_idxs]
    frame_bad = nx.take(frame, slerped_q_idxs)

    #############################################################

    # first position derivative (velocity)
    dPdt = (P[2:] - P[:-2]) / (2 * delta_t)
    t_dPdt = t_P[1:-1]

    # second position derivative (acceleration)
    d2Pdt2 = (P[2:] - 2 * P[1:-1] + P[:-2]) / (delta_t ** 2)
    t_d2Pdt2 = t_P[1:-1]

    # first orientation derivative (angular velocity)
    omega = (Q[:-1].inverse() * Q[1:]).log() / delta_t
    t_omega = t_P[:-1]

    # second orientation derivative (angular acceleration)
    omega_dot = (
        (Q[1:-1].inverse() * Q[2:]).log() - (Q[:-2].inverse() * Q[1:-1]).log()
    ) / (delta_t ** 2)
    t_omega_dot = t_P[1:-1]

    if had_post:
        post_top_center = array([130.85457512, 169.45421191, 50.53490689])
        post_radius = 5  # mm
        post_height = 10  # mm

    ################# Get smooth Position #############################

    if Psmooth is not None:
        Psmooth = nx.array(Psmooth) * to_meters
    elif not do_smooth_position:
        if 1:
            # Psmooth is None and we don't desire recomputation
            # see if we can load cached Psmooth from pytables file
            try:
                smooth_data = results.root.smooth_data
                fPQ = [
                    (
                        row["frame"],
                        row["x"],
                        row["y"],
                        row["z"],
                        row["qw"],
                        row["qx"],
                        row["qy"],
                        row["qz"],
                    )
                    for row in smooth_data
                    if start_frame <= row["frame"] <= stop_frame
                ]  # XXX
                ##               data3d.where( start_frame <= data3d.cols.frame <= stop_frame )]
                fPQ.sort(sort_on_col0)
                if 0 < len(fPQ) < (stop_frame - start_frame + 1):
                    raise ValueError(
                        "pytables file had some but not all data cached for file %s %d-%d"
                        % (results.filename, start_frame, stop_frame)
                    )
                fPQ = nx.array(fPQ)
                Psmooth = fPQ[:, 1:4]
                Psmooth = nx.array(Psmooth) * to_meters
                print("loaded cached Psmooth from file", results.filename)
                if Qsmooth is None and not do_smooth_quats:
                    Qsmooth = QuatSeq([cgtypes.quat(q_wxyz) for q_wxyz in fPQ[:, 4:]])
                    print("loaded cached Qsmooth from file", results.filename)
                print("Psmooth.shape", Psmooth.shape)
            except Exception as exc:
                print("WARNING:", str(exc))
                print("Not using cached smoothed data")
        else:
            ftype = "cheby1"
            wp_hz = 14.0
            gp = 0.001
            ws_hz = 28.0
            gs = 20.0
            hz = 1.0 / delta_t
            wp = wp_hz / hz
            ws = ws_hz / hz
            filt_b, filt_a = scipy.signal.iirdesign(wp, ws, gp, gs, ftype=ftype)

            Psmooth_cols = []
            import scipy_utils

            for col in range(P.shape[1]):
                Psmooth_cols.append(scipy_utils.filtfilt(filt_b, filt_a, P[:, col]))

            Psmooth = nx.array(Psmooth_cols)
            Psmooth.transpose()

    if Psmooth is None and do_smooth_position:
        of = ObjectiveFunctionPosition(
            P, delta_t, alpha, no_distance_penalty_idxs=interped_p_idxs
        )
        # epsilon1 = 200e6
        epsilon1 = 0
        # epsilon1 = 150e6
        # epsilon1 = 1.0
        percent_error_eps = 9
        Psmooth = P.copy()
        last_err = None
        max_iter1 = 10000
        count = 0
        while count < max_iter1:
            count += 1
            start = time.time()
            del_F = of.get_del_F(Psmooth)
            stop = time.time()
            print("P elapsed: % 4.2f secs," % (stop - start,), end=" ")
            err = nx.sum(nx.sum(del_F ** 2, axis=1))
            print("sum( norm(del F)):", err)
            if err < epsilon1:
                break
            elif last_err is not None:
                if err > last_err:
                    print("ERROR: error is increasing, aborting")
                    break
                pct_err = (last_err - err) / last_err * 100.0
                print("   (%3.1f%%)" % (pct_err,))
                if pct_err < percent_error_eps:
                    print("reached percent_error_eps")
                    break
            last_err = err
            Psmooth = Psmooth - lambda1 * del_F
    if do_smooth_position or return_smooth_position:
        outputs.append(Psmooth / to_meters)
    if Psmooth is not None:
        dPdt_smooth = (Psmooth[2:] - Psmooth[:-2]) / (2 * delta_t)
        d2Pdt2_smooth = (Psmooth[2:] - 2 * Psmooth[1:-1] + Psmooth[:-2]) / (
            delta_t ** 2
        )

    if Qsmooth is None and do_smooth_quats:
        print("smoothing quats...")
        # gamma = 1000
        # gamma = 0.0
        of = ObjectiveFunctionQuats(
            Q, delta_t, beta, gamma, no_distance_penalty_idxs=slerped_q_idxs
        )

        # epsilon2 = 200e6
        epsilon2 = 0
        # lambda2 = 2e-9
        # lambda2 = 1e-9
        # lambda2 = 1e-11
        Q_k = Q[:]  # make copy
        last_err = None
        max_iter2 = 2000
        count = 0
        while count < max_iter2:
            count += 1
            start = time.time()
            del_G = of.get_del_G(Q_k)
            D = of._getDistance(Q_k)
            E = of._getEnergy(Q_k)
            R = of._getRoll(Q_k)
            print(
                "  G = %s + %s*%s + %s*%s"
                % (str(D), str(beta), str(E), str(gamma), str(R))
            )
            stop = time.time()
            err = math.sqrt(nx.sum(nx.array(abs(del_G)) ** 2))
            if err < epsilon2:
                print("reached epsilon2")
                break
            elif last_err is not None:
                pct_err = (last_err - err) / last_err * 100.0
                print("Q elapsed: % 6.2f secs," % (stop - start,), end=" ")
                print("current gradient:", err, end=" ")
                print("   (%4.2f%%)" % (pct_err,))

                if err > last_err:
                    print("ERROR: error is increasing, aborting")
                    break
                if pct_err < percent_error_eps_quats:
                    print("reached percent_error_eps_quats")
                    break
            else:
                print("Q elapsed: % 6.2f secs," % (stop - start,), end=" ")
                print("current gradient:", err)
            last_err = err
            Q_k = Q_k * (del_G * -lambda2).exp()
        if count >= max_iter2:
            print("reached max_iter2")
        Qsmooth = Q_k
    if do_smooth_quats or return_smooth_quats:
        outputs.append(Qsmooth)
    if Qsmooth is not None:
        omega_smooth = (Qsmooth[:-1].inverse() * Qsmooth[1:]).log() / delta_t

        omega_dot_smooth = (
            (Qsmooth[1:-1].inverse() * Qsmooth[2:]).log()
            - (Qsmooth[:-2].inverse() * Qsmooth[1:-1]).log()
        ) / (delta_t ** 2)
        do_smooth_quats = True  # we've got 'em now, one way or another

    # body-centric groundspeed (using quaternion rotation)
    body_ground_V = rotate_velocity_by_orientation(dPdt, Q[1:-1])
    if Qsmooth is not None:
        body_ground_V_smooth = rotate_velocity_by_orientation(
            dPdt_smooth, Qsmooth[1:-1]
        )

    airspeed = nx.array((0.0, 0, 0))
    # airspeed = nx.array((-.4,0,0))
    dPdt_air = dPdt - airspeed  # world centric airspeed

    # No need to calculate acceleration relative to air because air
    # velocity is always constant thus d2Pdt2_air == d2Pdt2.

    if Psmooth is not None:
        dPdt_smooth_air = dPdt_smooth - airspeed  # world centric airspeed
    # body-centric airspeed (using quaternion rotation)
    body_air_V = rotate_velocity_by_orientation(dPdt_air, Q[1:-1])

    if 0:
        # check that coordinate xform doesn't affect velocity magnitude
        tmp_V2_a = body_air_V.x ** 2 + body_air_V.y ** 2 + body_air_V.z ** 2
        tmp_V2_b = dPdt_air[:, 0] ** 2 + dPdt_air[:, 1] ** 2 + dPdt_air[:, 2] ** 2

        for i in range(len(tmp_V2_a)):
            print(abs(tmp_V2_a[i] - tmp_V2_b[i]), " near 0?")

    if Qsmooth is not None:
        body_air_V_smooth = rotate_velocity_by_orientation(
            dPdt_smooth_air, Qsmooth[1:-1]
        )

    # compute body-centric angular velocity
    omega_body = rotate_velocity_by_orientation(omega, Q[:-1])
    if Qsmooth is not None:
        omega_smooth_body = rotate_velocity_by_orientation(omega_smooth, Qsmooth[:-1])
    t_omega_body = t_P[:-1]

    if Qsmooth is not None:  # compute forces (for now, smooth data only)
        # vector for current orientation (use only indices with velocity info)
        orient_parallel = quat_to_orient(Qsmooth)[1:-1]

        # vector for current velocity
        Vair_orient = dPdt_air / nx.sqrt(nx.sum(dPdt_air ** 2, axis=1)[:, nx.newaxis])
        # compute alpha == angle of attack
        aattack = nx.arccos(
            [nx.dot(v, p) for v, p in zip(Vair_orient, orient_parallel)]
        )
        # print aattack*rad2deg

        if 0:
            Vmag_air2 = body_air_V.x ** 2 + body_air_V.y ** 2 + body_air_V.z ** 2
        else:
            Vmag_air2 = (
                body_air_V_smooth.x ** 2
                + body_air_V_smooth.y ** 2
                + body_air_V_smooth.z ** 2
            )

        make_norm = reconstruct.norm_vec  # normalize vector
        if 0:
            # calculate body drag based on wooden semi-cylinder model
            # of Fry & Dickson (unpublished)

            # find vector for normal force
            tmp_out_of_plane = [
                cross(v, p) for v, p in zip(Vair_orient, orient_parallel)
            ]
            orient_normal = [
                cross(p, t) for t, p in zip(tmp_out_of_plane, orient_parallel)
            ]
            orient_normal = nx.array([make_norm(o_n) for o_n in orient_normal])

            cyl_diam = 0.5  # mm
            cyl_diam = cyl_diam / 1e3  # meters
            cyl_height = 1.75  # mm
            cyl_height = cyl_height / 1e3  # meters
            A = cyl_diam * cyl_height

            rho = 1.25  # kg/m^3

            C_P = (
                0.16664033221423064 * nx.cos(aattack)
                + 0.33552465566450407 * nx.cos(aattack) ** 3
            )
            C_N = 0.75332031249999987 * nx.sin(aattack)

            F_P = 0.5 * rho * A * C_P * Vmag_air2
            F_N = 0.5 * rho * A * C_N * Vmag_air2

            # convert normal and parallel forces back to world coords
            body_drag_world1 = nx.array(
                [orient_parallel[i] * -F_P[i] for i in range(len(F_P))]
            )
            body_drag_world2 = nx.array(
                [orient_normal[i] * -F_N[i] for i in range(len(F_N))]
            )
            body_drag_world = body_drag_world1 + body_drag_world2
        else:
            body_drag_world = None

        t_forces = t_dPdt

        # force required to stay aloft
        fly_mass = 1e-6  # guesstimate (1 milligram)
        G = 9.81  # gravity: meters / second / second
        aloft_force = fly_mass * G

        # resultant force
        # my'' = F + mg - Cy'^2
        # F = m(y''-g) - Cy'^2
        # F = m(y''-g) - body_drag_world

        Garr = nx.array([[0, 0, -9.81]] * len(t_forces))
        resultant = fly_mass * (d2Pdt2_smooth - Garr)
        if body_drag_world is not None:
            resultant = resultant - body_drag_world
        if return_resultant_forces:
            outputs.append((frame[1:-1], resultant))  # return frame numbers also

        # We can attempt to decompose the resultant force r into
        # "thrust" t (up and forward relative to body) and "drag" d as
        # follows. t + d = r --> r + -d = t

        # Calculate "drag" at given velocity.

        Vmag_air = nx.sqrt(Vmag_air2)
        Vmag_air.shape = Vmag_air.shape[0], 1
        Vdir_air = dPdt_air / Vmag_air
        if drag_model_for_roll == "linear":

            # Assume drag is linearly proportional to velocity as
            # asserted # by Charlie David, 1978, but with the
            # appropriate # coefficients, this makes little difference
            # on the body angle # vs. terminal velocity relationship.

            # Cf_linear was calculated to produce angle of attack
            # vs. terminal velocity relation roughly equal to curve of
            # # David 1978.

            Cf_linear = -0.000012
            drag_force = Cf_linear * Vmag_air * Vdir_air
        elif drag_model_for_roll == "v^2":
            Cf_V2 = -0.000015
            V2 = Vmag_air2[:, nx.newaxis]
            drag_force = Cf_V2 * V2 * Vdir_air

        if return_drag_force:
            outputs.append((frame[1:-1], drag_force))  # return frame numbers also

        print("used drag model:", drag_model_for_roll, "to compute roll angle")
        thrust_force = resultant - drag_force

        if return_thrust_force:
            outputs.append((frame[1:-1], thrust_force))  # return frame numbers also

        # 2 planes : saggital and coronal
        # Project thrust_force onto coronal plane.

        # Do this by eliminating component of thrust_force in body
        # axis direction.

        # subtract component of force in direction of fly's body
        coronal_thrust_force = nx.array(
            [tf - nx.dot(tf, op) * op for tf, op in zip(thrust_force, orient_parallel)]
        )

        # get direction of this force
        coronal_thrust_dir = nx.array([make_norm(ctf) for ctf in coronal_thrust_force])

        fly_up = cgtypes.quat(0, 0, 0, 1)  # fly up vector in fly coords
        # make sure there is no roll component to offset our results:
        Qsmooth_zero_roll = QuatSeq(
            [
                euler_to_quat(
                    yaw=quat_to_euler(q)[0], pitch=quat_to_euler(q)[1], roll=0
                )
                for q in Qsmooth
            ]
        )
        fly_up_world = [q * fly_up * q.inverse() for q in Qsmooth_zero_roll[1:-1]]
        fly_up_world = nx.array([(v.x, v.y, v.z) for v in fly_up_world])

        if 1:
            cos_roll = nx.array(
                [nx.dot(ctd, fuw) for ctd, fuw in zip(coronal_thrust_dir, fly_up_world)]
            )
            guess_roll = nx.arccos(cos_roll)

        if 0:
            # mcp = | u x v | = |u||v|sin t
            # dp = u . v =     |u||v|cos t
            # atan2(mcp,dp) = t
            cp = [cross(u, v) for u, v in zip(fly_up_world, coronal_thrust_dir)]
            mcp = nx.array([math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) for u in cp])
            dp = nx.array(
                [nx.dot(u, v) for u, v in zip(fly_up_world, coronal_thrust_dir)]
            )

            guess_roll2 = [math.atan2(num, denom) for num, denom in zip(mcp, dp)]

            if 0:
                for r1, r2 in zip(guess_roll, guess_roll2):
                    print(r1, "?=", r2)

        if 1:
            # XXX hack to fix some sign error somewhere (ARGH!!)
            # Note: may not be sign error -- quats represent same rotation
            # with 2 quats...
            Qsmooth_roll_guess = Qsmooth[:]
            for i in range(1, len(Qsmooth_roll_guess) - 1):
                q = Qsmooth[i]
                yaw, pitch, old_roll = quat_to_euler(q)
                new_roll = guess_roll[i - 1]
                # roll = old_roll - new_roll

                roll = new_roll
                qnew = euler_to_quat(yaw=yaw, pitch=pitch, roll=roll)
                fly_up = cgtypes.quat(0, 0, 0, 1)  # fly up vector in fly coords
                v = qnew * fly_up * qnew.inverse()
                fly_up_world = nx.array((v.x, v.y, v.z))

                ctd = coronal_thrust_dir[i - 1]
                dotprod = nx.dot(ctd, fly_up_world)
                # print '=== ROLL',roll*rad2deg, dotprod
                if dotprod < 1 - 1e-10:
                    roll = -new_roll
                    qnew = euler_to_quat(yaw=yaw, pitch=pitch, roll=roll)
                    fly_up = cgtypes.quat(0, 0, 0, 1)  # fly up vector in fly coords
                    v = qnew * fly_up * qnew.inverse()
                    fly_up_world = nx.array((v.x, v.y, v.z))

                    dotprod = nx.dot(ctd, fly_up_world)
                    # print '   -ROLL',roll*rad2deg, dotprod
                Qsmooth_roll_guess[i] = qnew

        if return_roll_qsmooth:
            outputs.append(Qsmooth_roll_guess)

        if return_coronal_dir:
            outputs.append(
                (frame[1:-1], coronal_thrust_dir)
            )  # return frame numbers also

    if 1:  # compute error angles (versus tangent line)
        make_norm = reconstruct.norm_vec
        rad2deg = 180 / math.pi

        # raw

        flat_heading = dPdt_air.copy()
        flat_heading[:, 2] = 0  # no Z component
        flat_heading = [make_norm(f) for f in flat_heading]
        flat_heading_angle = nx.array(
            [math.atan2(f[1], f[0]) * rad2deg for f in flat_heading]
        )

        vel_air_dir = [make_norm(v) for v in dPdt_air]
        vel_pitch_angle = nx.array([math.asin(v[2]) * rad2deg for v in vel_air_dir])

        flat_orientation = quat_to_orient(Q)
        flat_orientation[:, 2] = 0
        flat_orientation = [make_norm(f) for f in flat_orientation]
        flat_orientation_angle = nx.array(
            [math.atan2(f[1], f[0]) * rad2deg for f in flat_orientation]
        )

        orient_pitch_angle = [quat_to_euler(q)[1] * rad2deg for q in Q]

        heading_err = flat_orientation_angle[1:-1] - flat_heading_angle
        t_heading_err = t_dPdt
        pitch_body_err = orient_pitch_angle[1:-1] - vel_pitch_angle
        t_pitch_body_err = t_dPdt
        # pitch_body_err = [ math.asin(p[2])*rad2deg for p in quat_to_orient(Q) ]
        # t_pitch_body_err = t_P

        #   derivs
        d_heading_err_dt = (heading_err[2:] - heading_err[:-2]) / (2 * delta_t)
        t_d_heading_err_dt = t_heading_err[1:-1]
        d_pitch_body_err_dt = (pitch_body_err[2:] - pitch_body_err[:-2]) / (2 * delta_t)
        t_d_pitch_body_err_dt = t_pitch_body_err[1:-1]

        # smooth

        if Psmooth is not None and Qsmooth is not None:
            flat_heading_smooth = dPdt_smooth_air.copy()
            flat_heading_smooth[:, 2] = 0  # no Z component
            flat_heading_smooth = [make_norm(f) for f in flat_heading_smooth]
            flat_heading_angle_smooth = nx.array(
                [math.atan2(f[1], f[0]) * rad2deg for f in flat_heading_smooth]
            )

            vel_smooth_air_dir = [make_norm(v) for v in dPdt_smooth_air]
            vel_smooth_pitch_angle = nx.array(
                [math.asin(v[2]) * rad2deg for v in vel_smooth_air_dir]
            )

            flat_orientation_smooth = quat_to_orient(Qsmooth)
            flat_orientation_smooth[:, 2] = 0
            flat_orientation_smooth = [make_norm(f) for f in flat_orientation_smooth]
            flat_orientation_angle_smooth = nx.array(
                [math.atan2(f[1], f[0]) * rad2deg for f in flat_orientation_smooth]
            )

            orient_pitch_angle_smooth = [quat_to_euler(q)[1] * rad2deg for q in Qsmooth]

            heading_err_smooth = (
                flat_orientation_angle_smooth[1:-1] - flat_heading_angle_smooth
            )
            t_heading_err_smooth = t_dPdt
            pitch_body_err_smooth = (
                orient_pitch_angle_smooth[1:-1] - vel_smooth_pitch_angle
            )
            t_pitch_body_err_smooth = t_dPdt
            # pitch_body_err_smooth = nx.array([ math.asin(p[2])*rad2deg for p in quat_to_orient(Qsmooth) ])
            # t_pitch_body_err_smooth = t_P

            #   derivs
            d_heading_err_smooth_dt = (
                heading_err_smooth[2:] - heading_err_smooth[:-2]
            ) / (2 * delta_t)
            t_d_heading_err_smooth_dt = t_heading_err_smooth[1:-1]
            d_pitch_body_err_smooth_dt = (
                pitch_body_err_smooth[2:] - pitch_body_err_smooth[:-2]
            ) / (2 * delta_t)
            t_d_pitch_body_err_smooth_dt = t_pitch_body_err_smooth[1:-1]

    if 1:  # compute horizontal velocity
        xvel, yvel = dPdt[:, 0], dPdt[:, 1]
        horiz_vel = nx.sqrt(xvel ** 2 + yvel ** 2)
        vert_vel = dPdt[:, 2]

    if plot_hists:
        print("plot_hists")
        ax1 = subplot(3, 1, 1)
        n, bins, patches = hist(dPdt[:, 0], bins=30)
        setp(patches, "facecolor", (1, 0, 0))

        ax2 = subplot(3, 1, 2, sharex=ax1)
        n, bins, patches = hist(dPdt[:, 1], bins=30)
        setp(patches, "facecolor", (0, 1, 0))

        ax3 = subplot(3, 1, 3, sharex=ax1)
        n, bins, patches = hist(dPdt[:, 2], bins=30)
        setp(patches, "facecolor", (0, 0, 1))

    elif plot_force_angle_info:
        print("plot_force_angle_info")
        R = [make_norm(r) for r in resultant]  # resultant orientation in V3
        B = [make_norm(r) for r in orient_parallel]  # body orientation in V3
        angle = nx.arccos([nx.dot(r, b) for (r, b) in zip(R, B)]) * rad2deg

        # remove periods where fly has landed
        vel_mag_smooth = nx.sqrt(nx.sum(dPdt_smooth ** 2, axis=1))
        good_idx = nx.where(vel_mag_smooth > 0.01)[0]  # only where vel > 10 mm/sec

        subplot(2, 1, 1)
        n, bins, patches = hist(nx.take(angle, good_idx), bins=30)
        xlabel("force angle (degrees)")
        ylabel("count")

        z_vel = dPdt[:, 2]

        subplot(2, 1, 2)
        if 1:
            plot(nx.take(z_vel, good_idx), nx.take(angle, good_idx), ".")
        else:
            resultant_mag = nx.sqrt(nx.sum(resultant ** 2, axis=1))
            scatter(
                nx.take(z_vel, good_idx),
                nx.take(angle, good_idx),
                10 * nx.ones((len(good_idx),)),
                # nx.take(resultant_mag*1e6,good_idx),
                nx.take(resultant_mag, good_idx),
            )

        if 0:
            for idx in good_idx:
                if z_vel[idx] < -0.6:
                    text(z_vel[idx], angle[idx], str(t_dPdt[idx]))
        xlabel("z velocity (m/sec)")
        ylabel("force angle (degrees)")

    elif plot_hist_horiz_vel:
        print("plot_hist_horiz_vel")
        hist(horiz_vel, bins=30)
        xlabel("horizontal velocity (m/sec)")
        ylabel("count")

    elif plot_hist_vert_vel:
        print("plot_hist_vert_vel")
        hist(vert_vel, bins=30)
        xlabel("vertical velocity (m/sec)")
        ylabel("count")

    elif plot_forward_vel_vs_pitch_angle:
        print("plot_forward_vel_vs_pitch_angle")
        vert_vel_limit = 0.1  # meter/sec
        hvels = []
        pitches = []
        for i in range(len(vert_vel)):
            if abs(vert_vel[i]) < vert_vel_limit:
                hvels.append(horiz_vel[i])
                pitch = (orient_pitch_angle[i] + orient_pitch_angle[i + 1]) / 2.0
                pitches.append(pitch)
        plot(hvels, pitches, "k.")
        xlabel("horizontal velocity (m/sec)")
        ylabel("pitch angle (degrees)")

    elif plot_pos_and_vel:
        print("plot_pos_and_vel")
        ioff()
        try:
            linewidth = 1.5
            ax1 = subplot(3, 1, 1)
            title("ground speed, global reference frame")
            plot(t_P, P[:, 0], "rx", t_P, P[:, 1], "gx", t_P, P[:, 2], "bx")
            print("t_P, P[:,0]", len(t_P), len(P[:, 0]))
            if Psmooth is not None:
                smooth_lines = plot(
                    t_P,
                    Psmooth[:, 0],
                    "r-",
                    t_P,
                    Psmooth[:, 1],
                    "g-",
                    t_P,
                    Psmooth[:, 2],
                    "b-",
                )
                setp(smooth_lines, "linewidth", linewidth)
            ylabel("Position\n(m)")
            grid()

            subplot(3, 1, 2, sharex=ax1)
            plot(
                t_dPdt,
                dPdt[:, 0],
                "rx",
                t_dPdt,
                dPdt[:, 1],
                "gx",
                t_dPdt,
                dPdt[:, 2],
                "bx",
                t_dPdt,
                nx.sqrt(nx.sum(dPdt ** 2, axis=1)),
                "kx",
            )
            if Psmooth is not None:
                smooth_lines = plot(
                    t_dPdt,
                    dPdt_smooth[:, 0],
                    "r-",
                    t_dPdt,
                    dPdt_smooth[:, 1],
                    "g-",
                    t_dPdt,
                    dPdt_smooth[:, 2],
                    "b-",
                    t_dPdt,
                    nx.sqrt(nx.sum(dPdt_smooth ** 2, axis=1)),
                    "k-",
                )
                legend(smooth_lines, ("x", "y", "z", "mag"))
                setp(smooth_lines, "linewidth", linewidth)
            ylabel("Velocity\n(m/sec)")
            grid()

            subplot(3, 1, 3, sharex=ax1)
            if Psmooth is not None:
                smooth_lines = plot(
                    t_d2Pdt2,
                    d2Pdt2_smooth[:, 0],
                    "r-",
                    t_d2Pdt2,
                    d2Pdt2_smooth[:, 1],
                    "g-",
                    t_d2Pdt2,
                    d2Pdt2_smooth[:, 2],
                    "b-",
                    t_d2Pdt2,
                    nx.sqrt(nx.sum(d2Pdt2_smooth ** 2, axis=1)),
                    "k-",
                )

                setp(smooth_lines, "linewidth", linewidth)
            else:
                plot(
                    t_d2Pdt2,
                    d2Pdt2[:, 0],
                    "r-",
                    t_d2Pdt2,
                    d2Pdt2[:, 1],
                    "g-",
                    t_d2Pdt2,
                    d2Pdt2[:, 2],
                    "b-",
                )

            ylabel("Acceleration\n(m/sec/sec)")
            xlabel("Time (sec)")
            grid()
        finally:
            ion()

    elif plot_ffts:
        print("plot_ffts")
        ioff()
        try:
            NFFT = 128
            ax1 = subplot(3, 1, 1)
            pxx, freqs = matplotlib.mlab.psd(
                P[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "ro-")
            pxx, freqs = matplotlib.mlab.psd(
                Psmooth[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "r-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                P[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "go-")
            pxx, freqs = matplotlib.mlab.psd(
                Psmooth[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "g-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                P[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "bo-")
            pxx, freqs = matplotlib.mlab.psd(
                Psmooth[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "b-", lw=2)

            ylabel("Position Power Spectrum (dB)")

            ax2 = subplot(3, 1, 2, sharex=ax1)
            pxx, freqs = matplotlib.mlab.psd(
                dPdt[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "ro-")
            pxx, freqs = matplotlib.mlab.psd(
                dPdt_smooth[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "r-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                dPdt[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "go-")
            pxx, freqs = matplotlib.mlab.psd(
                dPdt_smooth[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "g-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                dPdt[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "bo-")
            pxx, freqs = matplotlib.mlab.psd(
                dPdt_smooth[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "b-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                nx.sqrt(nx.sum(dPdt ** 2, axis=1)),
                detrend=detrend_mean,
                NFFT=NFFT,
                Fs=100.0,
            )
            plot(freqs, 10 * nx.log10(pxx), "ko-")
            pxx, freqs = matplotlib.mlab.psd(
                nx.sqrt(nx.sum(dPdt_smooth ** 2, axis=1)),
                detrend=detrend_mean,
                NFFT=NFFT,
                Fs=100.0,
            )
            plot(freqs, 10 * nx.log10(pxx), "k-", lw=2)

            ylabel("Velocity Power Spectrum (dB)")

            ax3 = subplot(3, 1, 3, sharex=ax1)
            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "ro-")
            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2_smooth[:, 0], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "r-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "go-")
            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2_smooth[:, 1], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "g-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "bo-")
            pxx, freqs = matplotlib.mlab.psd(
                d2Pdt2_smooth[:, 2], detrend=detrend_mean, NFFT=NFFT, Fs=100.0
            )
            plot(freqs, 10 * nx.log10(pxx), "b-", lw=2)

            pxx, freqs = matplotlib.mlab.psd(
                nx.sqrt(nx.sum(d2Pdt2 ** 2, axis=1)),
                detrend=detrend_mean,
                NFFT=NFFT,
                Fs=100.0,
            )
            plot(freqs, 10 * nx.log10(pxx), "ko-")
            # plot( freqs, pxx, 'ko-' )
            pxx, freqs = matplotlib.mlab.psd(
                nx.sqrt(nx.sum(d2Pdt2_smooth ** 2, axis=1)),
                detrend=detrend_mean,
                NFFT=NFFT,
                Fs=100.0,
            )
            plot(freqs, 10 * nx.log10(pxx), "k-", lw=2)
            # plot( freqs, pxx, 'k-', lw=2 )
            xlabel("freq (Hz)")
            # ylabel( 'Acceleration Power Spectrum')
            ylabel("Acceleration Power Spectrum (dB)")

        finally:
            ion()

    elif plot_vel_vs_accel or return_vel_vs_pitch_info:
        print("plot_vel_vs_accel")
        #    elif plot_vel_vs_accel or return_vel_vs_pitch_info or plot_z_vel_vs_horiz_vel:
        if Psmooth is None:
            # raise RuntimeError("need smoothed postion data")
            print("WARNING: using un-smoothed acceleration data")
            z_vel = dPdt[:, 2]
            abs_z_vel = nx.abs(z_vel)
            acc_mag = nx.sqrt(nx.sum(d2Pdt2 ** 2, axis=1))
        else:
            z_vel = dPdt_smooth[:, 2]
            abs_z_vel = nx.abs(z_vel)
            acc_mag = nx.sqrt(nx.sum(d2Pdt2_smooth ** 2, axis=1))

        # find where acceleration meets criterea
        if 1:  # not much trend
            criterion1 = set(nx.where(acc_mag < 1.2)[0])
            criterion2 = set(nx.where(abs_z_vel < 0.025)[0])
        else:
            criterion1 = set(nx.where(acc_mag < 2)[0])
            criterion2 = set(nx.where(abs_z_vel < 0.05)[0])

        ok_acc_idx = list(criterion1 & criterion2)
        ok_acc_idx.sort()
        ok_acc_idx = nx.array(ok_acc_idx)

        # break into sequences of contiguous frames
        ok_seqs = []
        fdiff = ok_acc_idx[1:] - ok_acc_idx[:-1]
        cur_seq = []
        if len(ok_acc_idx) >= 4:
            for i in range(len(ok_acc_idx) - 1):
                idx = ok_acc_idx[i]
                cur_seq.append(idx)
                if fdiff[i] != 1:
                    ok_seqs.append(cur_seq)
                    cur_seq = []
            cur_seq.append(ok_acc_idx[-1])
            ok_seqs.append(cur_seq)

        # only accept sequences of length 4 or greater
        ok_seqs = [ok_seq for ok_seq in ok_seqs if len(ok_seq) >= 4]

        if return_vel_vs_pitch_info:

            # convert quaternions to orientations (unit vectors)
            if Qsmooth is not None:
                orients_smooth = quat_to_orient(Qsmooth)
            orients = quat_to_orient(Q)

            # calculate average velocity and average pitch angle for each OK sequence
            select_vels = []
            select_pitches = []
            if Qsmooth is not None:
                select_pitches_smooth = []

            for ok_seq in ok_seqs:
                vel_mag = nx.sqrt(nx.sum(nx.take(dPdt_smooth, ok_seq) ** 2, axis=1))

                if Qsmooth is not None:
                    ok_orients_smooth = []
                    for idx in ok_seq:
                        orient = orients_smooth[idx]
                        if orient[2] < 0:
                            orient = -orient  # flip so fly points up
                        ok_orients_smooth.append(orient)

                    ok_orients_smooth = nx.array(ok_orients_smooth)
                    ok_orients_smooth_z = ok_orients_smooth[:, 2]
                    ok_pitch_smooth = nx.arcsin(ok_orients_smooth_z) * rad2deg

                ok_orients = []
                good_pitch_idxs = []
                for iii, idx in enumerate(ok_seq):
                    if idx not in slerped_q_idxs:  # don't use interpolated orientation
                        good_pitch_idxs.append(iii)
                    ##                    else:
                    ##                        print 'WARNING: using interpolated Qsmooth'
                    ##                        #continue
                    orient = orients[idx]
                    if orient[2] < 0:
                        orient = -orient  # flip so fly points up
                    ok_orients.append(orient)

                ok_orients = nx.array(ok_orients)
                ok_orients_z = ok_orients[:, 2]
                ok_pitch = nx.arcsin(ok_orients_z) * rad2deg

                if Qsmooth is not None:
                    outputs.append(
                        (vel_mag, ok_pitch, ok_pitch_smooth, good_pitch_idxs)
                    )
                else:
                    outputs.append((vel_mag, ok_pitch, good_pitch_idxs))

        elif plot_vel_vs_accel:
            ioff()
            try:

                plot(z_vel, acc_mag, "kx-")
                for i in range(0, len(z_vel), 10):
                    text(z_vel[i], acc_mag[i], "%0.02f" % t_d2Pdt2[i])
                for ok_seq in ok_seqs:
                    tmp_vel = [z_vel[i] for i in ok_seq]
                    tmp_acc = [acc_mag[i] for i in ok_seq]
                    plot(tmp_vel, tmp_acc, "b-", lw=2)

                xlabel("Z velocity (m/sec)")
                ylabel("total acceleration (m/sec/sec)")
            finally:
                ion()

    elif plot_pos_err_histogram:
        print("plot_pos_err_histogram")

        # subplot(2,1,1)
        axes([0.075, 0.575, 0.85, 0.375])
        x_err = list((Psmooth[:, 0] - P[:, 0]) * 1000.0)
        y_err = list((Psmooth[:, 1] - P[:, 1]) * 1000.0)
        z_err = list((Psmooth[:, 2] - P[:, 2]) * 1000.0)

        xlim = -0.2, 0.2
        ylim = 0, 20
        color_alpha = 0.5

        xlines = hist(x_err, bins=17)[2]
        ##        ylabel('counts')
        ##        set(gca(),'ylim',ylim)
        ##        set(gca(),'xlim',xlim)
        setp(xlines, "alpha", color_alpha)
        setp(xlines, "facecolor", (1, 0, 0))

        ylines = hist(y_err, bins=19)[2]
        ##        ylabel('counts')
        ##        setp(gca(),'xlim',xlim)
        ##        setp(gca(),'ylim',ylim)
        setp(ylines, "alpha", color_alpha)
        setp(ylines, "facecolor", (0, 1, 0))

        zlines = hist(z_err, bins=50)[2]
        legend((xlines[0], ylines[0], zlines[0]), ["X", "Y", "Z"])
        ylabel("counts")
        setp(gca(), "ylim", ylim)
        setp(gca(), "xlim", xlim)
        setp(zlines, "alpha", color_alpha)
        setp(zlines, "facecolor", (0, 0, 1))

        grid()
        xlabel("distance from smoothed data (mm)")

        # subplot(2,1,2)
        axes([0.075, 0.0975, 0.85, 0.375])
        rad2deg = 180 / math.pi
        euler_smooth = nx.array([quat_to_euler(q) for q in Qsmooth]) * rad2deg
        euler = nx.array([quat_to_euler(q) for q in Q]) * rad2deg

        yaw_err = list(euler_smooth[:, 0] - euler[:, 0])
        pitch_err = list(euler_smooth[:, 1] - euler[:, 1])
        roll_err = list(euler_smooth[:, 2] - euler[:, 2])

        xlim = -60, 60
        ylim = 0, 33
        ##        color_alpha = 0.6

        yawlines = hist(yaw_err, bins=25)[2]
        ylabel("counts")
        ##        setp(gca(),'ylim',ylim)
        ##        setp(gca(),'xlim',xlim)
        setp(yawlines, "alpha", color_alpha)
        setp(yawlines, "facecolor", (1, 0, 0))

        pitchlines = hist(pitch_err, bins=50)[2]
        ##        ylabel('counts')
        setp(gca(), "xlim", xlim)
        setp(gca(), "ylim", ylim)
        setp(pitchlines, "alpha", color_alpha)
        setp(pitchlines, "facecolor", (0, 1, 0))

        legend([yawlines[0], pitchlines[0]], ["yaw", "pitch"])

        ##        rolllines = hist(roll_err, bins = 5)[2]
        ##        legend((xlines[0],ylines[0],rolllines[0]),['yaw','pitch','roll'])
        ##        ylabel('counts')
        ##        setp(gca(),'ylim',ylim)
        ##        setp(gca(),'xlim',xlim)
        ##        setp(rolllines,'alpha',color_alpha)
        ##        setp(rolllines,'facecolor',(0,0,1))

        grid()
        xlabel("distance from smoothed data (deg)")

    elif plot_srini_landing_fig:
        print("plot_srini_landing_fig")
        ioff()
        try:
            clf()
            # subplot 2,2,1 ##############################
            ax = subplot(2, 2, 1)

            plot(P[:, 0] * 1000, P[:, 1] * 1000, "ko", mfc=(1, 1, 1), markersize=4)

            # if plot_xy_Psmooth:
            #    plot(Psmooth[:,0]*1000,Psmooth[:,1]*1000,'b-')#,mfc=(1,1,1),markersize=2)

            for idx in range(len(t_P)):
                if xtitle == "all frames":
                    text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(frame[idx]))
                elif idx % 10 == 0:
                    if xtitle == "time":
                        text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(t_P[idx]))
                    elif xtitle == "frame":
                        text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(frame[idx]))

            for use_it, data, color in [  # [plot_xy_Qsmooth,Qsmooth,  (0,0,1,1)],
                [plot_xy_Qraw, Q, (0, 0, 0, 1)]
            ]:
                if use_it:
                    segments = []
                    for i in range(len(P)):
                        pi = P[i]
                        qi = data[i]
                        Pqi = quat_to_orient(qi)
                        segment = (
                            (pi[0] * 1000, pi[1] * 1000),  # x1  # y1
                            (
                                pi[0] * 1000 - Pqi[0] * 3,  # x2
                                pi[1] * 1000 - Pqi[1] * 3,
                            ),
                        )  # y2
                        segments.append(segment)

                    collection = LineCollection(
                        segments, colors=[color] * len(segments)
                    )
                    gca().add_collection(collection)
            xlabel("x (mm)")
            ylabel("y (mm)")

            if show_grid:
                grid()

            # subplot 2,2,2 ##############################
            ax = subplot(2, 2, 2)
            horiz_dists = nx.sqrt(nx.sum((P[1, 0:2] - P[:-1, 0:2]) ** 2, axis=1))
            horiz_dists_cum = [0.0]
            for horiz_dist in horiz_dists:
                horiz_dists_cum.append(horiz_dists_cum[-1] + horiz_dist)
            horiz_dists_cum = nx.array(horiz_dists_cum[1:])
            horiz_dist_height = (P[1:, 2] + P[:-1, 2]) * 0.5  # average between 2 points
            height_offset = min(horiz_dist_height)
            horiz_dist_height = horiz_dist_height - height_offset
            plot(
                horiz_dists_cum * 1000.0, horiz_dist_height * 1000.0, "ko", markersize=4
            )
            xlabel("Horizontal distance travelled (mm)")
            ylabel("Height (mm)")

            # subplot 2,2,3 ##############################
            ax = subplot(2, 2, 3)
            horiz_vel = nx.sqrt(nx.sum(dPdt[:, 0:2] ** 2, axis=1))
            horiz_dist_height = P[1:-1, 2]
            height_offset = min(horiz_dist_height)
            horiz_dist_height = horiz_dist_height - height_offset
            plot(horiz_dist_height * 1000.0, horiz_vel, "ko", mfc="white", markersize=4)
            if xtitle == "frame":
                for ip1 in range(len(horiz_dist_height)):
                    fno = frame[ip1 - 1]
                    text(horiz_dist_height[ip1] * 1000.0, horiz_vel[ip1], str(fno))
            xlabel("Height (mm)")
            ylabel("Horizontal flight speed (m/s)")
        finally:
            ion()

    elif plot_xy:
        print("plot_xy")
        ioff()
        try:
            print("plotting")
            axes([0.1, 0.1, 0.8, 0.8])
            title("top view")

            if had_post:
                theta = linspace(0, 2 * math.pi, 30)[:-1]
                postxs = post_radius * nx.cos(theta) + post_top_center[0]
                postys = post_radius * nx.sin(theta) + post_top_center[1]
                fill(postxs, postys)

            ##        title('top view (ground frame)')
            plot(P[:, 0] * 1000, P[:, 1] * 1000, "ko", mfc=(1, 1, 1), markersize=2)
            print("len(P[:,0])", len(P[:, 0]))
            ##        plot(P[:,0]*1000,P[:,1]*1000,'ko',mfc=(1,1,1),markersize=4)

            if 1:
                fx0 = P[1:-1, 0] * 1000
                fy0 = P[1:-1, 1] * 1000
                for force_type, force_color in [
                    (resultant, (1, 0, 0, 1)),
                    (body_drag_world, (0, 1, 0, 1)),
                ]:
                    if force_type is None:
                        continue
                    fx1 = fx0 + force_type[:, 0] * force_scaling
                    fy1 = fy0 + force_type[:, 1] * force_scaling

                    segments = []
                    for i in range(len(fx0)):
                        segment = ((fx0[i], fy0[i]), (fx1[i], fy1[i]))
                        segments.append(segment)
                    collection = LineCollection(
                        segments, colors=[force_color] * len(segments)
                    )
                    gca().add_collection(collection)

            if plot_xy_Psmooth:
                print("plotted Psmooth")
                plot(
                    Psmooth[:, 0] * 1000, Psmooth[:, 1] * 1000, "b-"
                )  # ,mfc=(1,1,1),markersize=2)

            for idx in range(len(t_P)):
                if xtitle == "all frames":
                    text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(frame[idx]))
                elif idx % 10 == 0:
                    if xtitle == "time":
                        text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(t_P[idx]))
                    elif xtitle == "frame":
                        text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(frame[idx]))

            for use_it, data, color in [
                [plot_xy_Qsmooth, Qsmooth, (0, 0, 1, 1)],
                [plot_xy_Qraw, Q, (0, 0, 0, 1)],
            ]:
                if use_it:
                    segments = []
                    for i in range(len(P)):
                        pi = P[i]
                        qi = data[i]
                        Pqi = quat_to_orient(qi)
                        segment = (
                            (pi[0] * 1000, pi[1] * 1000),  # x1  # y1
                            (
                                pi[0] * 1000 - Pqi[0] * 2,  # x2
                                pi[1] * 1000 - Pqi[1] * 2,
                            ),
                        )  # y2
                        segments.append(segment)

                    collection = LineCollection(
                        segments, colors=[color] * len(segments)
                    )
                    gca().add_collection(collection)
            xlabel("x (mm)")
            ylabel("y (mm)")
            # t=text( 0.6, .2, '<- wind (0.4 m/sec)', transform = gca().transAxes)

            if show_grid:
                grid()
        finally:
            ion()
    ##        show()
    ##        print 'shown...'
    elif plot_xz:
        print("plot_xz")
        ioff()
        try:
            axes([0.1, 0.1, 0.8, 0.8])
            title("side view")
            # title('side view (ground frame)')

            if had_post:
                postxs = [
                    post_top_center[0] + post_radius,
                    post_top_center[0] + post_radius,
                    post_top_center[0] - post_radius,
                    post_top_center[0] - post_radius,
                ]
                postzs = [
                    post_top_center[2],
                    post_top_center[2] - post_height,
                    post_top_center[2] - post_height,
                    post_top_center[2],
                ]
                fill(postxs, postzs)

            # plot(P[:,0]*1000,P[:,2]*1000,'ko',mfc=(1,1,1),markersize=4)
            plot(P[:, 0] * 1000, P[:, 2] * 1000, "ko", mfc=(1, 1, 1), markersize=2)

            for idx in range(len(t_P)):
                if idx % 10 == 0:
                    if xtitle == "time":
                        text(P[idx, 0] * 1000, P[idx, 2] * 1000, str(t_P[idx]))
                    elif xtitle == "frame":
                        text(P[idx, 0] * 1000, P[idx, 2] * 1000, str(frame[idx]))

            if 0:
                plot(Psmooth[:, 0] * 1000, Psmooth[:, 2] * 1000, "b-")

            if 1:
                fx0 = P[1:-1, 0] * 1000
                fz0 = P[1:-1, 2] * 1000

                for force_type, force_color in [
                    (resultant, (1, 0, 0, 1)),
                    (body_drag_world, (0, 1, 0, 1)),
                ]:
                    if force_type is None:
                        continue
                    fx1 = fx0 + force_type[:, 0] * force_scaling
                    fz1 = fz0 + force_type[:, 2] * force_scaling

                    segments = []
                    for i in range(len(fx0)):
                        segment = ((fx0[i], fz0[i]), (fx1[i], fz1[i]))
                        segments.append(segment)
                    collection = LineCollection(
                        segments, colors=[force_color] * len(segments)
                    )
                    gca().add_collection(collection)

            for use_it, data, color in [
                [plot_xy_Qsmooth, Qsmooth, (0, 0, 1, 1)],
                [plot_xy_Qraw, Q, (0, 0, 0, 1)],
            ]:
                if use_it:
                    segments = []
                    for i in range(len(P)):
                        pi = P[i]
                        qi = data[i]
                        Pqi = quat_to_orient(qi)
                        segment = (
                            (pi[0] * 1000, pi[2] * 1000),  # x1  # y1
                            (
                                pi[0] * 1000 - Pqi[0] * 2,  # x2
                                pi[2] * 1000 - Pqi[2] * 2,
                            ),
                        )  # y2
                        segments.append(segment)

                    collection = LineCollection(
                        segments, colors=[color] * len(segments)
                    )
                    gca().add_collection(collection)
            xlabel("x (mm)")
            ylabel("z (mm)")
            ##        t=text( 0, 1.0, '<- wind (0.4 m/sec)',
            ###        t=text( 0.6, .2, '<- wind (0.4 m/sec)',
            ##                transform = gca().transAxes,
            ##                horizontalalignment = 'left',
            ##                verticalalignment = 'top',
            ##                )

            if show_grid:
                grid()
        finally:
            ion()

    elif plot_xy_air:
        print("plot_xy_air")
        axes([0.1, 0.1, 0.8, 0.8])
        title("position (wind frame)")

        xairvel = 0.4  # m/sec
        xairvel = xairvel / 100.0  # 100 positions/sec

        Pair = P.copy()
        for i in range(len(Pair)):
            Pair[i, 0] = P[i, 0] + xairvel * i
        plot(Pair[:, 0] * 1000, Pair[:, 1] * 1000, "ko", mfc=(1, 1, 1), markersize=2)

        Psmooth_air = Psmooth.copy()
        for i in range(len(Psmooth_air)):
            Psmooth_air[i, 0] = Psmooth[i, 0] + xairvel * i

        for idx in range(len(t_P)):
            if idx % 10 == 0:
                if xtitle == "time":
                    text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(t_P[idx]))
                elif xtitle == "frame":
                    text(P[idx, 0] * 1000, P[idx, 1] * 1000, str(frame[idx]))

        for use_it, data, color in [
            [plot_xy_Qsmooth, Qsmooth, (0, 0, 1, 1)],
            [plot_xy_Qraw, Q, (0, 0, 0, 1)],
        ]:
            if use_it:
                segments = []
                for i in range(len(Pair)):
                    pi = Pair[i]
                    qi = data[i]
                    Pqi = quat_to_orient(qi)
                    segment = (
                        (pi[0] * 1000, pi[1] * 1000),  # x1  # y1
                        (pi[0] * 1000 - Pqi[0] * 2, pi[1] * 1000 - Pqi[1] * 2),  # x2
                    )  # y2
                    segments.append(segment)

                collection = LineCollection(segments, colors=[color] * len(segments))
                gca().add_collection(collection)
        xlabel("x (mm)")
        ylabel("y (mm)")
        grid()

    elif plot_accel:
        print("plot_accel")
        subplot(3, 1, 1)
        plot(t_d2Pdt2, d2Pdt2[:, 0], "r-")
        grid()

        subplot(3, 1, 2)
        plot(t_d2Pdt2, d2Pdt2[:, 1], "g-")
        grid()

        subplot(3, 1, 3)
        plot(t_d2Pdt2, d2Pdt2[:, 2], "b-")
        grid()
    elif plot_smooth_pos_and_vel:
        print("plot_smooth_pos_and_vel")
        linewidth = 1.5
        subplot(3, 1, 1)
        title("Global reference frame, ground speed")
        raw_lines = plot(t_P, P[:, 0], "rx", t_P, P[:, 1], "gx", t_P, P[:, 2], "bx")
        smooth_lines = plot(
            t_P, Psmooth[:, 0], "r-", t_P, Psmooth[:, 1], "g-", t_P, Psmooth[:, 2], "b-"
        )
        legend(smooth_lines, ("X", "Y", "Z"), 2)
        setp(smooth_lines, "linewidth", linewidth)
        ylabel("Position (m)")
        grid()

        subplot(3, 1, 2)
        raw_lines = plot(
            t_dPdt, dPdt[:, 0], "rx", t_dPdt, dPdt[:, 1], "gx", t_dPdt, dPdt[:, 2], "bx"
        )
        smooth_lines = plot(
            t_dPdt,
            dPdt_smooth[:, 0],
            "r-",
            t_dPdt,
            dPdt_smooth[:, 1],
            "g-",
            t_dPdt,
            dPdt_smooth[:, 2],
            "b-",
        )
        setp(smooth_lines, "linewidth", linewidth)
        ylabel("Velocity (m/sec)")
        grid()

        subplot(3, 1, 3)
        raw_lines = plot(
            t_d2Pdt2,
            d2Pdt2[:, 0],
            "rx",
            t_d2Pdt2,
            d2Pdt2[:, 1],
            "gx",
            t_d2Pdt2,
            d2Pdt2[:, 2],
            "bx",
        )
        smooth_lines = plot(
            t_d2Pdt2,
            d2Pdt2_smooth[:, 0],
            "r-",
            t_d2Pdt2,
            d2Pdt2_smooth[:, 1],
            "g-",
            t_d2Pdt2,
            d2Pdt2_smooth[:, 2],
            "b-",
        )
        setp(smooth_lines, "linewidth", linewidth)
        ylabel("Acceleration (m/sec/sec)")
        xlabel("Time (sec)")
        grid()
    elif plot_Q:
        print("plot_Q")
        linewidth = 1.5
        ax1 = subplot(3, 1, 1)
        title("quaternions in R4")
        raw_lines = plot(t_P, Q.w, "kx", t_P, Q.x, "rx", t_P, Q.y, "gx", t_P, Q.z, "bx")
        if do_smooth_quats:
            smooth_lines = plot(
                t_P,
                Qsmooth.w,
                "k",
                t_P,
                Qsmooth.x,
                "r",
                t_P,
                Qsmooth.y,
                "g",
                t_P,
                Qsmooth.z,
                "b",
            )
            setp(smooth_lines, "linewidth", linewidth)
            legend(smooth_lines, ["w", "x", "y", "z"])
        else:
            legend(raw_lines, ["w", "x", "y", "z"])
        ylabel("orientation")
        grid()

        ax2 = subplot(3, 1, 2, sharex=ax1)
        if 0:
            # only plot raw if no smooth (derivatives of raw data are very noisy)
            raw_lines = plot(
                t_omega,
                omega.w,
                "kx",
                t_omega,
                omega.x,
                "rx",
                t_omega,
                omega.y,
                "gx",
                t_omega,
                omega.z,
                "bx",
            )
        if do_smooth_quats:
            rad2deg = 180 / math.pi
            mag_omega = nx.array([abs(q) for q in omega_smooth]) * rad2deg
            ##            print 't_omega.shape',t_omega.shape
            ##            print 'mag_omega.shape',mag_omega.shape
            ##            smooth_lines = plot( t_omega, mag_omega)
            ##            smooth_lines = plot( t_omega, nx.arctan2(omega_smooth.y, omega_smooth.x)*rad2deg, 'k-')
            smooth_lines = plot(
                t_omega,
                omega_smooth.w,
                "k",
                t_omega,
                omega_smooth.x,
                "r",
                t_omega,
                omega_smooth.y,
                "g",
                t_omega,
                omega_smooth.z,
                "b",
            )
            setp(smooth_lines, "linewidth", linewidth)
            legend(smooth_lines, ["w", "x", "y", "z"])
        else:
            legend(raw_lines, ["w", "x", "y", "z"])
        ylabel("omega")
        xlabel("Time (sec)")
        grid()

        ax3 = subplot(3, 1, 3, sharex=ax1)
        if 0:
            # only plot raw if no smooth (derivatives of raw data are very noisy)
            raw_lines = plot(
                t_omega_dot,
                omega_dot.w,
                "kx",
                t_omega_dot,
                omega_dot.x,
                "rx",
                t_omega_dot,
                omega_dot.y,
                "gx",
                t_omega_dot,
                omega_dot.z,
                "bx",
            )
        if do_smooth_quats:
            smooth_lines = plot(
                t_omega_dot,
                omega_dot_smooth.w,
                "k",
                t_omega_dot,
                omega_dot_smooth.x,
                "r",
                t_omega_dot,
                omega_dot_smooth.y,
                "g",
                t_omega_dot,
                omega_dot_smooth.z,
                "b",
            )
            setp(smooth_lines, "linewidth", linewidth)
            legend(smooth_lines, ["w", "x", "y", "z"])
        else:
            legend(raw_lines, ["w", "x", "y", "z"])
        ylabel("omega dot")
        xlabel("Time (sec)")
        grid()

    elif plot_body_angular_vel:
        print("plot_body_angular_vel")
        rad2deg = 180 / math.pi
        linewidth = 1.5
        smooth = 1
        rad2deg = 180 / math.pi
        fontsize = 10

        # useQsmooth = Qsmooth
        useQsmooth = Qsmooth_zero_roll
        if useQsmooth == Qsmooth_zero_roll:
            # theses are already present in case useQsmooth == Qsmooth
            omega_smooth = (useQsmooth[:-1].inverse() * useQsmooth[1:]).log() / delta_t
            omega_smooth_body = rotate_velocity_by_orientation(
                omega_smooth, Qsmooth_zero_roll[:-1]
            )

        use_roll_guess = True

        if use_roll_guess:
            omega_smooth2 = (
                Qsmooth_roll_guess[:-1].inverse() * Qsmooth_roll_guess[1:]
            ).log() / delta_t

            omega_dot_smooth2 = (
                (Qsmooth_roll_guess[1:-1].inverse() * Qsmooth_roll_guess[2:]).log()
                - (Qsmooth_roll_guess[:-2].inverse() * Qsmooth_roll_guess[1:-1]).log()
            ) / (delta_t ** 2)

        ax1 = subplot(3, 1, 1)  ##########################
        title("angles and angular velocities")
        if use_roll_guess:
            euler_smooth2 = (
                nx.array([quat_to_euler(q) for q in Qsmooth_roll_guess]) * rad2deg
            )
        euler_smooth = nx.array([quat_to_euler(q) for q in useQsmooth]) * rad2deg
        euler = nx.array([quat_to_euler(q) for q in Q]) * rad2deg
        yaw = euler[:, 0]
        pitch = euler[:, 1]
        roll = euler[:, 2]
        yaw_smooth = euler_smooth[:, 0]
        pitch_smooth = euler_smooth[:, 1]
        roll_smooth = euler_smooth[:, 2]
        if use_roll_guess:
            roll_smooth2 = euler_smooth2[:, 2]
        if xtitle == "time":
            xdata = t_P
        elif xtitle == "frame":
            xdata = t_P * 100 + start_frame
        lines = plot(xdata, yaw, "r-", xdata, pitch, "g-", xdata, roll, "b-")
        lines_smooth = plot(
            xdata, yaw_smooth, "r-", xdata, pitch_smooth, "g-", xdata, roll_smooth, "b-"
        )
        if use_roll_guess:
            lines_smooth2 = plot(xdata, roll_smooth2, "y-")
        if xtitle == "time":
            plot(t_bad, [0.0] * len(t_bad), "ko")
        elif xtitle == "frame":
            plot(frame_bad, [0.0] * len(frame_bad), "ko")
        setp(lines_smooth, "lw", linewidth)
        if use_roll_guess:
            setp(lines_smooth2, "lw", linewidth)
        legend(lines, ["heading", "pitch (body)", "roll"])
        ylabel("angular position (global)\n(deg)")
        setp(gca().yaxis.label, "size", fontsize)
        # setp(gca(),'ylim',(-15,75))
        grid()

        plot_mag = True
        plot_roll = True
        subplot(3, 1, 2, sharex=ax1)  ##########################
        if xtitle == "time":
            xdata = t_omega
        elif xtitle == "frame":
            xdata = t_omega * 100 + start_frame
        if 0:
            if plot_mag:
                mag_omega = nx.array([abs(q) for q in omega]) * rad2deg
                args = [xdata, mag_omega, "k-"]
                line_titles = ["mag"]
            else:
                args = []
                line_titles = []
            args.extend(
                [xdata, omega.z * rad2deg, "r-", xdata, omega.y * rad2deg, "g-"]
            )
            line_titles.extend(["heading", "pitch (body)"])
            if plot_roll:
                args.extend([xdata, omega.x * rad2deg, "b-"])
                line_titles.extend(["roll"])
            lines = plot(*args)
        else:
            lines = []
            line_titles = []

        if plot_mag:
            mag_omega = nx.array([abs(q) for q in omega_smooth]) * rad2deg
            args = [xdata, mag_omega, "k-"]
            if use_roll_guess:
                mag_omega2 = nx.array([abs(q) for q in omega_smooth2]) * rad2deg
                args.extend([xdata, mag_omega2, "g:"])
        else:
            args = []
        args.extend(
            [
                xdata,
                omega_smooth.z * rad2deg,
                "r-",
                xdata,
                omega_smooth.y * rad2deg,
                "g-",
            ]
        )
        if plot_roll:
            args.extend([xdata, omega_smooth.x * rad2deg, "b-"])
        if use_roll_guess:
            args.extend(
                [
                    xdata,
                    omega_smooth2.z * rad2deg,
                    "c-",
                    xdata,
                    omega_smooth2.y * rad2deg,
                    "m-",
                ]
            )
            if plot_roll:
                args.extend([xdata, omega_smooth2.x * rad2deg, "y-"])

        lines_smooth = plot(*args)
        setp(lines_smooth, "lw", linewidth)
        if len(lines):
            legend(lines, line_titles)
        ylabel("angular velocity\nglobal frame (deg/sec)")
        setp(gca().yaxis.label, "size", fontsize)
        # setp(gca(),'ylim',[-750,600])
        grid()

        subplot(3, 1, 3, sharex=ax1)  ##########################
        if 0:
            if plot_mag:
                mag_omega_body = nx.array([abs(q) for q in omega_body]) * rad2deg
                args = [xdata, mag_omega_body, "k-"]
                line_titles = ["mag"]
            else:
                args = []
                line_titles = []
            args.extend(
                [
                    xdata,
                    omega_body.z * rad2deg,
                    "r-",
                    xdata,
                    omega_body.y * rad2deg,
                    "g-",
                ]
            )
            line_titles.extend(["yaw", "pitch"])
            if plot_roll:
                args.extend([xdata, omega_body.x * rad2deg, "b-"])
                line_titles.extend(["roll"])
            lines = plot(*args)
            legend(lines, line_titles)

        if use_roll_guess:
            omega_smooth2_body = rotate_velocity_by_orientation(
                omega_smooth2, Qsmooth_roll_guess[:-1]
            )

        if plot_mag:
            mag_omega_body = nx.array([abs(q) for q in omega_smooth_body]) * rad2deg
            args = [xdata, mag_omega_body, "k-"]
            line_titles = ["mag"]

            if use_roll_guess:
                mag_omega2_body = (
                    nx.array([abs(q) for q in omega_smooth2_body]) * rad2deg
                )
                args.extend([xdata, mag_omega2_body, "g:"])
                line_titles.extend(["mag (roll corrected)"])
        else:
            args = []
            line_titles = []

        args.extend(
            [
                xdata,
                omega_smooth_body.z * rad2deg,
                "r-",
                xdata,
                omega_smooth_body.y * rad2deg,
                "g-",
            ]
        )
        line_titles.extend(["yaw", "pitch"])
        if plot_roll:
            args.extend([xdata, omega_smooth_body.x * rad2deg, "b-"])
            line_titles.extend(["roll"])

        if use_roll_guess:
            args.extend(
                [
                    xdata,
                    omega_smooth2_body.z * rad2deg,
                    "c-",
                    xdata,
                    omega_smooth2_body.y * rad2deg,
                    "m-",
                ]
            )
            line_titles.extend(["yaw (roll corrected)", "pitch (roll corrected)"])
            if plot_roll:
                args.extend([xdata, omega_smooth2_body.x * rad2deg, "y-"])
                line_titles.extend(["roll (roll corrected)"])

        lines_smooth = plot(*args)
        setp(lines_smooth, "lw", linewidth)
        ylabel("angular velocity\nbody frame (deg/sec)")
        setp(gca().yaxis.label, "size", fontsize)
        legend(lines_smooth, line_titles)

        # setp(gca(),'ylim',[-500,500])
        if xtitle == "time":
            xlabel("time (sec)")
        elif xtitle == "frame":
            xlabel("frame")
        grid()
    elif plot_body_angular_vel2:  # angular vels w and w/o roll guess
        print("plot_body_angular_vel2")
        rad2deg = 180 / math.pi
        linewidth = 1.5
        smooth = 1
        rad2deg = 180 / math.pi
        fontsize = 10

        ##        useQsmooth = Qsmooth
        ##        useQsmooth = Qsmooth_zero_roll

        if 0:
            omega_smooth_body = rotate_velocity_by_orientation(
                omega_smooth, Qsmooth[:-1]
            )
        else:
            omega_smooth_body = rotate_velocity_by_orientation(
                omega_smooth, Qsmooth_zero_roll[:-1]
            )

        omega_smooth2 = (
            Qsmooth_roll_guess[:-1].inverse() * Qsmooth_roll_guess[1:]
        ).log() / delta_t
        omega_smooth2_body = rotate_velocity_by_orientation(
            omega_smooth2, Qsmooth_roll_guess[:-1]
        )

        ##        omega_dot_smooth2 = ((Qsmooth_roll_guess[1:-1].inverse()*Qsmooth_roll_guess[2:]).log() -
        ##                             (Qsmooth_roll_guess[:-2].inverse()*Qsmooth_roll_guess[1:-1]).log()) / (delta_t**2)
        ##        euler_smooth2 = nx.array([quat_to_euler(q) for q in Qsmooth_roll_guess])*rad2deg
        ##        euler_smooth = nx.array([quat_to_euler(q) for q in useQsmooth])*rad2deg
        ##        euler = nx.array([quat_to_euler(q) for q in Q])*rad2deg

        ##        yaw = euler[:,0]
        ##        pitch = euler[:,1]
        ##        roll = euler[:,2]

        ##        yaw_smooth = euler_smooth[:,0]
        ##        pitch_smooth = euler_smooth[:,1]
        ##        roll_smooth = euler_smooth[:,2]

        ##        roll_smooth2 = euler_smooth2[:,2]

        if xtitle == "time":
            xdata = t_omega
        elif xtitle == "frame":
            xdata = t_omega * 100 + start_frame

        ax1 = subplot(3, 1, 1)  ##########################

        args = []
        line_titles = []

        args.extend(
            [
                xdata,
                omega_smooth_body.z * rad2deg,
                "g-",
                xdata,
                omega_smooth2_body.z * rad2deg,
                "r-",
            ]
        )
        line_titles.extend(["yaw (roll=0)", "yaw (w roll model)"])
        lines_smooth = plot(*args)
        setp(lines_smooth, "lw", linewidth)
        ylabel("yaw angular velocity\nbody frame (deg/sec)")
        setp(gca().yaxis.label, "size", fontsize)
        legend(lines_smooth, line_titles)

        grid()

        subplot(3, 1, 2, sharex=ax1, sharey=ax1)  ##########################

        args = []
        line_titles = []

        args.extend(
            [
                xdata,
                omega_smooth_body.y * rad2deg,
                "g-",
                xdata,
                omega_smooth2_body.y * rad2deg,
                "r-",
            ]
        )
        line_titles.extend(["pitch (roll=0)", "pitch (w roll model)"])
        lines_smooth = plot(*args)
        setp(lines_smooth, "lw", linewidth)
        ylabel("pitch angular velocity\nbody frame (deg/sec)")
        setp(gca().yaxis.label, "size", fontsize)
        legend(lines_smooth, line_titles)

        grid()

        subplot(3, 1, 3, sharex=ax1, sharey=ax1)  ##########################

        args = []
        line_titles = []

        args.extend(
            [
                xdata,
                omega_smooth_body.x * rad2deg,
                "g-",
                xdata,
                omega_smooth2_body.x * rad2deg,
                "r-",
            ]
        )
        line_titles.extend(["roll (roll=0)", "roll (w roll model)"])
        lines_smooth = plot(*args)
        setp(lines_smooth, "lw", linewidth)
        ylabel("roll angular velocity\nbody frame (deg/sec)")
        setp(gca().yaxis.label, "size", fontsize)
        legend(lines_smooth, line_titles)

        if xtitle == "time":
            xlabel("time (sec)")
        elif xtitle == "frame":
            xlabel("frame")
        grid()

        ####################

        if 0:
            xlim = 0.09, 0.86
            ylim = -1115, 2270
            print("setting xlim to", xlim)
            print("setting ylim to", ylim)
            setp(gca(), "xlim", xlim)
            setp(gca(), "ylim", ylim)

    elif plot_error_angles:
        print("plot_error_angles")
        # plot
        linewidth = 1.5
        subplot(2, 1, 1)
        title("orientation - course direction = error in earlier work")
        plot(t_heading_err, heading_err, "b-")
        heading_lines = plot(
            t_heading_err_smooth, heading_err_smooth, "b-", lw=linewidth
        )
        plot(t_pitch_body_err, pitch_body_err, "r-")
        pitch_lines = plot(
            t_pitch_body_err_smooth, pitch_body_err_smooth, "r-", lw=linewidth
        )
        legend((heading_lines[0], pitch_lines[0]), ("heading", "pitch (body)"))
        ylabel("angle (deg)")
        grid()

        subplot(2, 1, 2)
        plot(t_d_heading_err_dt, d_heading_err_dt, "b-")
        plot(t_d_heading_err_smooth_dt, d_heading_err_smooth_dt, "b-", lw=linewidth)
        plot(t_d_pitch_body_err_dt, d_pitch_body_err_dt, "r-")
        plot(
            t_d_pitch_body_err_smooth_dt, d_pitch_body_err_smooth_dt, "r-", lw=linewidth
        )
        setp(gca(), "ylim", [-2000, 2000])
        ylabel("anglular velocity (deg/sec)")
        xlabel("time (sec)")
        grid()
    elif plot_body_ground_V:
        print("plot_body_ground_V")
        linewidth = 1.5
        subplot(4, 1, 1)
        title("groundspeed (body frame)")
        plot(t_dPdt, body_ground_V.x, "kx")
        plot(t_dPdt, body_ground_V_smooth.x, "b-", lw=linewidth)
        ylabel("forward\n(m/sec)")
        grid()
        subplot(4, 1, 2)
        plot(t_dPdt, body_ground_V.y, "kx")
        plot(t_dPdt, body_ground_V_smooth.y, "b-", lw=linewidth)
        ylabel("sideways\n(m/sec)")
        grid()
        subplot(4, 1, 3)
        plot(t_dPdt, body_ground_V.z, "kx")
        plot(t_dPdt, body_ground_V_smooth.z, "b-", lw=linewidth)
        ylabel("upward\n(m/sec)")
        grid()
        subplot(4, 1, 4)
        raw_norm = nx.sqrt(
            body_ground_V.x ** 2 + body_ground_V.y ** 2 + body_ground_V.z ** 2
        )
        plot(t_dPdt, raw_norm, "kx")
        smooth_norm = nx.sqrt(
            body_ground_V_smooth.x ** 2
            + body_ground_V_smooth.y ** 2
            + body_ground_V_smooth.z ** 2
        )
        plot(t_dPdt, smooth_norm, "b-", lw=linewidth)
        ylabel("|V|\n(m/sec)")
        xlabel("time (sec)")
        grid()
    elif plot_body_air_V:
        print("plot_body_air_V")
        linewidth = 1.5
        subplot(4, 1, 1)
        title("airspeed (body frame)")
        plot(t_dPdt, body_air_V.x, "kx")
        plot(t_dPdt, body_air_V_smooth.x, "b-", lw=linewidth)
        ylabel("forward\n(m/sec)")
        subplot(4, 1, 2)
        plot(t_dPdt, body_air_V.y, "kx")
        plot(t_dPdt, body_air_V_smooth.y, "b-", lw=linewidth)
        ylabel("sideways\n(m/sec)")
        subplot(4, 1, 3)
        plot(t_dPdt, body_air_V.z, "kx")
        plot(t_dPdt, body_air_V_smooth.z, "b-", lw=linewidth)
        ylabel("upward\n(m/sec)")
        subplot(4, 1, 4)
        raw_norm = nx.sqrt(body_air_V.x ** 2 + body_air_V.y ** 2 + body_air_V.z ** 2)
        plot(t_dPdt, raw_norm, "kx")
        smooth_norm = nx.sqrt(
            body_air_V_smooth.x ** 2
            + body_air_V_smooth.y ** 2
            + body_air_V_smooth.z ** 2
        )
        plot(t_dPdt, smooth_norm, "b-", lw=linewidth)
        ylabel("|V|\n(m/sec)")
        xlabel("time (sec)")
    elif plot_forces:
        print("plot_forces")

        ax1 = subplot(3, 1, 1)
        title("predicted aerodynamic forces on body")
        lines = plot(t_forces, F_P, "b-", t_forces, F_N, "r-", lw=1.5)
        ylabel("force (N)")
        legend(lines, ["parallel", "normal"])
        ylim = get(gca(), "ylim")
        setp(gca(), "ylim", [0, ylim[1]])
        text(
            0.1,
            0.9,
            "Force to keep 1 mg aloft: %.1e" % aloft_force,
            transform=gca().transAxes,
        )

        ax2 = subplot(3, 1, 2, sharex=ax1)
        aattack_lines = ax2.plot(t_dPdt, aattack * rad2deg, lw=1.5)
        ylabel("alpha (deg)")
        ax2.yaxis.tick_left()

        ax3 = twinx(ax2)
        vel_mag = nx.sqrt(nx.sum(dPdt_smooth_air ** 2, axis=1))
        vel_lines = ax3.plot(
            t_dPdt, nx.sqrt(nx.sum(dPdt_smooth_air ** 2, axis=1)), "k", lw=1.5
        )
        ax3.yaxis.tick_right()
        legend((aattack_lines[0], vel_lines[0]), ("alpha", "|V|"))

        ax4 = subplot(3, 1, 3, sharex=ax1)
        xlabel("time (sec)")

    print("returning...")
    return outputs


def two_posts():
    fd = open("fXl-fixed.pkl", "rb")
    fXl = pickle.load(fd)
    fd.close()
    results = fXl
    if 1:
        start = 59647
        stop = start + 100 * 2
        frames, psmooth, qsmooth = do_it(
            results,
            start_frame=start,
            stop_frame=stop,
            interp_OK=True,
            return_frame_numbers=True,
            beta=1e-3,
            lambda2=1e-13,
            # percent_error_eps_quats = 2.0,
            percent_error_eps_quats=0.1,
            # do_smooth_position=False,
            do_smooth_position=True,
            do_smooth_quats=True,
            ##                                     plot_xy_Psmooth=True,
            ##                                     plot_xy_Qsmooth=True,
            ##                                     #plot_xy_Praw=False,
            ##                                     plot_xy_Qraw=False,
            plot_xy=False,
            plot_Q=True,
            plot_body_angular_vel2=True,
        )
        # result_browser.save_smooth_data(results,frames,psmooth,qsmooth)

        qsmooth = qsmooth.to_numpy()

        results = {}
        results["frames"] = frames
        results["psmooth"] = psmooth
        results["qsmooth"] = qsmooth
        scipy.io.savemat("smoothed", results)
        show()


def check_icb():
    import result_browser

    filename = "DATA20060315_170142.h5"
    results = result_browser.get_results(filename, mode="r+")
    fstart = 993900
    fend = 994040
    if 1:
        frames, psmooth, qsmooth = do_it(
            results,
            start_frame=fstart,
            stop_frame=fend,
            interp_OK=True,
            return_frame_numbers=True,
            beta=1.0,
            lambda2=1e-13,
            percent_error_eps_quats=2.0,
            do_smooth_position=True,
            do_smooth_quats=True,
            ##                                     plot_xy_Psmooth=True,
            ##                                     plot_xy_Qsmooth=True,
            ##                                     #plot_xy_Praw=False,
            ##                                     plot_xy_Qraw=False,
            plot_xy=False,
            plot_body_angular_vel2=True,
        )
        result_browser.save_smooth_data(results, frames, psmooth, qsmooth)

        results = {}
        results["frames"] = frames
        results["psmooth"] = psmooth
        results["qsmooth"] = qsmooth
        scipy.io.savemat("smoothed", results)
        show()


def calculate_roll_and_save(results, start_frame, stop_frame, **kwargs):
    import result_browser

    if 1:
        frames, psmooth, qsmooth = do_it(
            results,
            start_frame=start_frame,
            stop_frame=stop_frame,
            interp_OK=True,
            return_frame_numbers=True,
            do_smooth_position=True,
            do_smooth_quats=True,
            **kwargs
        )

        result_browser.save_smooth_data(results, frames, psmooth, qsmooth)
    if 1:
        # linear drag model
        frames, psmooth, qlin = do_it(
            results,
            start_frame=start_frame,
            stop_frame=stop_frame,
            interp_OK=True,
            return_frame_numbers=True,
            return_smooth_position=True,
            drag_model_for_roll="linear",
            return_roll_qsmooth=True,
            **kwargs
        )
        result_browser.save_smooth_data(
            results, frames, psmooth, qlin, "smooth_data_roll_fixed_lin"
        )

        # v2 drag model
        frames, psmooth, qv2 = do_it(
            results,
            start_frame=start_frame,
            stop_frame=stop_frame,
            interp_OK=True,
            return_frame_numbers=True,
            return_smooth_position=True,
            drag_model_for_roll="v^2",
            return_roll_qsmooth=True,
            **kwargs
        )
        result_browser.save_smooth_data(
            results, frames, psmooth, qv2, "smooth_data_roll_fixed_v2"
        )

    if 1:
        outputs = do_it(
            results,
            start_frame=start_frame,
            stop_frame=stop_frame,
            interp_OK=True,
            return_resultant_forces=True,
            **kwargs
        )
        frames, resultants = outputs[0]
        result_browser.save_timed_forces("resultants", results, frames, resultants)
        print("saved resultants table")


def delete_calculated_roll(results):
    del results.root.smooth_data
    del results.root.smooth_data_roll_fixed_v2
    del results.root.smooth_data_roll_fixed_lin
    del results.root.resultants


if __name__ == "__main__":
    two_posts()
