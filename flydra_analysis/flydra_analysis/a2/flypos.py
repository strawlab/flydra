from __future__ import print_function
import numpy
import numpy as np
import flydra_analysis.a2.core_analysis as core_analysis
import adskalman.adskalman as adskalman

import flydra_core.kalman.dynamic_models
import flydra_analysis.analysis.PQmath as PQmath
import sys, warnings


def find_first_idxs(to_search, searched):
    result = []
    import warnings

    warnings.warn("slow find_first_idxs implementation being used")
    for s in to_search:
        idx = numpy.nonzero(searched == s)[0][0]
        result.append(idx)
    return result


def fuse_obj_ids(
    use_obj_ids, data_file, dynamic_model_name=None, frames_per_second=None
):
    """take multiple obj_id tracks and fuse them into one long trajectory

    Current implementation
    ======================
    Load 'observations' (MLEs of fly positions) across all obj_ids,
    and then do Kalman smoothing across all this, which fills in gaps
    (but ignores single camera views).

    """
    ca = core_analysis.get_global_CachingAnalyzer()

    frames = []
    xs = []
    ys = []
    zs = []
    for obj_id in use_obj_ids:
        # print
        # print obj_id
        kalman_rows = ca.load_dynamics_free_MLE_position(obj_id, data_file)

        if len(frames):
            tmp = kalman_rows["frame"]
            # print tmp

            # (hmm, why do we care? this seems wrong, anyway...)
            # assert tmp[0] > frames[-1][-1] # ensure new frames follow last

            assert numpy.all((tmp[1:] - tmp[:-1]) > 0)  # ensure new frames are ordered

        this_x = kalman_rows["x"]
        full_obs_idx = ~numpy.isnan(this_x)
        if 1:
            warnings.warn(
                "dropping last ML estimate of position in fuse_obj_ids because it is frequently noisy"
            )
            full_obs_idx = np.nonzero(full_obs_idx)[0]
            full_obs_idx = full_obs_idx[:-1]
            if not len(full_obs_idx):
                warnings.warn("no data used for obj_id %d in fuse_obj_ids()" % obj_id)
                continue  # no data
        frames.append(kalman_rows["frame"][full_obs_idx])
        xs.append(kalman_rows["x"][full_obs_idx])
        ys.append(kalman_rows["y"][full_obs_idx])
        zs.append(kalman_rows["z"][full_obs_idx])

    frames = numpy.hstack(frames)
    xs = numpy.hstack(xs)
    ys = numpy.hstack(ys)
    zs = numpy.hstack(zs)
    X = numpy.array([xs, ys, zs])

    if 0:
        import pylab

        X = X.T
        ax = pylab.subplot(3, 1, 1)
        ax.plot(frames, X[:, 0], ".")
        ax = pylab.subplot(3, 1, 2, sharex=ax)
        ax.plot(frames, X[:, 1], ".")
        ax = pylab.subplot(3, 1, 3, sharex=ax)
        ax.plot(frames, X[:, 2], ".")
        pylab.show()
        sys.exit()

    # convert to a single continuous masked array
    frames_all = numpy.arange(frames[0], frames[-1] + 1)

    xs_all = numpy.ma.masked_array(
        data=numpy.ones(frames_all.shape),
        mask=numpy.ones(frames_all.shape, dtype=numpy.bool),
    )
    ys_all = numpy.ma.masked_array(
        data=numpy.ones(frames_all.shape),
        mask=numpy.ones(frames_all.shape, dtype=numpy.bool),
    )
    zs_all = numpy.ma.masked_array(
        data=numpy.ones(frames_all.shape),
        mask=numpy.ones(frames_all.shape, dtype=numpy.bool),
    )

    idxs = frames_all.searchsorted(frames)
    # idxs = find_first_idxs(frames,frames_all)
    xs_all[idxs] = xs
    ys_all[idxs] = ys
    zs_all[idxs] = zs
    orig_data_present = np.zeros(frames_all.shape, dtype=bool)
    orig_data_present[idxs] = True

    # "obs" == "observations" == ML estimates of position without dynamics
    if 0:
        # requires numpy >= r5284
        obs = numpy.ma.masked_array(
            [xs_all, ys_all, zs_all]
        ).T  # Nx3 array for N frames of data
    else:
        obs = numpy.ma.hstack(
            [
                xs_all[:, numpy.newaxis],
                ys_all[:, numpy.newaxis],
                zs_all[:, numpy.newaxis],
            ]
        )

    if 0:
        import pylab

        X = obs
        frames = frames_all
        ax = pylab.subplot(3, 1, 1)
        ax.plot(frames, X[:, 0], ".")
        ax = pylab.subplot(3, 1, 2, sharex=ax)
        ax.plot(frames, X[:, 1], ".")
        ax = pylab.subplot(3, 1, 3, sharex=ax)
        ax.plot(frames, X[:, 2], ".")
        pylab.show()
        sys.exit()

    if 1:
        # convert from masked array to array with nan
        obs[obs.mask] = numpy.nan
        obs = numpy.ma.getdata(obs)

    if 0:
        obs = obs[:100, :]
        print("obs")
        print(obs)

    if 1:
        # now do kalman smoothing across all obj_ids
        model = flydra_core.kalman.dynamic_models.get_kalman_model(
            name=dynamic_model_name, dt=(1.0 / frames_per_second)
        )
        # initial state guess: postion = observation, other parameters = 0
        ss = model["ss"]
        init_x = numpy.zeros((ss,))
        init_x[:3] = obs[0, :]

        P_k1 = numpy.zeros((ss, ss))  # initial state error covariance guess

        for i in range(0, 3):
            P_k1[i, i] = model["initial_position_covariance_estimate"]
        for i in range(3, 6):
            P_k1[i, i] = model.get("initial_velocity_covariance_estimate", 0.0)
        if ss > 6:
            for i in range(6, 9):
                P_k1[i, i] = model.get("initial_acceleration_covariance_estimate", 0.0)

        if not "C" in model:
            raise ValueError('model does not have a linear observation matrix "C".')
        xsmooth, Psmooth = adskalman.kalman_smoother(
            obs, model["A"], model["C"], model["Q"], model["R"], init_x, P_k1,
        )
    X = xsmooth[:, :3]  # kalman estimates of position
    if 0:
        print("X")
        print(X)
        print(dynamic_model_name)
        sys.exit()

    recarray = numpy.rec.fromarrays(
        [frames_all, X[:, 0], X[:, 1], X[:, 2], orig_data_present,],
        names="frame,x,y,z,orig_data_present",
    )
    return recarray


def pos2ori(X, force_pitch_0=False):
    # forward difference
    dX = X[1:, :] - X[:-1, :]
    dXU = (
        dX / numpy.sqrt(numpy.sum(dX ** 2, axis=1))[:, numpy.newaxis]
    )  # make unit vectors
    Q = []
    for U in dXU:
        q = PQmath.orientation_to_quat(U, roll_angle=0, force_pitch_0=force_pitch_0)
        Q.append(q)
    # make same length as X
    Q.append(Q[-1])
    return Q
