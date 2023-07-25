from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tables as PT
import numpy
from numpy import nan
from . import result_utils
import flydra_core.reconstruct
import flydra_core._reconstruct_utils as ru


def reconstruct_line_3ds(kresults, recon2, use_obj_id, return_fXl=False):

    data2d = kresults.root.data2d_distorted  # make sure we have 2d data table
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(kresults)
    kobs_2d = kresults.root.ML_estimates_2d_idxs

    obj_ids = kresults.root.kalman_estimates.read(field="obj_id", flavor="numpy")

    if PT.__version__ <= "1.3.3":
        obj_id_find = int(use_obj_id)
    else:
        obj_id_find = use_obj_id

    observation_frame_idxs = kresults.root.ML_estimates.get_where_list(
        kresults.root.ML_estimates.cols.obj_id == obj_id_find, flavor="numpy"
    )

    observation_frames = kresults.root.ML_estimates.read_coordinates(
        observation_frame_idxs, field="frame", flavor="numpy"
    )
    observation_xs = kresults.root.ML_estimates.read_coordinates(
        observation_frame_idxs, field="x", flavor="numpy"
    )
    observation_ys = kresults.root.ML_estimates.read_coordinates(
        observation_frame_idxs, field="y", flavor="numpy"
    )
    observation_zs = kresults.root.ML_estimates.read_coordinates(
        observation_frame_idxs, field="z", flavor="numpy"
    )
    obs_2d_idxs = kresults.root.ML_estimates.read_coordinates(
        observation_frame_idxs, field="obs_2d_idx", flavor="numpy"
    )

    line3d_by_frame = {}
    if return_fXl:
        X_by_frame = {}

    for frame_i, (kframe, obs_2d_idx) in enumerate(
        zip(observation_frames, obs_2d_idxs)
    ):
        ##        if frame_i >= 10:
        ##            break
        if frame_i % 100 == 0:
            print("frame %d of %d" % (frame_i, len(observation_frames)))

        if PT.__version__ <= "1.3.3":
            obs_2d_idx_find = int(obs_2d_idx)
            kframe_find = int(kframe)
        else:
            obs_2d_idx_find = obs_2d_idx
            kframe_find = kframe

        kobs_2d_data = kobs_2d.read(start=obs_2d_idx_find, stop=obs_2d_idx_find + 1)

        assert len(kobs_2d_data) == 1

        kobs_2d_data = kobs_2d_data[0]
        this_camns = kobs_2d_data[0::2]
        this_camn_idxs = kobs_2d_data[1::2]

        # print
        print("kframe", kframe)

        # print ' this_camns',this_camns
        # print ' this_camn_idxs',this_camn_idxs

        done_frame = True

        # Really, I want to iterate through this_camns, but this
        # (iterating through pytables using a condition) will be much
        # faster.

        by_this_camns = {}

        for row in data2d.where(data2d.cols.frame == kframe_find):
            # print '*',row

            camn = row["camn"]
            done = False

            if camn not in this_camns:
                continue

            want_pt_idx = this_camn_idxs[this_camns == camn]

            frame_pt_idx = row["frame_pt_idx"]

            if want_pt_idx != frame_pt_idx:
                continue

            varnames = "x", "y", "eccentricity", "p1", "p2", "p3", "p4", "area", "slope"
            by_this_camns[camn] = {}
            for varname in varnames:
                by_this_camns[camn][varname] = row[varname]

            # by_this_camns[camn] = row['x'], row['y'], row['eccentricity'], row['p1'], row['p2'], row['p3'], row['p4']
            # print '-> usign previous row'
            cam_id = camn2cam_id[camn]

        if len(by_this_camns) < len(this_camns):
            print("WARNING: missing data.")
            continue

        d2 = {}
        for camn, row in by_this_camns.items():
            cam_id = camn2cam_id[camn]
            rx = row["x"]
            ry = row["y"]
            # rx,ry=reconstructor.undistort(cam_id,(rx,ry))
            rx, ry = recon2.undistort(cam_id, (rx, ry))
            d2[cam_id] = (
                rx,
                ry,
                row["area"],
                row["slope"],
                row["eccentricity"],
                row["p1"],
                row["p2"],
                row["p3"],
                row["p4"],
            )

        (
            X,
            line3d,
            cam_ids_used,
            # mean_dist) = ru.find_best_3d(reconstructor,d2)
            mean_dist,
        ) = ru.find_best_3d(recon2, d2)

        try:
            # make sure reconstructed 3D point matches original
            X_orig = numpy.array(
                (
                    observation_xs[frame_i],
                    observation_ys[frame_i],
                    observation_zs[frame_i],
                )
            )
            assert numpy.allclose(X, X_orig)
        except AssertionError as err:
            print("*" * 80)
            print("*" * 80)
            print()
            print("WARNING: 3D positions and original 3D positions not the same!")
            print("X", X)
            print("X_orig", X_orig)
            print()
            print("*" * 80)
            print("*" * 80)

        if return_fXl:
            X_by_frame[int(kframe)] = X
        line3d_by_frame[int(kframe)] = line3d

    if return_fXl:
        frames = X_by_frame.keys()
        frames.sort()
        fXl = []
        for frame in frames:

            X_by_frame[frame]

            line3d_by_frame[frame]

            list(X_by_frame[frame])

            # print 'line3d_by_frame[frame]',line3d_by_frame[frame]

            L = line3d_by_frame[frame]
            if L is None:
                L = (nan, nan, nan, nan, nan, nan)

            fXl.append([frame] + list(X_by_frame[frame]) + list(L))
        fXl = numpy.array(fXl, dtype=numpy.float64)
        return fXl
    else:
        return line3d_by_frame


if __name__ == "__main__":
    import sys

    filename = sys.argv[1]
    use_obj_id = int(sys.argv[2])

    kresults = result_utils.get_results(filename, mode="r+")
    reconstructor = flydra_core.reconstruct.Reconstructor(kresults)
    recon2 = reconstructor.get_scaled(reconstructor.scale_factor)

    fXl = reconstruct_line_3ds(kresults, recon2, use_obj_id, return_fXl=True)
    import pickle

    fd = open("fXl.pkl", mode="wb")
    pickle.dump(fXl, fd)
    fd.close()
