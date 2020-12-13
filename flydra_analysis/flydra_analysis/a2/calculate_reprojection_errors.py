from __future__ import print_function
import argparse
import numpy as np
import pandas
import collections
import progressbar
import os
import warnings

import flydra_analysis.a2.core_analysis as core_analysis
import flydra_analysis.analysis.result_utils as result_utils
import flydra_core.reconstruct as reconstruct


class StringWidget(progressbar.Widget):
    def set_string(self, ts):
        self.ts = ts

    def update(self, pbar):
        if hasattr(self, "ts"):
            return self.ts
        else:
            return ""


def calculate_reprojection_errors(
    h5_filename=None,
    output_h5_filename=None,
    kalman_filename=None,
    from_source=None,
    start=None,
    stop=None,
    show_progress=False,
    show_progress_json=False,
):
    assert from_source in ["ML_estimates", "smoothed"]
    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)

    out = {
        "camn": [],
        "frame": [],
        "obj_id": [],
        "dist": [],
        "z": [],
    }

    ca = core_analysis.get_global_CachingAnalyzer()
    with ca.kalman_analysis_context(
        kalman_filename, data2d_fname=h5_filename
    ) as h5_context:
        R = h5_context.get_reconstructor()
        ML_estimates_2d_idxs = h5_context.load_entire_table("ML_estimates_2d_idxs")
        use_obj_ids = h5_context.get_unique_obj_ids()

        extra = h5_context.get_extra_info()

        if from_source == "smoothed":
            dynamic_model_name = extra["dynamic_model_name"]
            if dynamic_model_name.startswith("EKF "):
                dynamic_model_name = dynamic_model_name[4:]

        fps = h5_context.get_fps()
        camn2cam_id, cam_id2camns = h5_context.get_caminfo_dicts()

        # associate framenumbers with timestamps using 2d .h5 file
        data2d = h5_context.load_entire_table("data2d_distorted", from_2d_file=True)
        data2d_idxs = np.arange(len(data2d))
        h5_framenumbers = data2d["frame"]
        h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

        if show_progress:
            string_widget = StringWidget()
            objs_per_sec_widget = progressbar.FileTransferSpeed(unit="obj_ids ")
            widgets = [
                string_widget,
                objs_per_sec_widget,
                progressbar.Percentage(),
                progressbar.Bar(),
                progressbar.ETA(),
            ]
            pbar = progressbar.ProgressBar(
                widgets=widgets, maxval=len(use_obj_ids)
            ).start()

        for obj_id_enum, obj_id in enumerate(use_obj_ids):
            if show_progress:
                string_widget.set_string("[obj_id: % 5d]" % obj_id)
                pbar.update(obj_id_enum)
            if show_progress_json and obj_id_enum % 100 == 0:
                rough_percent_done = float(obj_id_enum) / len(use_obj_ids) * 100.0
                result_utils.do_json_progress(rough_percent_done)

            obj_3d_rows = h5_context.load_dynamics_free_MLE_position(obj_id)

            if from_source == "smoothed":

                smoothed_rows = None
                try:
                    smoothed_rows = h5_context.load_data(
                        obj_id,
                        use_kalman_smoothing=True,
                        dynamic_model_name=dynamic_model_name,
                        frames_per_second=fps,
                    )
                except core_analysis.NotEnoughDataToSmoothError as err:
                    # OK, we don't have data from this obj_id
                    pass
                except core_analysis.DiscontiguousFramesError:
                    pass

            for this_3d_row in obj_3d_rows:
                # iterate over each sample in the current camera
                framenumber = this_3d_row["frame"]
                if start is not None:
                    if not framenumber >= start:
                        continue
                if stop is not None:
                    if not framenumber <= stop:
                        continue
                h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)
                if len(h5_2d_row_idxs) == 0:
                    # At the start, there may be 3d data without 2d data.
                    continue

                if from_source == "ML_estimates":
                    X3d = this_3d_row["x"], this_3d_row["y"], this_3d_row["z"]
                elif from_source == "smoothed":
                    if smoothed_rows is None:
                        X3d = np.nan, np.nan, np.nan
                    else:
                        this_smoothed_rows = smoothed_rows[
                            smoothed_rows["frame"] == framenumber
                        ]
                        assert len(this_smoothed_rows) <= 1
                        if len(this_smoothed_rows) == 0:
                            X3d = np.nan, np.nan, np.nan
                        else:
                            X3d = (
                                this_smoothed_rows["x"][0],
                                this_smoothed_rows["y"][0],
                                this_smoothed_rows["z"][0],
                            )

                # If there was a 3D ML estimate, there must be 2D data.

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
                    try:
                        cam_id = camn2cam_id[camn]
                    except KeyError:
                        warnings.warn("camn %d not found" % (camn,))
                        continue

                    # find 2D point corresponding to object
                    cond = (frame2d["camn"] == camn) & (
                        frame2d["frame_pt_idx"] == camn_pt_no
                    )
                    idxs = np.nonzero(cond)[0]
                    if len(idxs) == 0:
                        # no frame for that camera (start or stop of file)
                        continue
                    elif len(idxs) > 1:
                        print(
                            "MEGA WARNING MULTIPLE 2D POINTS\n",
                            camn,
                            camn_pt_no,
                            "\n\n",
                        )
                        continue

                    idx = idxs[0]

                    frame2d_row = frame2d[idx]
                    x2d_real = frame2d_row["x"], frame2d_row["y"]
                    x2d_reproj = R.find2d(cam_id, X3d, distorted=True)
                    dist = np.sqrt(np.sum((x2d_reproj - x2d_real) ** 2))

                    out["camn"].append(camn)
                    out["frame"].append(framenumber)
                    out["obj_id"].append(obj_id)
                    out["dist"].append(dist)
                    out["z"].append(X3d[2])

    # convert to numpy arrays
    for k in out:
        out[k] = np.array(out[k])
    reprojection = pandas.DataFrame(out)
    del out  # free memory

    # new tables
    camns = []
    cam_ids = []
    for camn in camn2cam_id:
        camns.append(camn)
        cam_ids.append(camn2cam_id[camn])
    cam_table = {
        "camn": np.array(camns),
        "cam_id": np.array(cam_ids),
    }
    cam_df = pandas.DataFrame(cam_table)

    # save to disk
    store = pandas.HDFStore(output_h5_filename)
    store.append("reprojection", reprojection, data_columns=reprojection.columns)
    store.append("cameras", cam_df)
    store.close()
    if show_progress_json:
        result_utils.do_json_progress(100)


def main():
    parser = argparse.ArgumentParser(
        description="calculate per-camera, per-frame, per-object reprojection errors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("h5", type=str, help=".h5 file with data2d_distorted")
    parser.add_argument(
        "-k",
        "--kalman-file",
        help="file with 3D data (if different that file with data2d_distorted)",
    )
    parser.add_argument("--output-h5", type=str, help="filename for output .h5 file")
    parser.add_argument(
        "--start", type=int, default=None, help="frame number to begin analysis on"
    )
    parser.add_argument(
        "--stop", type=int, default=None, help="frame number to end analysis on"
    )
    parser.add_argument(
        "--from-source",
        type=str,
        default="ML_estimates",
        help="source of 3D data for reprojection ('ML_estimates' or 'smoothed')",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="show progress bar on console",
    )
    parser.add_argument(
        "--progress-json",
        dest="show_progress_json",
        action="store_true",
        default=False,
        help="show JSON progress messages",
    )
    args = parser.parse_args()

    if args.kalman_file is None:
        args.kalman_file = args.h5

    if args.output_h5 is None:
        if args.from_source == "ML_estimates":
            args.output_h5 = args.kalman_file + ".repro_errors.h5"
        else:
            args.output_h5 = args.kalman_file + ".%s_repro_errors.h5" % args.from_source

    calculate_reprojection_errors(
        h5_filename=args.h5,
        output_h5_filename=args.output_h5,
        kalman_filename=args.kalman_file,
        from_source=args.from_source,
        start=args.start,
        stop=args.stop,
        show_progress=args.progress,
        show_progress_json=args.show_progress_json,
    )


def print_summarize_file(fname):
    import pandas

    orig_store = pandas.HDFStore(fname, mode="r")
    orig_df = orig_store["reprojection"]
    cam_df = orig_store["cameras"]
    print(fname, "-" * 50)
    print(cam_df)
    for camn, y in orig_df.groupby("camn"):
        print(camn, y.dist.mean())
    orig_store.close()


if __name__ == "__main__":
    main()
