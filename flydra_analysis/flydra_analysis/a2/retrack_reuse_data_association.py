from __future__ import print_function
import argparse
import numpy as np
import progressbar
import os
import warnings

import tables


import flydra_analysis.a2.core_analysis as core_analysis
import flydra_analysis.analysis.result_utils as result_utils
from flydra_analysis.a2.tables_tools import open_file_safe
import flydra_core.reconstruct as reconstruct
import flydra_core.kalman.dynamic_models as dynamic_models
from flydra_analysis.kalmanize import KalmanSaver
from flydra_core._flydra_tracked_object import TrackedObject
import flydra_analysis.version


class StringWidget(progressbar.Widget):
    def set_string(self, ts):
        self.ts = ts

    def update(self, pbar):
        if hasattr(self, "ts"):
            return self.ts
        else:
            return ""


def retrack_reuse_data_association(
    h5_filename=None,
    output_h5_filename=None,
    kalman_filename=None,
    start=None,
    stop=None,
    less_ram=False,
    show_progress=False,
    show_progress_json=False,
):
    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)

    ca = core_analysis.get_global_CachingAnalyzer()
    with ca.kalman_analysis_context(
        kalman_filename, data2d_fname=h5_filename
    ) as h5_context:
        R = h5_context.get_reconstructor()
        if less_ram:
            ML_estimates_2d_idxs = h5_context.get_pytable_node("ML_estimates_2d_idxs")
        else:
            ML_estimates_2d_idxs = h5_context.load_entire_table("ML_estimates_2d_idxs")
        use_obj_ids = h5_context.get_unique_obj_ids()
        extra = h5_context.get_extra_info()
        dt = 1.0 / extra["frames_per_second"]
        dynamic_model_name = extra["dynamic_model_name"]
        kalman_model = dynamic_models.get_kalman_model(name=dynamic_model_name, dt=dt)
        kalman_model["max_frames_skipped"] = 2 ** 62  # close to max i64

        fps = extra["frames_per_second"]
        camn2cam_id, cam_id2camns = h5_context.get_caminfo_dicts()

        parsed = h5_context.read_textlog_header()
        if "trigger_CS3" not in parsed:
            parsed["trigger_CS3"] = "unknown"

        textlog_save_lines = [
            "retrack_reuse_data_association running at %s fps, (top %s, trigger_CS3 %s, flydra_version %s)"
            % (
                str(fps),
                str(parsed.get("top", "unknown")),
                str(parsed["trigger_CS3"]),
                flydra_analysis.version.__version__,
            ),
            "original file: %s" % (kalman_filename,),
            "dynamic model: %s" % (dynamic_model_name,),
            "reconstructor file: %s" % (kalman_filename,),
        ]

        with open_file_safe(
            output_h5_filename,
            mode="w",
            title="tracked Flydra data file",
            delete_on_error=True,
        ) as output_h5:

            h5saver = KalmanSaver(
                output_h5,
                R,
                cam_id2camns=cam_id2camns,
                min_observations_to_save=0,
                textlog_save_lines=textlog_save_lines,
                dynamic_model_name=dynamic_model_name,
                dynamic_model=kalman_model,
            )

            # associate framenumbers with timestamps using 2d .h5 file
            if less_ram:
                data2d = h5_context.get_pytable_node(
                    "data2d_distorted", from_2d_file=True
                )
                h5_framenumbers = data2d.cols.frame[:]
            else:
                data2d = h5_context.load_entire_table(
                    "data2d_distorted", from_2d_file=True
                )
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

                tro = None
                first_frame_per_obj = True
                obj_3d_rows = h5_context.load_dynamics_free_MLE_position(obj_id)
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

                    # If there was a 3D ML estimate, there must be 2D data.

                    frame2d = data2d[h5_2d_row_idxs]

                    obs_2d_idx = this_3d_row["obs_2d_idx"]
                    kobs_2d_data = ML_estimates_2d_idxs[int(obs_2d_idx)]

                    # Parse VLArray.
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]

                    # Now, for each camera viewing this object at this
                    # frame, extract images.
                    observation_camns = []
                    observation_idxs = []
                    data_dict = {}
                    used_camns_and_idxs = []
                    cam_ids_and_points2d = []

                    for camn, frame_pt_idx in zip(this_camns, this_camn_idxs):
                        try:
                            cam_id = camn2cam_id[camn]
                        except KeyError:
                            warnings.warn("camn %d not found" % (camn,))
                            continue

                        # find 2D point corresponding to object
                        cond = (frame2d["camn"] == camn) & (
                            frame2d["frame_pt_idx"] == frame_pt_idx
                        )
                        idxs = np.nonzero(cond)[0]
                        if len(idxs) == 0:
                            # no frame for that camera (start or stop of file)
                            continue
                        elif len(idxs) > 1:
                            print(
                                "MEGA WARNING MULTIPLE 2D POINTS\n",
                                camn,
                                frame_pt_idx,
                                "\n\n",
                            )
                            continue

                        idx = idxs[0]

                        frame2d_row = frame2d[idx]
                        x2d_real = frame2d_row["x"], frame2d_row["y"]
                        pt_undistorted = R.undistort(cam_id, x2d_real)
                        x2d_area = frame2d_row["area"]

                        observation_camns.append(camn)
                        observation_idxs.append(idx)
                        candidate_point_list = []
                        data_dict[camn] = candidate_point_list
                        used_camns_and_idxs.append((camn, frame_pt_idx, None))

                        # with no orientation
                        observed_2d = (pt_undistorted[0], pt_undistorted[1], x2d_area)

                        cam_ids_and_points2d.append((cam_id, observed_2d))

                    if first_frame_per_obj:
                        if len(cam_ids_and_points2d) < 2:
                            warnings.warn(
                                "some 2D data seems to be missing, cannot completely reconstruct"
                            )
                        else:
                            X3d = R.find3d(
                                cam_ids_and_points2d,
                                return_line_coords=False,
                                simulate_via_tracking_dynamic_model=kalman_model,
                            )

                            # first frame
                            tro = TrackedObject(
                                R,
                                obj_id,
                                framenumber,
                                X3d,  # obs0_position
                                None,  # obs0_Lcoords
                                observation_camns,  # first_observation_camns
                                observation_idxs,  # first_observation_idxs
                                kalman_model=kalman_model,
                            )
                            del X3d
                            first_frame_per_obj = False
                    else:
                        tro.calculate_a_posteriori_estimate(
                            framenumber,
                            data_dict,
                            camn2cam_id,
                            skip_data_association=True,
                            original_camns_and_idxs=used_camns_and_idxs,
                            original_cam_ids_and_points2d=cam_ids_and_points2d,
                        )

                # done with all data for this obj_id
                if tro is not None:
                    tro.kill()
                    h5saver.save_tro(tro, force_obj_id=obj_id)
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
    parser.add_argument(
        "--output-h5",
        type=str,
        help="filename for output .h5 file with data2d_distorted",
    )
    parser.add_argument(
        "--start", type=int, default=None, help="frame number to begin analysis on"
    )
    parser.add_argument(
        "--stop", type=int, default=None, help="frame number to end analysis on"
    )
    parser.add_argument(
        "--less-ram",
        action="store_true",
        default=False,
        help="use less RAM (but slower)",
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
        args.output_h5 = args.kalman_file + ".retracked.h5"

    retrack_reuse_data_association(
        h5_filename=args.h5,
        kalman_filename=args.kalman_file,
        output_h5_filename=args.output_h5,
        start=args.start,
        stop=args.stop,
        less_ram=args.less_ram,
        show_progress=args.progress,
        show_progress_json=args.show_progress_json,
    )


if __name__ == "__main__":
    main()
