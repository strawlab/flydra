#!/usr/bin/env python2
import argparse
import os
import numpy as np

import flydra_analysis.analysis.result_utils as result_utils
import flydra_analysis.a2.core_analysis as core_analysis

import pandas as pd


def calculate_skipped_frames(
    h5_filename=None, output_h5_filename=None, kalman_filename=None,
):
    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)

    pre_df = {"obj_id": [], "start_frame": [], "stop_frame": [], "duration": []}
    ca = core_analysis.get_global_CachingAnalyzer()
    with ca.kalman_analysis_context(kalman_filename) as h5_context:
        R = h5_context.get_reconstructor()
        ML_estimates_2d_idxs = h5_context.load_entire_table("ML_estimates_2d_idxs")
        use_obj_ids = h5_context.get_unique_obj_ids()
        for obj_id_enum, obj_id in enumerate(use_obj_ids):
            obj_3d_rows = h5_context.load_dynamics_free_MLE_position(obj_id)
            prev_frame = None
            for this_3d_row in obj_3d_rows:
                # iterate over each sample in the current camera
                framenumber = this_3d_row["frame"]
                if prev_frame is not None:
                    if framenumber - prev_frame > 1:
                        pre_df["obj_id"].append(obj_id)
                        pre_df["start_frame"].append(prev_frame)
                        pre_df["stop_frame"].append(framenumber)
                        pre_df["duration"].append(framenumber - prev_frame)
                prev_frame = framenumber

    df = pd.DataFrame(pre_df)

    # save to disk
    store = pd.HDFStore(output_h5_filename)
    store.append("skipped_info", df)
    store.close()


def main():
    parser = argparse.ArgumentParser(
        description="calculate skipped frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("kalman_file", type=str, help="file with 3D data")
    parser.add_argument(
        "--output-h5", type=str, help="filename for output .h5 pandas datastore"
    )
    args = parser.parse_args()

    if args.output_h5 is None:
        args.output_h5 = args.kalman_file + ".skipped_info.h5"

    calculate_skipped_frames(
        kalman_filename=args.kalman_file, output_h5_filename=args.output_h5,
    )
