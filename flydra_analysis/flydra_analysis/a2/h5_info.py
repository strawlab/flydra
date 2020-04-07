#!/usr/bin/env python
from __future__ import division
from __future__ import absolute_import
import tables
import argparse
import numpy as np
import pprint
from . import core_analysis
from distutils.version import LooseVersion
from . import orientation_ekf_fitter
from flydra_analysis.analysis.result_utils import timestamp2string, get_tz


def get_data2d_distorted_info(filename):
    results = {}
    global_results = {}

    ca = core_analysis.get_global_CachingAnalyzer()
    tmp1, tmp2, tmp3, h5, extra = ca.initial_file_load(filename)
    version = LooseVersion(extra["header"]["flydra_version"])

    tz = get_tz(h5)
    if hasattr(h5.root, "data2d_distorted"):

        data2d = h5.root.data2d_distorted

        timestamps = data2d.read(field="timestamp")
        results["start_timestamp"] = np.min(timestamps)
        results["stop_timestamp"] = np.max(timestamps)

        results["start_timestamp_iso"] = timestamp2string(
            results["start_timestamp"], tz
        )
        results["stop_timestamp_iso"] = timestamp2string(results["stop_timestamp"], tz)

        frames = data2d.read(field="frame")
        results["start_frame"] = np.min(frames)
        results["stop_frame"] = np.max(frames)

        slopes = data2d.read(field="slope")
        results["num_valid_slopes"] = np.sum(~np.isnan(slopes))
        results["num_total_slopes"] = len(slopes)
        results["fraction_valid_slopes"] = np.sum(~np.isnan(slopes)) / len(slopes)

        if version >= LooseVersion("0.4.69+git"):
            default_ibo_value = False
        else:
            default_ibo_value = "unknown"

        results["has_image_based_2d_orientation"] = getattr(
            data2d.attrs, "has_ibo_data", default_ibo_value
        )

        global_results["summary_data2d_distorted"] = results
        global_results["has_image_based_2d_orientation"] = results[
            "has_image_based_2d_orientation"
        ]
    else:
        global_results["summary_data2d_distorted"] = None
        global_results["has_image_based_2d_orientation"] = False

    return global_results


def get_kalman_estimates_info(filename):
    results = {}
    global_results = {}

    ca = core_analysis.get_global_CachingAnalyzer()
    tmp1, unique_object_ids, tmp3, h5, extra = ca.initial_file_load(filename)
    tz = get_tz(h5)

    if hasattr(h5.root, "kalman_estimates"):

        data = h5.root.kalman_estimates

        timestamps = data.read(field="timestamp")
        if not len(timestamps):
            global_results["summary_kalman_estimates"] = None
        else:
            results["start_timestamp"] = np.min(timestamps)
            results["stop_timestamp"] = np.max(timestamps)

            results["start_timestamp_iso"] = timestamp2string(
                results["start_timestamp"], tz
            )
            results["stop_timestamp_iso"] = timestamp2string(
                results["stop_timestamp"], tz
            )

            frames = data.read(field="frame")
            results["start_frame"] = np.min(frames)
            results["stop_frame"] = np.max(frames)

            results["obj_ids"] = unique_object_ids
            global_results["summary_kalman_estimates"] = results
    else:
        global_results["summary_kalman_estimates"] = None

    return global_results


def get_dynamics_free_MLE_info(filename):
    results = {}
    global_results = {}

    ca = core_analysis.get_global_CachingAnalyzer()
    tmp1, unique_object_ids, tmp3, h5, extra = ca.initial_file_load(filename)

    if hasattr(h5.root, "ML_estimates"):

        data = h5.root.ML_estimates

        frames = data.read(field="frame")
        if not len(frames):
            global_results["summary_dynamics_free_MLE"] = None
            global_results["is_3d_orientation_fit"] = False
        else:
            results["start_frame"] = np.min(frames)
            results["stop_frame"] = np.max(frames)

            results["obj_ids"] = unique_object_ids
            results[
                "is_3d_orientation_fit"
            ] = orientation_ekf_fitter.is_orientation_fit(filename)

            global_results["summary_dynamics_free_MLE"] = results
            global_results["is_3d_orientation_fit"] = results["is_3d_orientation_fit"]
    else:
        global_results["summary_dynamics_free_MLE"] = None
        global_results["is_3d_orientation_fit"] = False

    return global_results


def get_all_h5_info(filename):
    ca = core_analysis.get_global_CachingAnalyzer()
    tmp1, tmp2, tmp3, h5, tmp4 = ca.initial_file_load(filename)

    results = {}
    results["filename"] = filename

    results.update(get_data2d_distorted_info(filename))
    results.update(get_kalman_estimates_info(filename))
    results.update(get_dynamics_free_MLE_info(filename))

    h5.close()
    return results


def print_h5_info(filename):
    h5_info_dict = get_all_h5_info(filename)
    pprint.pprint(h5_info_dict)


def cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    for filename in args.filenames:
        print_h5_info(filename)
