from __future__ import print_function
import tables
import argparse
import warnings
import numpy as np

from flydra_analysis.a2.tables_tools import open_file_safe
import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils


def get_start_stop(src_h5, name, start, stop):
    print("  finding start and stop row of %s" % name)
    input_node = getattr(src_h5.root, name)
    frames = input_node[:]["frame"]
    if start is not None:
        valid_cond = frames >= start
    else:
        valid_cond = np.ones(frames.shape, dtype=np.bool_)
    if stop is not None:
        valid_cond &= frames <= stop
    valid_idx = np.nonzero(valid_cond)[0]
    start = np.min(valid_idx)
    stop = np.max(valid_idx)
    # this may capture some non-relevant rows...
    return start, stop


def do_data_association_tables(src_h5, output_file, options):
    print(
        'selectively copying relevant rows from "ML_estimates" and '
        '"ML_estimates_2d_idxs"'
    )
    # these are more tricky
    startrow, stoprow = get_start_stop(
        src_h5, "ML_estimates", options.start, options.stop
    )
    orig_ML_est_rows = src_h5.root.ML_estimates[startrow : stoprow + 1]
    new_obs_rows = []
    new_ML_est_rows = orig_ML_est_rows[:]  # copy
    idxcol = orig_ML_est_rows["obs_2d_idx"]
    for i in range(len(orig_ML_est_rows)):
        idxidx2d = int(
            idxcol[i]
        )  # orig_ML_est_row['obs_2d_idx']) # XXX pytables fails with np.uint64?
        idx2d = src_h5.root.ML_estimates_2d_idxs[idxidx2d]
        new_ML_est_rows["obs_2d_idx"][i] = len(new_obs_rows)
        new_obs_rows.append(idx2d)
    # Save data association information.
    ML_estimates_2d_idxs_type = flydra_kalman_utils.ML_estimates_2d_idxs_type
    h5_2d_obs = output_file.create_vlarray(
        output_file.root,
        "ML_estimates_2d_idxs",
        ML_estimates_2d_idxs_type(),  # dtype should match with tro.observations_2d
        "camns and idxs",
    )
    for camns_and_idxs in new_obs_rows:
        h5_2d_obs.append(camns_and_idxs)
    h5_2d_obs.flush()

    # Save ML_estimates.
    ct = output_file.create_table  # shorthand
    root = output_file.root  # shorthand
    FilteredObservations = flydra_kalman_utils.FilteredObservations
    h5data3d_ML_estimates = ct(
        root,
        "ML_estimates",
        FilteredObservations,
        "dynamics-free maximum liklihood estimates",
        expectedrows=len(new_ML_est_rows),
    )
    h5data3d_ML_estimates.append(new_ML_est_rows)
    h5data3d_ML_estimates.flush()


def copy_selective(src_h5, input_node, output_group, options):
    if input_node.name in [
        "data2d_distorted",
        "kalman_estimates",
    ]:

        startrow, stoprow = get_start_stop(
            src_h5, input_node.name, options.start, options.stop
        )
        print(
            "  copying data for frames %s - %s (rows %s - %s)"
            % (options.start, options.stop, startrow, stoprow)
        )
        input_node._f_copy(output_group, start=startrow, stop=stoprow)
    elif input_node.name == "ML_estimates_2d_idxs":
        # ML_estimates and ML_estimates_2d_idxs must be
        # reduced with knowledge of each other. Column obs_2d_idx of
        # ML_estimates must point to row number of
        # corresponding ML_estimates_2d_idxs. Therefore, if
        # rows are dropped from start of ML_estimates_2d_idxs,
        # ML_estimates must be updated.

        if options.start is not None:
            warnings.warn(
                "filtering initial data in table "
                "ML_estimates_2d_idxs not implemented"
            )

        startrow, stoprow = get_start_stop(
            src_h5, "ML_estimates", options.start, options.stop
        )
        my_stoprow = int(src_h5.root.ML_estimates[stoprow]["obs_2d_idx"])
        input_node._f_copy(output_group, start=0, stop=my_stoprow + 1)

    else:
        warnings.warn(
            "no filtering implemented, copying entire " "table %s" % input_node.name
        )
        input_node._f_copy(output_group, recursive=True)


def h5_shorten(input_filename, output_filename, options):
    with open_file_safe(input_filename, mode="r") as h5:
        with open_file_safe(
            output_filename, mode="w", delete_on_error=True
        ) as output_h5:
            if not options.data2d_only:
                do_data_association_tables(h5, output_h5, options)
            for node in h5.root._f_iter_nodes():
                if hasattr(node, "name") and node.name in [
                    "data2d_distorted",
                    "kalman_estimates",
                ]:
                    if options.data2d_only:
                        if node.name != "data2d_distorted":
                            continue
                    print("selectively copying", node)
                    copy_selective(h5, node, output_h5.root, options)
                elif hasattr(node, "name") and node.name in [
                    "ML_estimates",
                    "ML_estimates_2d_idxs",
                ]:
                    continue
                else:
                    # copy everything from source to dest
                    print("copying entire", node)
                    node._f_copy(output_h5.root, recursive=True)


def main():
    parser = argparse.ArgumentParser(
        description="shorten flydra mainbrain HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--data2d-only", action="store_true", default=False)

    options = parser.parse_args()
    input = options.input
    output = options.output
    h5_shorten(input, output, options)


if __name__ == "__main__":
    main()
