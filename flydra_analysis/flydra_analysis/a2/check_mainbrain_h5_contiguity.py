#!/usr/bin/env python
from __future__ import print_function
import tables
import argparse
import numpy as np
import sys


def check_mainbrain_h5_contiguity(
    filename, slow_but_less_ram=False, shortcircuit=False, verbose=False
):
    failed_obj_ids = []
    if verbose:
        print("opening %r" % filename)
    with tables.open_file(filename, mode="r") as f:
        table = f.root.kalman_estimates

        all_obj_ids = table.cols.obj_id[:]
        obj_ids = np.unique(all_obj_ids)
        if verbose:
            print("checking %d obj_ids" % len(obj_ids))
        if not slow_but_less_ram:
            # faster but more RAM
            all_frames = table.cols.frame[:]
            for obj_id in obj_ids:
                frame = all_frames[all_obj_ids == obj_id]
                diff = frame[1:] - frame[:-1]
                if np.any(diff != 1):
                    failed_obj_ids.append(obj_id)
                    if verbose:
                        print("failed: %d" % obj_id)
                    if shortcircuit:
                        return failed_obj_ids
        else:
            # slower but more memory efficient
            for obj_id in obj_ids:
                cond = all_obj_ids == obj_id
                idxs = np.nonzero(cond)[0]
                frame = table.read_coordinates(idxs, field="frame")
                diff = frame[1:] - frame[:-1]
                if np.any(diff != 1):
                    failed_obj_ids.append(obj_id)
                    if verbose:
                        print("failed: %d" % obj_id)
                    if shortcircuit:
                        return failed_obj_ids

    return failed_obj_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, default=None, help="file to check")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print stuff"
    )
    parser.add_argument(
        "--findall",
        action="store_true",
        default=False,
        help="continue after first hit (only make sense with verbose or output-log)",
    )
    parser.add_argument(
        "--slow-but-less-ram", action="store_true", default=False, help="print stuff"
    )
    parser.add_argument(
        "--no-output-log",
        action="store_true",
        default=False,
        help="do not print a final summary",
    )
    options = parser.parse_args()
    failed_obj_ids = check_mainbrain_h5_contiguity(
        filename=options.file,
        slow_but_less_ram=options.slow_but_less_ram,
        shortcircuit=not options.findall,
        verbose=options.verbose,
    )
    if len(failed_obj_ids):
        if not options.no_output_log:
            print("%s some objects failed: %r" % (options.file, failed_obj_ids))
        sys.exit(1)
    else:
        if not options.no_output_log:
            print("%s no objects failed" % options.file)
        sys.exit(0)


def cls(root="/mnt/strawscience/data/auto_pipeline/raw_archive/by_date"):
    """Generates example command lines amenable to use, for example, with GNU parallel."""
    from itertools import product
    import os.path as op

    for year, month in product(
        (2015, 2014, 2013, 2012), ["%02d" % d for d in xrange(1, 13)]
    ):
        print(
            "find %s -iname '*.mainbrain.h5' "
            "-exec flydra_analysis_check_mainbrain_h5_contiguity --findall {} \; "
            "&>~/%d-%s.log" % (op.join(root, str(year), month), year, month)
        )


if __name__ == "__main__":
    main()
