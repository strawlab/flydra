"""Get Flydra Image Latency.

Usage:
  flydra_analysis_get_2D_image_latency FILENAME [options]

Options:
  -h --help     Show this screen.
  --end-idx=N   Only show this many rows [default: 100000]
"""
from __future__ import print_function
from __future__ import absolute_import
from docopt import docopt

import tables
import numpy as np
import sys
from . import get_clock_sync
import flydra_analysis.analysis.result_utils as result_utils


def main():
    args = docopt(__doc__)

    filename = args["FILENAME"]
    results = tables.open_file(filename, mode="r")
    time_model = result_utils.get_time_model_from_data(results, debug=False)
    worst_sync_dict = get_clock_sync.get_worst_sync_dict(results)
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)

    hostnames = worst_sync_dict.keys()
    hostnames.sort()

    cam_ids = cam_id2camns.keys()
    cam_ids.sort()

    camns = camn2cam_id.keys()

    # read all data
    end_idx = int(args["--end-idx"])
    d2d = results.root.data2d_distorted[:end_idx]
    cam_info = results.root.cam_info
    results.close()

    if 1:
        for cam_id_enum, cam_id in enumerate(cam_ids):
            camns = cam_id2camns[cam_id]
            if not len(camns) == 1:
                raise NotImplementedError
            camn = camns[0]

            cond1 = cam_info["cam_id"] == cam_id
            assert np.sum(cond1) == 1
            hostname = str(cam_info[cond1]["hostname"][0])

            cond = d2d["camn"] == camn
            mydata = d2d[cond]

            frame = mydata["frame"]

            if 1:
                # Find frames that we never received.  Note that
                # frames we received but with no detections are saved
                # with a single feature point at coords (nan,nan).

                frame_sorted = np.sort(frame)  # We can get out-of-order frames.
                frame_min = frame_sorted[0]
                frame_max = frame_sorted[-1]

                fdiff = frame_sorted[1:] - frame_sorted[:-1]

                skips = (
                    fdiff - 1
                )  # IFI of 1 is not a skip, but normal. (We also have IFIs of 0.)
                skips = skips[skips > 0]
                n_skipped = np.sum(skips)
                n_total = frame_max - frame_min
                frac_skipped = n_skipped / float(n_total)

            trigger_timestamp = time_model.framestamp2timestamp(frame)

            # on camera computer:
            cam_received_timestamp = mydata["cam_received_timestamp"]

            latency_sec = cam_received_timestamp - trigger_timestamp
            median_latency_sec = np.median(latency_sec)
            mean_latency_sec = latency_sec.mean()
            max_latency_sec = np.max(latency_sec)

            err_est = worst_sync_dict.get(hostname, np.nan)
            print(
                "%s (on %s): median: %.1f, mean: %.1f, worst: %.1f (estimate error: %.1f msec). %.2f%% skipped"
                % (
                    cam_id,
                    hostname,
                    median_latency_sec * 1000.0,
                    mean_latency_sec * 1000.0,
                    max_latency_sec * 1000.0,
                    err_est * 1000.0,
                    frac_skipped * 100.0,
                )
            )


if __name__ == "__main__":
    main()
