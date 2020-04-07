#!/usr/bin/env python
"""Get Flydra Latency.

Usage:
  get_2D_image_latency_plot.py FILENAME [options]

        Options:
  -h --help     Show this screen.
  --3d          Plot the 3D tracking latency
  --2d          Plot the 2D tracking latency
  --end-idx=N   Only show this many rows [default: 100000]
  --save        Only save the plots, do not open them to screen
"""
from __future__ import print_function
from docopt import docopt

import tables
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

import flydra_analysis.analysis.result_utils as result_utils


def plot_latency(
    fname, do_3d_latency=False, do_2d_latency=False, end_idx=100000, save=False
):
    if do_3d_latency == False and do_2d_latency == False:
        print("hmm, not plotting 3d or 2d data. nothing to do")
        return

    with tables.open_file(fname, mode="r") as h5:
        if do_2d_latency:
            d2d = h5.root.data2d_distorted[:end_idx]
        if do_3d_latency:
            dk = h5.root.kalman_estimates[:end_idx]
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        time_model = result_utils.get_time_model_from_data(h5)

    if do_2d_latency:
        df2d = pd.DataFrame(d2d)
        camn_list = list(df2d["camn"].unique())
        camn_list.sort()

    figs = {}

    if do_3d_latency:
        dfk = pd.DataFrame(dk)

        fig = plt.figure()
        figs["3d"] = fig
        ax = fig.add_subplot(111)
        for obj_id, dfobj in dfk.groupby("obj_id"):
            frame = dfobj["frame"].values
            reconstruct_timestamp = dfobj["timestamp"].values
            trigger_timestamp = time_model.framestamp2timestamp(frame)
            latency = reconstruct_timestamp - trigger_timestamp
            latency[latency < -1e8] = np.nan
            ax.plot(frame, latency, "b.-")
        ax.text(0, 1, "3D reconstruction", va="top", ha="left", transform=ax.transAxes)
        ax.set_xlabel("frame")
        ax.set_ylabel("time (s)")

    if do_2d_latency:
        fig2 = plt.figure()
        figs["2"] = fig2
        axn = None
        fig3 = plt.figure()
        figs["3"] = fig3
        ax3n = None
        fig4 = plt.figure()
        figs["4"] = fig4
        ax4n = None

        for camn, dfcam in df2d.groupby("camn"):
            cam_id = camn2cam_id[camn]
            df0 = dfcam[dfcam["frame_pt_idx"] == 0]
            ts0s = df0["timestamp"].values
            tss = df0["cam_received_timestamp"].values
            frames = df0["frame"].values
            dts = tss - ts0s
            dframes = frames[1:] - frames[:-1]

            axn = fig2.add_subplot(
                len(camn_list), 1, camn_list.index(camn) + 1, sharex=axn
            )
            axn.plot(frames, dts, "r.-", label="camnode latency")
            axn.plot(
                frames[:-1],
                (ts0s[1:] - ts0s[:-1]) / dframes,
                "g.-",
                label="mean inter-frame interval",
            )
            axn.set_xlabel("frame")
            axn.set_ylabel("time (s)")
            axn.text(0, 1, cam_id, va="top", ha="left", transform=axn.transAxes)
            if camn_list.index(camn) == 0:
                axn.legend()

            ax3n = fig3.add_subplot(
                len(camn_list), 1, camn_list.index(camn) + 1, sharex=ax3n
            )
            ax3n.plot(frames, ts0s, "g.-", label="calculated triggerbox timestamp")
            ax3n.set_xlabel("frame")
            ax3n.set_ylabel("time (s)")
            ax3n.text(0, 1, cam_id, va="top", ha="left", transform=ax3n.transAxes)
            if camn_list.index(camn) == 0:
                ax3n.legend()

            ax4n = fig4.add_subplot(
                len(camn_list), 1, camn_list.index(camn) + 1, sharex=ax4n
            )
            ax4n.plot(frames[:-1], ts0s[1:] - ts0s[:-1], "g.-")
            ax4n.set_xlabel("frame")
            ax4n.set_ylabel("inter-frame-interval (s)")
            ax4n.text(0, 1, cam_id, va="top", ha="left", transform=ax4n.transAxes)

    if save:
        for key in figs:
            fig = figs[key]
            fig.savefig("%s-latency-%s.png" % (fname, key))
    else:
        plt.show()


def main():
    args = docopt(__doc__)
    fname = args["FILENAME"]
    plot_latency(
        fname,
        do_3d_latency=args["--3d"],
        do_2d_latency=args["--2d"],
        end_idx=int(args["--end-idx"]),
        save=args["--save"],
    )


if __name__ == "__main__":
    main()
