from __future__ import print_function
from __future__ import absolute_import
import sys
import tables
import pylab
import numpy
from .result_utils import model_remote_to_local, drift_estimates


def main():
    fname = sys.argv[1]
    results = tables.open_file(fname, mode="r")
    d = drift_estimates(results)
    hostnames = d["hostnames"]
    gain = {}
    offset = {}
    for i, hostname in enumerate(hostnames):
        tgain, toffset = model_remote_to_local(
            d["remote_timestamp"][hostname][::10], d["local_timestamp"][hostname][::10]
        )
        gain[hostname] = tgain
        offset[hostname] = toffset
        print(repr(hostname), tgain, toffset)

    print()

    table = results.root.data2d_distorted

    if 1:
        from . import result_utils

        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
        camn2hostname = {}
        for camn, cam_id in camn2cam_id.items():
            hostname = "_".join(cam_id.split("_")[:-1])
            camn2hostname[camn] = hostname

    cur_frame = None
    cur_ts = []
    ##    start_timestamp = None
    for row in table:
        camn = row["camn"]
        hostname = camn2hostname[camn]
        remote_timestamp = row["timestamp"]
        local_timestamp = remote_timestamp * gain[hostname] + offset[hostname]
        ##        if start_timestamp is None:
        ##            start_timestamp=local_timestamp
        frame = row["frame"]

        ##        print frame,local_timestamp-start_timestamp
        ##        print

        if frame == cur_frame:
            cur_ts.append(local_timestamp)
        else:
            if len(cur_ts) > 2:
                # print last frame
                cur_ts = numpy.array(cur_ts)
                mn = cur_ts.min()
                mx = cur_ts.max()
                spread = mx - mn
                spread_msec = spread * 1e3
                print("% 9d % 6.2f" % (cur_frame, spread_msec))

            # reset for current frame
            cur_ts = [local_timestamp]
            cur_frame = frame


def main_old():
    fname = sys.argv[1]
    results = tables.open_file(fname, mode="r")
    d = drift_estimates(results)
    hostnames = d["hostnames"]
    for i, hostname in enumerate(hostnames):
        gain, offset = model_remote_to_local(
            d["remote_timestamp"][hostname][::10], d["local_timestamp"][hostname][::10]
        )

    if 1:
        ax = None
        for i, hostname in enumerate(hostnames):
            ax = pylab.subplot(len(hostnames), 1, i + 1, sharex=ax)

            clock_diff = (
                d["local_timestamp"][hostname] - d["remote_timestamp"][hostname]
            )

            x = d["local_timestamp"][hostname][::10]
            x = x - x[0]
            y = clock_diff[::10]
            yerr = d["measurement_error"][hostname][::10]

            sys.stdout.flush()

            # pylab.errorbar(x,y, yerr=yerr)
            pylab.plot(x, y, "k-")
            pylab.plot(x, y + yerr, "b-")
            pylab.plot(x, y - yerr, "b-")
            pylab.text(0.05, 0.05, str(hostname), transform=ax.transAxes)

        pylab.show()


if __name__ == "__main__":
    main()
