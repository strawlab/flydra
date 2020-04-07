from __future__ import division
from __future__ import print_function
import pkg_resources
import os
from optparse import OptionParser
import numpy
import numpy as np
import pylab


def doit(caldir):
    filename = os.path.join(caldir, "camera_order.txt")
    cam_ids = open(filename, mode="r").read().split()
    print("cam_ids", cam_ids)

    filename = os.path.join(caldir, "obj_ids_zero_indexed.dat")
    if os.path.exists(filename):
        idx_and_obj_id = np.loadtxt(filename)
    else:
        idx_and_obj_id = None

    filename = os.path.join(caldir, "points.dat")

    points = np.loadtxt(filename)
    print(points.shape)
    N_cams = points.shape[0] // 3
    assert N_cams * 3 == points.shape[0]
    assert len(cam_ids) == N_cams
    points_by_cam_id = {}
    for i, cam_id in enumerate(cam_ids):
        points_by_cam_id[cam_id] = points[i * 3 : i * 3 + 3, :]

    fig = pylab.figure()
    ax = None
    for i, cam_id in enumerate(cam_ids):
        ax = fig.add_subplot(len(cam_ids) + 1, 1, i + 1, sharex=ax)
        ax.plot(points_by_cam_id[cam_id][0, :], "r.")
        ax.plot(points_by_cam_id[cam_id][1, :], "g.")
        if idx_and_obj_id is not None:
            for idx, obj_id in idx_and_obj_id:
                ax.text(idx, 0, "%d" % obj_id)

        pylab.ylabel("%d\n%s" % (i + 1, cam_id))
    # plot sum
    ax = fig.add_subplot(len(cam_ids) + 1, 1, len(cam_ids) + 1, sharex=ax)
    ax.plot(numpy.sum(~numpy.isnan(points) / 3, axis=0), "k.")
    pylab.show()


def main():
    usage = "%prog CALDIR"

    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    caldir = args[0]
    doit(caldir)
