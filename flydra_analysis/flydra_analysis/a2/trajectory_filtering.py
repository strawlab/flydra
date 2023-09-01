from __future__ import print_function
from __future__ import absolute_import
import warnings
from . import xml_stimulus, utils  # flydra.a2 modules
import numpy as np

DEBUG = 0
if DEBUG:
    import matplotlib.pyplot as plt


def iterate_over_subtrajectories(
    style="not on walls", dist=0.01, min_samples=10, data=None, stimulus=None
):
    """break data into subtrajectories according to style

    Sample usage:

    for rows in iterate_over_subtrajectories(style='not on walls',
        dist=0.01, # in meters
        min_samples=10,
        data=all_rows,
        stimulus=stimulus):
    """
    if style != "not on walls":
        raise NotImplementedError("")
    if not isinstance(stimulus, xml_stimulus.Stimulus):
        raise ValueError("stimulus must be instance of xml_stimulus.Stimulus")
    if 1:
        warnings.warn("ignoring stimulus file, using sample_stim_cubic_arena")
        good_cond = np.ones((len(data),), dtype=np.bool_)
        if 0:
            print()
            print("obj %d" % data["obj_id"][0])
            print("all")
            good_idx = np.nonzero(good_cond)[0]
            for start, stop in utils.get_contig_chunk_idxs(good_idx):
                print(start, stop)
        good_cond &= ((-0.5 + dist) < data["x"]) & (data["x"] < (0.5 - dist))
        if 0:
            print("x")
            good_idx = np.nonzero(good_cond)[0]
            for start, stop in utils.get_contig_chunk_idxs(good_idx):
                print(start, stop)
        good_cond &= ((-0.15 + dist) < data["y"]) & (data["y"] < (0.15 - dist))
        if 0:
            ## print data['y'][30:40]
            ## print range(30,40)
            ## print good_cond[30:40]
            print("y")
            good_idx = np.nonzero(good_cond)[0]
            ## print good_idx[:40]
            for start, stop in utils.get_contig_chunk_idxs(good_idx):
                print(start, stop)
        good_cond &= ((0 + dist) < data["z"]) & (data["z"] < (0.3 - dist))
        if 0:
            print("z")
            good_idx = np.nonzero(good_cond)[0]
            for start, stop in utils.get_contig_chunk_idxs(good_idx):
                print(start, stop)
    else:
        1 / 0  # not implemented
    good_idx = np.nonzero(good_cond)[0]
    if DEBUG:
        ax1 = plt.subplot(211)
        ax1.plot(data["x"], data["y"], "r.")
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(data["x"], data["z"], "r.")
    sub_data = None
    if 0:
        print("good idx")
        print(good_idx)
    for starti, stopi in utils.get_contig_chunk_idxs(good_idx):
        start = good_idx[starti]
        stop = good_idx[stopi - 1] + 1  # (-1 is to convert from stop idx)
        if 0:
            print("starti,stopi", starti, stopi, end=" ")
            print("start,stop", start, stop)
        if (stop - start) >= min_samples:
            sub_data = data[start:stop]
            yield sub_data
            if DEBUG:
                ax1.plot(sub_data["x"], sub_data["y"], "b.")
                ax2.plot(sub_data["x"], sub_data["z"], "b.")
    if DEBUG:
        if sub_data is None:
            plt.clf()  # clear plot for next round if no sub-data found
        else:
            plt.title("obj %d" % data["obj_id"][0])
            ax1.set_xlabel("x (m)")
            ax1.set_ylabel("y (m)")
            ax2.set_xlabel("x (m)")
            ax2.set_ylabel("z (m)")
            plt.show()
