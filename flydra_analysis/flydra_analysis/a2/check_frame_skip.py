from __future__ import print_function
import tables
import numpy
import math
from flydra_analysis.analysis.result_utils import get_caminfo_dicts
import sys

if sys.platform == "darwin":
    # for interactive use in IPython:
    import matplotlib

    matplotlib.use("TkAgg")

if __name__ == "__main__":
    filename = "DATA20070214_192124.h5"
    print("filename", filename)
    n_cams = 4
    print("n_cams", n_cams)
    kresults = tables.open_file(filename, mode="r")
    camn2cam_id, cam_id2camns = get_caminfo_dicts(kresults)
    data2d = kresults.root.data2d_distorted

    print("len(data2d)", len(data2d))
    timestamp_vectors = []
    framenumbers = []
    if 1:
        # This is a slow but correct method that gets all data for a given frame
        try:
            # ipython speedup
            allframes
            uframes
            sortidx
            sortedframes
        except NameError:
            allframes = data2d.read(field="frame", flavor="numpy")
            uframes = numpy.unique(allframes)
            sortidx = numpy.argsort(allframes)
            sortedframes = allframes[sortidx]

        print("n frames %d (%d-%d)" % (len(uframes), uframes[0], uframes[-1]))
        doframes = (  # list(range(3047,10000)) +
            # list(range(1000000,1001000)) +
            #                     list(range(2000000,2001000)) +
            #                     list(range(3000000,3001000)) +
            #                     list(range(4000000,4001000)) +
            list(range(5000000, 5001000))
        )
        doframes = numpy.array(doframes)

        left_sorted_idxs = numpy.searchsorted(sortedframes, doframes, side="left")
        right_sorted_idxs = numpy.searchsorted(sortedframes, doframes, side="right")
        for i, frame in enumerate(doframes):
            start_idx = left_sorted_idxs[i]
            stop_idx = right_sorted_idxs[i]
            idxs = sortidx[start_idx:stop_idx]

            if 1:
                # test assumption
                assert numpy.alltrue(allframes[idxs] == frame)
            framedata = data2d.read_coordinates(idxs, flavor="numpy")
            camns = framedata.field("camn")
            test_ucamns = numpy.unique(camns)
            if len(test_ucamns) < n_cams:
                continue
            ucamns = test_ucamns

            ucamns.sort()
            cond2_idxs = numpy.zeros((n_cams,), dtype=numpy.int32)
            for i, camn in enumerate(ucamns):
                first_idx = numpy.nonzero(camns == camn)[0][0]
                cond2_idxs[i] = first_idx

            timestamp_vector = framedata.field("timestamp")[cond2_idxs]
            timestamp_vectors.append(timestamp_vector)
            framenumbers.append(frame)

    elif 0:
        # This is a fast but incorrect method in that it doesn't try to
        # get ALL saved data for a given frame, but rather loads saved
        # data in chunks.

        # chunksize = 100000 # rows
        chunksize = 10000  # rows
        chunknum = 0
        maxchunks = 2

        num_chunks = int(math.ceil(len(data2d) / chunksize))
        print("num_chunks", num_chunks)
        # for chunknum in [0,1]:
        for chunknum in [0, 500, 1000, 2000]:
            chunkstart = chunknum * chunksize
            chunkstop = (chunknum + 1) * chunksize

            # ignore frames that get caught between two chunks...
            chunk = data2d.read(start=chunkstart, stop=chunkstop, flavor="numpy")

            frames = chunk.field("frame")
            uframes = numpy.unique(frames)
            print("chunknum %d, %d frames" % (chunknum, len(uframes)))
            framecount = 0
            for frame_idx, frame in enumerate(uframes):
                if frame_idx % 100 == 0:
                    print("  frame_idx %d of %d" % (frame_idx, len(uframes)))
                cond1 = frames == frame
                camns = chunk.field("camn")[cond1]
                test_ucamns = numpy.unique(camns)
                if len(test_ucamns) < n_cams:
                    continue
                ucamns = test_ucamns
                framecount += 1

                ucamns.sort()
                cond2_idxs = numpy.zeros((n_cams,), dtype=numpy.int32)
                for i, camn in enumerate(ucamns):
                    first_idx = numpy.nonzero(camns == camn)[0][0]
                    cond2_idxs[i] = first_idx

                timestamps_unordered = chunk.field("timestamp")[cond1]
                timestamp_vector = timestamps_unordered[cond2_idxs]
                timestamp_vectors.append(timestamp_vector)
                framenumbers.append(frame)
            print("    total frames with all data %d" % (len(timestamp_vectors),))
    kresults.close()

    if len(timestamp_vectors) == 0:
        raise ValueError("no frames found")

    dtype = timestamp_vectors[0].dtype
    timestamp_vectors = numpy.array(timestamp_vectors, dtype=dtype)
    framenumbers = numpy.array(framenumbers)

    import pylab

    if 1:
        for ref_idx in range(len(ucamns)):
            diff_vectors = (
                timestamp_vectors - timestamp_vectors[:, ref_idx, numpy.newaxis]
            )

            cam_ids = [camn2cam_id[ucamn] for ucamn in ucamns]
            neworder = numpy.argsort(cam_ids)

            # plot timestamp differences
            pylab.figure()
            pylab.figtext(
                0.5, 0.99, "ref: %s" % (cam_ids[ref_idx],),
            )
            ax = None
            maxrange = 0
            for i in range(n_cams):
                dvmsec = diff_vectors[:, i] * 1e3
                thisrange = dvmsec.max() - dvmsec.min()
                maxrange = max(maxrange, thisrange)
            for i in range(n_cams):
                idx = neworder[i]
                ax = pylab.subplot(n_cams, 1, i + 1, sharex=ax)
                dvmsec = diff_vectors[:, idx] * 1e3
                pylab.plot(framenumbers, dvmsec, ".-", label=camn2cam_id[ucamns[idx]])
                miny = dvmsec.min()
                ax.set_ylim((miny - 0.5, miny + maxrange + 0.5))
                pylab.ylabel("msec")
                pylab.xlabel("frame")
                pylab.legend(loc="upper left")

    elif 0:
        # make all timestamps relative to first camn
        reference_cam_id = "cam2_0"
        for i, camn in enumerate(ucamns):
            if camn2cam_id[camn] == reference_cam_id:
                ref_idx = i
                break
        diff_vectors = timestamp_vectors - timestamp_vectors[:, ref_idx, numpy.newaxis]

        # find consecutive frames
        frame_diff = framenumbers[1:] - framenumbers[:-1]
        consecutive_idx = frame_diff == 1

        # find L2 distance (consecutive) timestamps
        l2dist = numpy.sqrt(
            numpy.sum((diff_vectors[1:] - diff_vectors[:-1]) ** 2, axis=1)
        )
        l2dist = l2dist[consecutive_idx]

        if 0:
            # plot l2 distance
            pylab.figure()
            pylab.plot(framenumbers[consecutive_idx], l2dist, ".")
            pylab.xlabel("prior frame")
            pylab.ylabel("L2-norm dist")

        if 1:
            cam_ids = [camn2cam_id[ucamn] for ucamn in ucamns]
            neworder = numpy.argsort(cam_ids)

            # plot timestamp differences
            pylab.figure()
            ax = None
            maxrange = 0
            for i in range(n_cams):
                dvmsec = diff_vectors[:, i] * 1e3
                thisrange = dvmsec.max() - dvmsec.min()
                maxrange = max(maxrange, thisrange)
            for i in range(n_cams):
                idx = neworder[i]
                ax = pylab.subplot(n_cams, 1, i + 1, sharex=ax)
                dvmsec = diff_vectors[:, idx] * 1e3
                pylab.plot(framenumbers, dvmsec, ".-", label=camn2cam_id[ucamns[idx]])
                miny = dvmsec.min()
                ax.set_ylim((miny - 0.5, miny + maxrange + 0.5))
                pylab.ylabel("msec")
                pylab.xlabel("frame")
                pylab.legend(loc="upper left")
    pylab.show()
