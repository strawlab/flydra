# see a2/save_movies_overlay.py
from __future__ import division
from __future__ import print_function
import numpy
from numpy import nan, pi
import tables as PT
import tables.flavor

tables.flavor.restrict_flavors(keep=["numpy"])  # ensure pytables 2.x
import pytz  # from http://pytz.sourceforge.net/
import datetime
import sys
from optparse import OptionParser
import pylab
import flydra_core.reconstruct
import flydra_analysis.analysis.result_utils as result_utils
import matplotlib.colors


def auto_subplot(fig, n, n_rows=2, n_cols=3):
    # 2 rows and n_cols

    rrow = n // n_cols  # reverse row
    row = n_rows - rrow - 1  # row number
    col = n % n_cols

    x_space = 0.02 / n_cols
    # y_space = 0.0125
    y_space = 0.03
    y_size = (1.0 / n_rows) - (2 * y_space)

    left = col * (1.0 / n_cols) + x_space
    bottom = row * y_size + y_space
    w = (1.0 / n_cols) - x_space
    h = y_size - 2 * y_space
    return fig.add_axes([left, bottom, w, h])


def show_it(
    fig, filename, frame_start=None, frame_stop=None,
):

    results = result_utils.get_results(filename, mode="r")
    reconstructor = flydra_core.reconstruct.Reconstructor(results)
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
    data2d = results.root.data2d_distorted  # make sure we have 2d data table

    print("reading frames...")
    frames = data2d.read(field="frame")
    print("OK")

    if frame_start is not None:
        print("selecting frames after start")
        after_start = numpy.nonzero(frames >= frame_start)[0]
    else:
        after_start = None

    if frame_stop is not None:
        print("selecting frames before stop")
        before_stop = numpy.nonzero(frames <= frame_stop)[0]
    else:
        before_stop = None

    print("finding all frames")
    if after_start is not None and before_stop is not None:
        use_idxs = numpy.intersect1d(after_start, before_stop)
    elif after_start is not None:
        use_idxs = after_start
    elif before_stop is not None:
        use_idxs = before_stop
    else:
        use_idxs = numpy.arange(data2d.nrows)

    # OK, we have data coords, plot
    print("reading cameras")
    frames = frames[use_idxs]
    camns = data2d.read(field="camn")
    camns = camns[use_idxs]
    unique_camns = numpy.unique1d(camns)
    unique_cam_ids = list(set([camn2cam_id[camn] for camn in unique_camns]))
    unique_cam_ids.sort()
    print("%d cameras with data" % (len(unique_cam_ids),))

    if len(unique_cam_ids) == 1:
        n_rows = 1
        n_cols = 1
    else:
        n_rows = 2
        n_cols = 3

    subplot_by_cam_id = {}
    for i, cam_id in enumerate(unique_cam_ids):
        ax = auto_subplot(fig, i, n_rows=n_rows, n_cols=n_cols)
        ax.text(
            0.5,
            0.95,
            "%s: %s" % (cam_id, str(cam_id2camns[cam_id])),
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ##        ax.set_xticks([])
        ##        ax.set_yticks([])
        subplot_by_cam_id[cam_id] = ax

    cdict = {
        "red": ((0, 0, 0), (1, 1, 1)),
        "green": ((0, 0, 0), (0, 0, 0)),
        "blue": ((0, 0, 0), (1, 1, 1)),
    }
    #    black2magenta = matplotlib.colors.LinearSegmentedColormap('black2magenta',cdict,256)

    for camn in unique_camns:
        cam_id = camn2cam_id[camn]
        ax = subplot_by_cam_id[cam_id]
        this_camn_idxs = use_idxs[camns == camn]

        w, h = reconstructor.get_resolution(cam_id)
        xbins = numpy.linspace(0, w, 40)
        ybins = numpy.linspace(0, h, 30)

        xs = data2d.read_coordinates(this_camn_idxs, field="x")
        ys = data2d.read_coordinates(this_camn_idxs, field="y")

        hist, xedges, yedges = numpy.histogram2d(xs, ys, bins=(xbins, ybins))
        hist = numpy.ma.masked_where(hist == 0, hist)
        im = getattr(results.root.images, cam_id)

        ax.imshow(im, origin="lower", extent=[0, w - 1, 0, h - 1], cmap=pylab.cm.gray)
        pcolor_im = ax.pcolor(
            xbins, ybins, hist.T, alpha=0.5,
        )  # cmap = black2magenta )
        # fig.colorbar( pcolor_im, cax )

        res = reconstructor.get_resolution(cam_id)
        ax.set_xlim([0, res[0]])
        # ax.set_ylim([0,res[1]])
        ax.set_ylim([res[1], 0])

    if 1:
        return

    # Old code below:
    # Do same as above for Kalman-filtered data

    if kalman_filename is None:
        return

    kresults = PT.open_file(kalman_filename, mode="r")
    kobs = kresults.root.ML_estimates
    kframes = kobs.read(field="frame")
    if frame_start is not None:
        k_after_start = numpy.nonzero(kframes >= frame_start)[0]
        # k_after_start = kobs.read_coordinates(idxs)
        # k_after_start = kobs.get_where_list(
        #    'frame>=frame_start')
    else:
        k_after_start = None
    if frame_stop is not None:
        k_before_stop = numpy.nonzero(kframes <= frame_stop)[0]
        # k_before_stop = kobs.read_coordinates(idxs)
        # k_before_stop = kobs.get_where_list(
        #    'frame<=frame_stop')
    else:
        k_before_stop = None

    if k_after_start is not None and k_before_stop is not None:
        k_use_idxs = numpy.intersect1d(k_after_start, k_before_stop)
    elif k_after_start is not None:
        k_use_idxs = k_after_start
    elif k_before_stop is not None:
        k_use_idxs = k_before_stop
    else:
        k_use_idxs = numpy.arange(kobs.nrows)

    obj_ids = kobs.read(field="obj_id")[k_use_idxs]
    # obj_ids = kobs.read_coordinates( k_use_idxs,
    #                                field='obj_id')
    obs_2d_idxs = kobs.read(field="obs_2d_idx")[k_use_idxs]
    # obs_2d_idxs = kobs.read_coordinates( k_use_idxs,
    #                                    field='obs_2d_idx')
    kframes = kframes[k_use_idxs]  # kobs.read_coordinates( k_use_idxs,
    # field='frame')

    kobs_2d = kresults.root.ML_estimates_2d_idxs
    xys_by_obj_id = {}
    for obj_id, kframe, obs_2d_idx in zip(obj_ids, kframes, obs_2d_idxs):
        obs_2d_idx_find = int(obs_2d_idx)  # XXX grr, why can't pytables do this?
        obj_id_save = int(obj_id)  # convert from possible numpy scalar
        xys_by_cam_id = xys_by_obj_id.setdefault(obj_id_save, {})
        kobs_2d_data = kobs_2d.read(start=obs_2d_idx_find, stop=obs_2d_idx_find + 1)
        assert len(kobs_2d_data) == 1
        kobs_2d_data = kobs_2d_data[0]
        this_camns = kobs_2d_data[0::2]
        this_camn_idxs = kobs_2d_data[1::2]

        this_use_idxs = use_idxs[frames == kframe]
        if debugADS:
            print()
            print(kframe, "===============")
            print("this_use_idxs", this_use_idxs)

        d2d = data2d.read_coordinates(this_use_idxs)
        if debugADS:
            print("d2d ---------------")
            for row in d2d:
                print(row)
        for this_camn, this_camn_idx in zip(this_camns, this_camn_idxs):
            this_idxs_tmp = numpy.nonzero(d2d["camn"] == this_camn)[0]
            this_camn_d2d = d2d[d2d["camn"] == this_camn]
            found = False
            for this_row in this_camn_d2d:  # XXX could be sped up
                if this_row["frame_pt_idx"] == this_camn_idx:
                    found = True
                    break
            if not found:
                if 0:
                    print("WARNING:point not found in data!?")
                    continue
                else:
                    raise RuntimeError("point not found in data!?")
            # this_row = this_camn_d2d[this_camn_idx]
            this_cam_id = camn2cam_id[this_camn]
            xys = xys_by_cam_id.setdefault(this_cam_id, ([], []))
            xys[0].append(this_row["x"])
            xys[1].append(this_row["y"])

    for obj_id in xys_by_obj_id:
        xys_by_cam_id = xys_by_obj_id[obj_id]
        for cam_id, (xs, ys) in xys_by_cam_id.items():
            ax = subplot_by_cam_id[cam_id]
            if 0:
                ax.plot(xs, ys, label="obs: %d" % obj_id)
            else:
                ax.plot(xs, ys, "x-", label="obs: %d" % obj_id)
            ax.text(xs[0], ys[0], "%d:" % (obj_id,))
            ax.text(xs[-1], ys[-1], ":%d" % (obj_id,))

    for cam_id in subplot_by_cam_id.keys():
        ax = subplot_by_cam_id[cam_id]
        ax.legend()
    print("note: could/should also plot re-projection of Kalman filtered/smoothed data")


def main():
    usage = "%prog FILE [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        type="string",
        help="hdf5 file with data to display FILE",
        metavar="FILE",
    )

    ## parser.add_option('-k', "--kalman-file", dest="kalman_filename", type='string',
    ##                   help="hdf5 file with kalman data to display KALMANFILE",
    ##                   metavar="KALMANFILE")

    parser.add_option(
        "--start", type="int", help="first frame to plot", metavar="START"
    )

    parser.add_option("--stop", type="int", help="last frame to plot", metavar="STOP")

    (options, args) = parser.parse_args()

    if options.filename is not None:
        args.append(options.filename)

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]

    fig = pylab.figure()
    show_it(
        fig,
        h5_filename,
        ## kalman_filename = options.kalman_filename,
        frame_start=options.start,
        frame_stop=options.stop,
    )
    pylab.show()


if __name__ == "__main__":
    main()
