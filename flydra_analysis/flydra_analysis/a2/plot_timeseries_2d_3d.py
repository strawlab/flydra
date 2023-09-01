from __future__ import division, print_function
from __future__ import absolute_import

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import os, sys, math

import pkg_resources
import numpy
import numpy as np
import tables as PT
from optparse import OptionParser
import flydra_core.reconstruct as reconstruct

import matplotlib
import matplotlib.ticker as ticker

import flydra_analysis.analysis.result_utils as result_utils
import flydra_analysis.a2.utils as utils
from flydra_core.kalman.point_prob import some_rough_negative_log_likelihood
from . import core_analysis

import datetime, time
import collections

all_kalman_lines = {}


def onpick_callback(event):
    # see matplotlib/examples/pick_event_demo.py
    thisline = event.artist
    obj_id = all_kalman_lines[thisline]
    print("obj_id", obj_id)
    if 0:
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print("picked line:", zip(numpy.take(xdata, ind), numpy.take(ydata, ind)))


class DateFormatter:
    def __init__(self, tz):
        self.tz = tz

    def format_date(self, x, pos=None):
        val = datetime.datetime.fromtimestamp(x, self.tz)
        return val.strftime("%Y-%m-%d %H:%M:%S.%f")


def doit(
    filenames=None,
    start=None,
    stop=None,
    kalman_filename=None,
    fps=None,
    use_kalman_smoothing=True,
    dynamic_model=None,
    up_dir=None,
    options=None,
):
    if options.save_fig is not None:
        matplotlib.use("Agg")
    import pylab

    if not use_kalman_smoothing:
        if (fps is not None) or (dynamic_model is not None):
            print(
                "WARNING: disabling Kalman smoothing "
                "(--disable-kalman-smoothing) is "
                "incompatable with setting fps and "
                "dynamic model options (--fps and "
                "--dynamic-model)",
                file=sys.stderr,
            )

    ax = None
    ax_by_cam = {}
    fig = pylab.figure()

    assert len(filenames) >= 1, "must give at least one filename!"

    n_files = 0
    for filename in filenames:

        if options.show_source_name:
            figtitle = filename
            if kalman_filename is not None:
                figtitle += " " + kalman_filename
        else:
            figtitle = ""
        if options.obj_only is not None:
            figtitle += " only showing objects: " + " ".join(map(str, options.obj_only))
        if figtitle != "":
            pylab.figtext(0.01, 0.01, figtitle, verticalalignment="bottom")

        with PT.open_file(filename, mode="r") as h5:
            if options.spreadh5 is not None:
                h5spread = PT.open_file(options.spreadh5, mode="r")
            else:
                h5spread = None

            if fps is None:
                fps = result_utils.get_fps(h5)

            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
            cam_ids = list(cam_id2camns.keys())
            cam_ids.sort()

            if start is not None or stop is not None:
                frames = h5.root.data2d_distorted.read(field="frame")
                valid_cond = numpy.ones(frames.shape, dtype=numpy.bool)
                if start is not None:
                    valid_cond = valid_cond & (frames >= start)
                if stop is not None:
                    valid_cond = valid_cond & (frames <= stop)
                read_idxs = np.nonzero(valid_cond)[0]
                all_data = []
                for start_stop in utils.iter_contig_chunk_idxs(read_idxs):
                    (read_idx_start_idx, read_idx_stop_idx) = start_stop
                    start_idx = read_idxs[read_idx_start_idx]
                    stop_idx = read_idxs[read_idx_stop_idx - 1]
                    these_rows = h5.root.data2d_distorted.read(
                        start=start_idx, stop=stop_idx + 1
                    )
                    all_data.append(these_rows)
                if len(all_data) == 0:
                    print(
                        "file %s has no frames in range %s - %s"
                        % (filename, start, stop)
                    )
                    continue
                all_data = np.concatenate(all_data)
                del valid_cond, frames, start_idx, stop_idx, these_rows, read_idxs
            else:
                all_data = h5.root.data2d_distorted[:]

            tmp_frames = all_data["frame"]
            if len(tmp_frames) == 0:
                print("file %s has no frames, skipping." % filename)
                continue
            n_files += 1
            start_frame = tmp_frames.min()
            stop_frame = tmp_frames.max()
            del tmp_frames

            for cam_id_enum, cam_id in enumerate(cam_ids):
                if cam_id in ax_by_cam:
                    ax = ax_by_cam[cam_id]
                else:
                    n_subplots = len(cam_ids)
                    if kalman_filename is not None:
                        n_subplots += 1
                    if h5spread is not None:
                        n_subplots += 1
                    ax = pylab.subplot(n_subplots, 1, cam_id_enum + 1, sharex=ax)
                    ax_by_cam[cam_id] = ax
                    ax.fmt_xdata = str
                    ax.fmt_ydata = str

                camns = cam_id2camns[cam_id]
                cam_id_n_valid = 0
                for camn in camns:
                    this_idx = numpy.nonzero(all_data["camn"] == camn)[0]
                    data = all_data[this_idx]

                    xdata = data["x"]
                    valid = ~numpy.isnan(xdata)

                    data = data[valid]
                    del valid

                    if options.area_threshold > 0.0:
                        area = data["area"]

                        valid2 = area >= options.area_threshold
                        data = data[valid2]
                        del valid2

                    if options.likely_only:
                        pt_area = data["area"]
                        cur_val = data["cur_val"]
                        mean_val = data["mean_val"]
                        sumsqf_val = data["sumsqf_val"]

                        p_y_x = some_rough_negative_log_likelihood(
                            pt_area, cur_val, mean_val, sumsqf_val
                        )
                        valid3 = np.isfinite(p_y_x)
                        data = data[valid3]

                    n_valid = len(data)
                    cam_id_n_valid += n_valid
                    if options.timestamps:
                        xdata = data["timestamp"]
                    else:
                        xdata = data["frame"]
                    if n_valid >= 1:
                        ax.plot(xdata, data["x"], "ro", ms=2, mew=0)
                        ax.plot(xdata, data["y"], "go", ms=2, mew=0)
                ax.text(
                    0.1,
                    0,
                    "%s %s: %d pts" % (cam_id, cam_id2camns[cam_id], cam_id_n_valid),
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=ax.transAxes,
                )
                ax.set_ylabel("pixels")
                if not options.timestamps:
                    ax.set_xlim((start_frame, stop_frame))
            ax.set_xlabel("frame")
            if options.timestamps:
                timezone = result_utils.get_tz(h5)
                df = DateFormatter(timezone)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(df.format_date))
            else:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            if h5spread is not None:
                if options.timestamps:
                    raise NotImplementedError(
                        "--timestamps is currently incompatible with --spreadh5"
                    )
                ax_by_cam["h5spread"] = ax
                if kalman_filename is not None:
                    # this is 2nd to last
                    ax = pylab.subplot(n_subplots, 1, n_subplots - 1, sharex=ax)
                else:
                    # this is last
                    ax = pylab.subplot(n_subplots, 1, n_subplots, sharex=ax)

                frames = h5spread.root.framenumber[:]
                spread = h5spread.root.spread[:]

                valid_cond = numpy.ones(frames.shape, dtype=numpy.bool)
                if start is not None:
                    valid_cond = valid_cond & (frames >= start)
                if stop is not None:
                    valid_cond = valid_cond & (frames <= stop)

                spread_msec = spread[valid_cond] * 1000.0
                ax.plot(frames[valid_cond], spread_msec, "o", ms=2, mew=0)

                if spread_msec.max() < 1.0:
                    ax.set_ylim((0, 1))
                    ax.set_yticks([0, 1])
                ax.set_xlabel("frame")
                ax.set_ylabel("timestamp spread (msec)")
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                h5spread.close()
                del frames
                del spread

    if options.timestamps:
        fig.autofmt_xdate()

    if kalman_filename is not None:
        if 1:
            ax = pylab.subplot(n_subplots, 1, n_subplots, sharex=ax)
            ax_by_cam["kalman pmean"] = ax
            ax.fmt_xdata = str
            ax.set_ylabel("3d error\nmeters")

        frame_start = start
        frame_stop = stop

        # copied from save_movies_overlay.py
        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(
            kalman_filename
        )
        if options.timestamps:
            time_model = result_utils.get_time_model_from_data(data_file)
        if "frames" in extra:
            frames = extra["frames"]
            valid_cond = np.ones((len(frames,)), dtype=np.bool_)
            if start is not None:
                valid_cond &= frames >= start
            if stop is not None:
                valid_cond &= frames <= stop
            obj_ids = obj_ids[valid_cond]
            use_obj_ids = np.unique(obj_ids)
            print("quick found use_obj_ids", use_obj_ids)
        if is_mat_file:
            raise ValueError(
                "cannot use .mat file for kalman_filename "
                "because it is missing the reconstructor "
                "and ability to get framenumbers"
            )
        R = reconstruct.Reconstructor(data_file)

        if options.obj_only is not None:
            use_obj_ids = options.obj_only

        if dynamic_model is None and use_kalman_smoothing:
            dynamic_model = extra["dynamic_model_name"]
            print('detected file loaded with dynamic model "%s"' % dynamic_model)
            if dynamic_model.startswith("EKF "):
                dynamic_model = dynamic_model[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model)

        if options.reproj_error:
            reproj_error = collections.defaultdict(list)
            max_reproj_error = {}
            kalman_rows = []
            for obj_id in use_obj_ids:
                kalman_rows.append(ca.load_observations(obj_id, data_file))
            kalman_rows = numpy.concatenate(kalman_rows)
            kalman_3d_frame = kalman_rows["frame"]

            if start is not None or stop is not None:
                if start is None:
                    start = -numpy.inf
                if stop is None:
                    stop = numpy.inf
                valid_cond = (kalman_3d_frame >= start) & (kalman_3d_frame <= stop)

                kalman_rows = kalman_rows[valid_cond]
                kalman_3d_frame = kalman_3d_frame[valid_cond]

            # modified from save_movies_overlay
            for this_3d_row_enum, this_3d_row in enumerate(kalman_rows):
                if this_3d_row_enum % 100 == 0:
                    print(
                        "doing reprojection error for MLE 3d estimate for "
                        "row %d of %d" % (this_3d_row_enum, len(kalman_rows))
                    )
                vert = numpy.array(
                    [this_3d_row["x"], this_3d_row["y"], this_3d_row["z"]]
                )
                obj_id = this_3d_row["obj_id"]
                if numpy.isnan(vert[0]):
                    # no observation this frame
                    continue
                obs_2d_idx = this_3d_row["obs_2d_idx"]
                try:
                    kobs_2d_data = data_file.root.ML_estimates_2d_idxs[int(obs_2d_idx)]
                except tables.exceptions.NoSuchNodeError as err:
                    # backwards compatibility
                    kobs_2d_data = data_file.root.kalman_observations_2d_idxs[
                        int(obs_2d_idx)
                    ]

                # parse VLArray
                this_camns = kobs_2d_data[0::2]
                this_camn_idxs = kobs_2d_data[1::2]

                # find original 2d data
                #   narrow down search
                obs2d = all_data[all_data["frame"] == this_3d_row["frame"]]

                for camn, this_camn_idx in zip(this_camns, this_camn_idxs):
                    cam_id = camn2cam_id[camn]

                    # do projection to camera image plane
                    vert_image = R.find2d(cam_id, vert, distorted=True)

                    new_cond = (obs2d["camn"] == camn) & (
                        obs2d["frame_pt_idx"] == this_camn_idx
                    )
                    assert numpy.sum(new_cond) == 1

                    x = obs2d[new_cond]["x"][0]
                    y = obs2d[new_cond]["y"][0]

                    this_reproj_error = numpy.sqrt(
                        (vert_image[0] - x) ** 2 + (vert_image[1] - y) ** 2
                    )
                    if this_reproj_error > 100:
                        print(
                            "  reprojection error > 100 (%.1f) at frame %d "
                            "for camera %s, obj_id %d"
                            % (this_reproj_error, this_3d_row["frame"], cam_id, obj_id)
                        )
                    if numpy.isnan(this_reproj_error):
                        print("error:")
                        print(this_camns, this_camn_idxs)
                        print(cam_id)
                        print(vert_image)
                        print(vert)
                        raise ValueError("nan at frame %d" % this_3d_row["frame"])
                    reproj_error[cam_id].append(this_reproj_error)
                    if cam_id in max_reproj_error:
                        (
                            cur_max_frame,
                            cur_max_reproj_error,
                            cur_obj_id,
                        ) = max_reproj_error[cam_id]
                        if this_reproj_error > cur_max_reproj_error:
                            max_reproj_error[cam_id] = (
                                this_3d_row["frame"],
                                this_reproj_error,
                                obj_id,
                            )
                    else:
                        max_reproj_error[cam_id] = (
                            this_3d_row["frame"],
                            this_reproj_error,
                            obj_id,
                        )

            del kalman_rows, kalman_3d_frame, obj_ids
            print("mean reprojection errors:")
            cam_ids = reproj_error.keys()
            cam_ids.sort()
            for cam_id in cam_ids:
                errors = reproj_error[cam_id]
                mean_error = numpy.mean(errors)
                worst_frame, worst_error, worst_obj_id = max_reproj_error[cam_id]
                print(
                    " %s: %.1f (worst: frame %d, obj_id %d, error %.1f)"
                    % (cam_id, mean_error, worst_frame, worst_obj_id, worst_error)
                )
            print()

        for kalman_smoothing in [True, False]:
            if use_kalman_smoothing == False and kalman_smoothing == True:
                continue
            print("loading frame numbers for kalman objects (estimates)")
            kalman_rows = []
            for obj_id in use_obj_ids:
                try:
                    my_rows = ca.load_data(
                        obj_id,
                        data_file,
                        use_kalman_smoothing=kalman_smoothing,
                        dynamic_model_name=dynamic_model,
                        frames_per_second=fps,
                        up_dir=up_dir,
                    )
                except core_analysis.NotEnoughDataToSmoothError as err:
                    # OK, we don't have data from this obj_id
                    continue
                else:
                    kalman_rows.append(my_rows)
            if not len(kalman_rows):
                # no data
                continue
            kalman_rows = numpy.concatenate(kalman_rows)
            kalman_3d_frame = kalman_rows["frame"]

            if start is not None or stop is not None:
                if start is None:
                    start = -numpy.inf
                if stop is None:
                    stop = numpy.inf
                valid_cond = (kalman_3d_frame >= start) & (kalman_3d_frame <= stop)

                kalman_rows = kalman_rows[valid_cond]
                kalman_3d_frame = kalman_3d_frame[valid_cond]

            obj_ids = kalman_rows["obj_id"]
            use_obj_ids = numpy.unique(obj_ids)
            non_nan_rows = ~np.isnan(kalman_rows["x"])
            print("plotting %d Kalman objects" % (len(use_obj_ids),))
            for obj_id in use_obj_ids:
                cond = obj_ids == obj_id
                cond &= non_nan_rows
                x = kalman_rows["x"][cond]
                y = kalman_rows["y"][cond]
                z = kalman_rows["z"][cond]
                w = numpy.ones(x.shape)
                X = numpy.vstack((x, y, z, w)).T
                frame = kalman_rows["frame"][cond]
                # print '%d %d %d'%(frame[0],obj_id, len(frame))
                if options.timestamps:
                    time_est = time_model.framestamp2timestamp(frame)

                if kalman_smoothing:
                    kwprops = dict(lw=0.5)
                else:
                    kwprops = dict(lw=1)

                for cam_id in cam_ids:
                    if cam_id not in R.get_cam_ids():
                        print(
                            "no calibration for %s: not showing 3D projections"
                            % (cam_id,)
                        )
                        continue
                    ax = ax_by_cam[cam_id]
                    x2d = R.find2d(cam_id, X, distorted=True)
                    ## print '%d %d %s (%f,%f)'%(
                    ##     obj_id,frame[0],cam_id,x2d[0,0],x2d[1,0])
                    if options.timestamps:
                        xdata = time_est
                    else:
                        xdata = frame
                    ax.text(xdata[0], x2d[0, 0], "%d" % obj_id)
                    (thisline,) = ax.plot(
                        xdata, x2d[0, :], "b-", pickradius=5, **kwprops
                    )  # 5pt tolerance
                    all_kalman_lines[thisline] = obj_id
                    (thisline,) = ax.plot(
                        xdata, x2d[1, :], "y-", pickradius=5, **kwprops
                    )  # 5pt tolerance
                    all_kalman_lines[thisline] = obj_id
                    ax.set_ylim([-100, 800])
                    if options.timestamps:
                        ## ax.set_xlim( *time_model.framestamp2timestamp(
                        ##     (start_frame, stop_frame) ))
                        pass
                    else:
                        ax.set_xlim((start_frame, stop_frame))
                if 1:
                    ax = ax_by_cam["kalman pmean"]
                    P00 = kalman_rows["P00"][cond]
                    P11 = kalman_rows["P11"][cond]
                    P22 = kalman_rows["P22"][cond]
                    Pmean = numpy.sqrt(P00 ** 2 + P11 ** 2 + P22 ** 2)  # variance
                    std = numpy.sqrt(Pmean)  # standard deviation (in meters)
                    if options.timestamps:
                        xdata = time_est
                    else:
                        xdata = frame
                    ax.plot(xdata, std, "k-", **kwprops)

                    if options.timestamps:
                        ax.set_xlabel("time (sec)")
                        timezone = result_utils.get_tz(h5)
                        df = DateFormatter(timezone)
                        ax.xaxis.set_major_formatter(
                            ticker.FuncFormatter(df.format_date)
                        )
                        for label in ax.get_xticklabels():
                            label.set_rotation(30)

                    else:
                        ax.set_xlabel("frame")
                        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%s"))

            if not kalman_smoothing:
                # plot 2D data contributing to 3D object
                # this is forked from flydra_analysis_plot_kalman_2d.py

                kresults = ca.get_pytables_file_by_filename(kalman_filename)
                try:
                    kobs = kresults.root.ML_estimates
                except tables.exceptions.NoSuchNodeError:
                    # backward compatibility
                    kobs = kresults.root.kalman_observations
                kframes = kobs.read(field="frame")
                if frame_start is not None:
                    k_after_start = numpy.nonzero(kframes >= frame_start)[0]
                else:
                    k_after_start = None
                if frame_stop is not None:
                    k_before_stop = numpy.nonzero(kframes <= frame_stop)[0]
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

                obs_2d_idxs = kobs.read(field="obs_2d_idx")[k_use_idxs]
                kframes = kframes[k_use_idxs]

                try:
                    kobs_2d = kresults.root.ML_estimates_2d_idxs
                except tables.exceptions.NoSuchNodeError:
                    # backwards compatibility
                    kobs_2d = kresults.root.kalman_observations_2d_idxs
                # this will be slooow...
                used_cam_ids = collections.defaultdict(list)
                for obs_2d_idx, kframe in zip(obs_2d_idxs, kframes):
                    obs_2d_row = kobs_2d[int(obs_2d_idx)]
                    # print kframe,obs_2d_row
                    for camn in obs_2d_row[::2]:
                        try:
                            cam_id = camn2cam_id[camn]
                        except KeyError:
                            cam_id = None
                        if cam_id is not None:
                            used_cam_ids[cam_id].append(kframe)
                for cam_id, kframes_used in used_cam_ids.items():
                    kframes_used = numpy.array(kframes_used)
                    yval = -99 * numpy.ones_like(kframes_used)
                    ax = ax_by_cam[cam_id]
                    if options.timestamps:
                        ax.plot(
                            time_model.framestamp2timestamp(kframes_used), yval, "kx"
                        )
                    else:
                        ax.plot(kframes_used, yval, "kx")
                        ax.set_xlim((start_frame, stop_frame))
                    ax.set_ylim([-100, 800])

        data_file.close()

    if n_files >= 1:
        if options.save_fig is not None:
            pylab.savefig(options.save_fig)
        else:
            fig.canvas.mpl_connect("pick_event", onpick_callback)
            pylab.show()
    else:
        print("No filename(s) with data given -- nothing to do!")


def main():
    usage = "%prog [options] FILE1 [FILE2] ..."

    parser = OptionParser(usage)

    parser.add_option(
        "-k",
        "--kalman-file",
        dest="kalman_filename",
        type="string",
        help=".h5 file with kalman data and 3D reconstructor",
    )

    parser.add_option(
        "--spreadh5",
        type="string",
        help=(
            ".spreadh5 file with frame synchronization info "
            "(make with flydra_analysis_check_sync)"
        ),
    )

    parser.add_option(
        "--start",
        dest="start",
        type="int",
        help="start frame (.h5 frame number reference)",
    )

    parser.add_option(
        "--stop",
        dest="stop",
        type="int",
        help="stop frame (.h5 frame number reference)",
    )

    parser.add_option(
        "--disable-kalman-smoothing",
        action="store_false",
        dest="use_kalman_smoothing",
        default=True,
        help=(
            "show original, causal Kalman filtered data "
            "(rather than Kalman smoothed observations)"
        ),
    )

    parser.add_option("--timestamps", action="store_true", default=False)

    parser.add_option(
        "--reproj-error",
        action="store_true",
        default=False,
        help=(
            "calculate and print to console the mean "
            "reprojection error for each camera"
        ),
    )

    parser.add_option(
        "--hide-source-name",
        action="store_false",
        dest="show_source_name",
        default=True,
        help="show the source filename?",
    )

    parser.add_option(
        "--fps",
        dest="fps",
        type="float",
        help=("frames per second (used for Kalman " "filtering/smoothing)"),
    )

    parser.add_option(
        "--area-threshold",
        type="float",
        default=0.0,
        help=(
            "area of 2D point required for plotting (NOTE: "
            "this is not related to the threshold used for "
            "Kalmanization)"
        ),
    )

    parser.add_option(
        "--likely-only",
        action="store_true",
        default=False,
        help=("plot only points that are deemed likely to " "be true positives"),
    )

    parser.add_option(
        "--save-fig",
        type="string",
        default=None,
        help="path name of figure to save (exits script " "immediately after save)",
    )

    parser.add_option(
        "--dynamic-model", type="string", dest="dynamic_model", default=None,
    )

    parser.add_option("--obj-only", type="string")

    parser.add_option("--up-dir", type="string")

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if options.up_dir is not None:
        up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        up_dir = None

    doit(
        filenames=args,
        kalman_filename=options.kalman_filename,
        start=options.start,
        stop=options.stop,
        fps=options.fps,
        dynamic_model=options.dynamic_model,
        use_kalman_smoothing=options.use_kalman_smoothing,
        up_dir=up_dir,
        options=options,
    )


if __name__ == "__main__":
    main()
