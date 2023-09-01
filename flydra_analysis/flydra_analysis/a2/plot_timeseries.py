from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

from optparse import OptionParser
import os, sys, math, hashlib
import numpy as np
import pkg_resources
import numpy
import tables as PT
import flydra_core.reconstruct as reconstruct
import flydra_analysis.analysis.result_utils as result_utils
from . import analysis_options
import flydra_analysis.a2.xml_stimulus as xml_stimulus
import flydra_analysis.a2.flypos
import flydra_analysis.a2.utils as utils

import matplotlib
import matplotlib.ticker as ticker

rcParams = matplotlib.rcParams
rcParams["xtick.major.pad"] = 10
rcParams["ytick.major.pad"] = 10

import flydra_analysis.analysis.result_utils as result_utils

from . import core_analysis

import datetime, time


class DateFormatter:
    def __init__(self, tz):
        self.tz = tz

    def format_date(self, x, pos=None):
        return str(datetime.datetime.fromtimestamp(x, self.tz))
        ## return datetime.datetime.fromtimestamp(x,self.tz).strftime(
        ##     '%Y-%m-%d %H:%M:%S.%f')


def plot_err(ax, x, mean, err, color=None):
    ax.plot(x, mean + err, color=color)
    ax.plot(x, mean - err, color=color)


class Frames2Time:
    def __init__(self, frame0, fps, time0=0.0):
        if not isinstance(frame0, numpy.ma.MaskedArray):
            self.f0 = int(frame0)
        else:
            self.f0 = int(frame0.data)
        self.fps = fps
        self.time0 = time0

    def __call__(self, farr):
        farr = numpy.array(farr, dtype=numpy.int64)
        f = farr - self.f0
        f2 = f / self.fps + self.time0
        return f2


class FixupAxesWithTimeZone:
    def __init__(self, tz):
        self.df = DateFormatter(tz)

    def fixup_ax(self, ax, ha="right", rotation=30):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.df.format_date))

        # inspired by matplotlib/figure.py autofmt_xdate()
        for label in ax.get_xticklabels():
            label.set_ha(ha)
            label.set_rotation(rotation)

        ax.set_xlabel("time")


def plot_timeseries(subplot=None, options=None):
    kalman_filename = options.kalman_filename

    if not hasattr(options, "frames"):
        options.frames = False

    if not hasattr(options, "show_landing"):
        options.show_landing = False

    if not hasattr(options, "unicolor"):
        options.unicolor = False

    if not hasattr(options, "show_obj_id"):
        options.show_obj_id = True

    if not hasattr(options, "show_track_ends"):
        options.show_track_ends = False

    start = options.start
    stop = options.stop
    obj_only = options.obj_only
    fps = options.fps
    dynamic_model = options.dynamic_model
    use_kalman_smoothing = options.use_kalman_smoothing

    if not use_kalman_smoothing:
        if dynamic_model is not None:
            print(
                (
                    "WARNING: disabling Kalman smoothing "
                    "(--disable-kalman-smoothing) is incompatable "
                    "with setting dynamic model options (--dynamic-model)"
                ),
                file=sys.stderr,
            )

    ca = core_analysis.get_global_CachingAnalyzer()

    if kalman_filename is None:
        raise ValueError("No kalman_filename given. Nothing to do.")

    m = hashlib.md5()
    m.update(open(kalman_filename, mode="rb").read())
    actual_md5 = m.hexdigest()
    (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(
        kalman_filename
    )
    print(
        "opened kalman file %s %s, %d obj_ids"
        % (kalman_filename, actual_md5, len(use_obj_ids))
    )

    if "frames" in extra:
        if (start is not None) or (stop is not None):
            valid_frames = np.ones((len(extra["frames"]),), dtype=np.bool_)
            if start is not None:
                valid_frames &= extra["frames"] >= start
            if stop is not None:
                valid_frames &= extra["frames"] <= stop
            this_use_obj_ids = np.unique(obj_ids[valid_frames])
            use_obj_ids = list(set(use_obj_ids).intersection(this_use_obj_ids))

    include_obj_ids = None
    exclude_obj_ids = None
    do_fuse = False
    if options.stim_xml:
        file_timestamp = data_file.filename[4:19]
        fanout = xml_stimulus.xml_fanout_from_filename(options.stim_xml)
        include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp(
            timestamp_string=file_timestamp
        )
        walking_start_stops = fanout.get_walking_start_stops_for_timestamp(
            timestamp_string=file_timestamp
        )
        if include_obj_ids is not None:
            use_obj_ids = include_obj_ids
        if exclude_obj_ids is not None:
            use_obj_ids = list(set(use_obj_ids).difference(exclude_obj_ids))
        if options.fuse:
            do_fuse = True
    else:
        walking_start_stops = []

    if dynamic_model is None:
        dynamic_model = extra["dynamic_model_name"]
        print('detected file loaded with dynamic model "%s"' % dynamic_model)
        if dynamic_model.startswith("EKF "):
            dynamic_model = dynamic_model[4:]
        print('  for smoothing, will use dynamic model "%s"' % dynamic_model)

    if not is_mat_file:
        mat_data = None

        if fps is None:
            fps = result_utils.get_fps(data_file, fail_on_error=False)

        if fps is None:
            fps = 100.0
            import warnings

            warnings.warn("Setting fps to default value of %f" % fps)

        tz = result_utils.get_tz(data_file)

    dt = 1.0 / fps

    all_vels = []

    if obj_only is not None:
        use_obj_ids = [i for i in use_obj_ids if i in obj_only]

    allX = {}
    frame0 = None

    line2obj_id = {}
    Xz_all = []

    fuse_did_once = False

    if not hasattr(options, "timestamp_file"):
        options.timestamp_file = None

    if not hasattr(options, "ori_qual"):
        options.ori_qual = None

    if options.timestamp_file is not None:
        h5 = tables.open_file(options.timestamp_file, mode="r")
        print("reading timestamps and frames")
        table_data2d_frames = h5.root.data2d_distorted.read(field="frame")
        table_data2d_timestamps = h5.root.data2d_distorted.read(field="timestamp")
        print("done")
        h5.close()
        table_data2d_frames_find = utils.FastFinder(table_data2d_frames)

    if len(use_obj_ids) == 0:
        print("No obj_ids to plot, quitting")
        sys.exit(0)

    time0 = 0.0  # set default value

    for obj_id in use_obj_ids:
        if not do_fuse:
            try:
                kalman_rows = ca.load_data(
                    obj_id,
                    data_file,
                    use_kalman_smoothing=use_kalman_smoothing,
                    dynamic_model_name=dynamic_model,
                    return_smoothed_directions=options.smooth_orientations,
                    frames_per_second=fps,
                    up_dir=options.up_dir,
                    min_ori_quality_required=options.ori_qual,
                )
            except core_analysis.ObjectIDDataError:
                continue
            # kobs_rows = ca.load_dynamics_free_MLE_position( obj_id, data_file )
        else:
            if options.show_3d_orientations:
                raise NotImplementedError(
                    "orientation data is not supported " "when fusing obj_ids"
                )
            if fuse_did_once:
                break
            fuse_did_once = True
            kalman_rows = flydra_analysis.a2.flypos.fuse_obj_ids(
                use_obj_ids,
                data_file,
                dynamic_model_name=dynamic_model,
                frames_per_second=fps,
            )
        frame = kalman_rows["frame"]

        if (start is not None) or (stop is not None):
            valid_cond = numpy.ones(frame.shape, dtype=numpy.bool)

            if start is not None:
                valid_cond = valid_cond & (frame >= start)

            if stop is not None:
                valid_cond = valid_cond & (frame <= stop)

            kalman_rows = kalman_rows[valid_cond]
            if not len(kalman_rows):
                continue

        walking_and_flying_kalman_rows = kalman_rows  # preserve original data

        for flystate in ["flying", "walking"]:
            frame = walking_and_flying_kalman_rows["frame"]  # restore
            if flystate == "flying":
                # assume flying unless we're told it's walking
                state_cond = numpy.ones(frame.shape, dtype=numpy.bool)
            else:
                state_cond = numpy.zeros(frame.shape, dtype=numpy.bool)

            if len(walking_start_stops):
                for walkstart, walkstop in walking_start_stops:
                    frame = walking_and_flying_kalman_rows["frame"]  # restore

                    # handle each bout of walking
                    walking_bout = numpy.ones(frame.shape, dtype=numpy.bool)
                    if walkstart is not None:
                        walking_bout &= frame >= walkstart
                    if walkstop is not None:
                        walking_bout &= frame <= walkstop
                    if flystate == "flying":
                        state_cond &= ~walking_bout
                    else:
                        state_cond |= walking_bout

                kalman_rows = np.take(
                    walking_and_flying_kalman_rows, np.nonzero(state_cond)[0]
                )
                assert len(kalman_rows) == np.sum(state_cond)
                frame = kalman_rows["frame"]

            if frame0 is None:
                frame0 = int(frame[0])

            time0 = 0.0
            if options.timestamp_file is not None:
                frame_idxs = table_data2d_frames_find.get_idxs_of_equal(frame0)
                if len(frame_idxs):
                    time0 = table_data2d_timestamps[frame_idxs[0]]
                else:
                    raise ValueError(
                        "could not fine frame %d in timestamp file" % frame0
                    )

            Xx = kalman_rows["x"]
            Xy = kalman_rows["y"]
            Xz = kalman_rows["z"]

            Dx = Dy = Dz = None
            if options.smooth_orientations:
                Dx = kalman_rows["dir_x"]
                Dy = kalman_rows["dir_y"]
                Dz = kalman_rows["dir_z"]
            elif "rawdir_x" in kalman_rows.dtype.fields:
                Dx = kalman_rows["rawdir_x"]
                Dy = kalman_rows["rawdir_y"]
                Dz = kalman_rows["rawdir_z"]

            if not options.frames:
                f2t = Frames2Time(frame0, fps, time0)
            else:

                def identity(x):
                    return x

                f2t = identity

            kws = {
                "linewidth": 2,
                "picker": 5,
            }
            if options.unicolor:
                kws["color"] = "k"

            line = None

            if "frame" in subplot:
                subplot["frame"].plot(f2t(frame), frame)

            if "P55" in subplot:
                subplot["P55"].plot(f2t(frame), kalman_rows["P55"])

            if "x" in subplot:
                (line,) = subplot["x"].plot(
                    f2t(frame), Xx, label="obj %d (%s)" % (obj_id, flystate), **kws
                )
                line2obj_id[line] = obj_id
                kws["color"] = line.get_color()

            if "y" in subplot:
                (line,) = subplot["y"].plot(
                    f2t(frame), Xy, label="obj %d (%s)" % (obj_id, flystate), **kws
                )
                line2obj_id[line] = obj_id
                kws["color"] = line.get_color()

            if "z" in subplot:
                frame_data = numpy.ma.getdata(frame)  # works if frame is masked or not

                # plot landing time
                if options.show_landing:
                    if flystate == "flying":  # only do this once
                        for walkstart, walkstop in walking_start_stops:
                            if walkstart in frame_data:
                                landing_dix = numpy.nonzero(frame_data == walkstart)[0][
                                    0
                                ]
                                subplot["z"].plot(
                                    [f2t(walkstart)],
                                    [Xz.data[landing_dix]],
                                    "rD",
                                    ms=10,
                                    label="landing",
                                )

                if options.show_track_ends:
                    if flystate == "flying":  # only do this once
                        subplot["z"].plot(
                            f2t([frame_data[0], frame_data[-1]]),
                            [numpy.ma.getdata(Xz)[0], numpy.ma.getdata(Xz)[-1]],
                            "cd",
                            ms=6,
                            label="track end",
                        )

                (line,) = subplot["z"].plot(
                    f2t(frame), Xz, label="obj %d (%s)" % (obj_id, flystate), **kws
                )
                kws["color"] = line.get_color()
                line2obj_id[line] = obj_id

                if flystate == "flying":
                    # only do this once
                    if options.show_obj_id:
                        subplot["z"].text(
                            f2t(frame_data[0]),
                            numpy.ma.getdata(Xz)[0],
                            "%d" % (obj_id,),
                        )
                        line2obj_id[line] = obj_id

            if flystate == "flying":
                Xz_all.append(np.ma.array(Xz).compressed())
                # bins = np.linspace(0,.8,30)
                # print 'Xz.shape',Xz.shape
                # pylab.hist(Xz, bins=bins)

            for (dir_var, Dd) in [("dx", Dx), ("dy", Dy), ("dz", Dz)]:
                if dir_var in subplot:
                    (line,) = subplot[dir_var].plot(
                        f2t(frame), Dd, label="obj %d (%s)" % (obj_id, flystate), **kws
                    )
                    line2obj_id[line] = obj_id
                    kws["color"] = line.get_color()

            if numpy.__version__ >= "1.2.0":
                X = numpy.ma.array((Xx, Xy, Xz))
            else:
                # See http://scipy.org/scipy/numpy/ticket/820
                X = numpy.ma.vstack(
                    (Xx[numpy.newaxis, :], Xy[numpy.newaxis, :], Xz[numpy.newaxis, :])
                )

            dist_central_diff = X[:, 2:] - X[:, :-2]
            vel_central_diff = dist_central_diff / (2 * dt)

            vel2mag = numpy.ma.sqrt(numpy.ma.sum(vel_central_diff ** 2, axis=0))
            xy_vel2mag = numpy.ma.sqrt(
                numpy.ma.sum(vel_central_diff[:2, :] ** 2, axis=0)
            )

            frames2 = frame[1:-1]

            accel4mag = (vel2mag[2:] - vel2mag[:-2]) / (2 * dt)
            frames4 = frames2[1:-1]

            if "vel" in subplot:
                (line,) = subplot["vel"].plot(
                    f2t(frames2),
                    vel2mag,
                    label="obj %d (%s)" % (obj_id, flystate),
                    **kws
                )
                line2obj_id[line] = obj_id
                kws["color"] = line.get_color()

            if "xy_vel" in subplot:
                (line,) = subplot["xy_vel"].plot(
                    f2t(frames2),
                    xy_vel2mag,
                    label="obj %d (%s)" % (obj_id, flystate),
                    **kws
                )
                line2obj_id[line] = obj_id
                kws["color"] = line.get_color()

            if len(accel4mag.compressed()) and "accel" in subplot:
                (line,) = subplot["accel"].plot(
                    f2t(frames4),
                    accel4mag,
                    label="obj %d (%s)" % (obj_id, flystate),
                    **kws
                )
                line2obj_id[line] = obj_id
                kws["color"] = line.get_color()

            if flystate == "flying":
                valid_vel2mag = vel2mag.compressed()
                all_vels.append(valid_vel2mag)
    if len(all_vels):
        all_vels = numpy.hstack(all_vels)
    else:
        all_vels = numpy.array([], dtype=float)

    if 1:
        cond = all_vels < 2.0
        if numpy.ma.sum(cond) != len(all_vels):
            all_vels = all_vels[cond]
            import warnings

            warnings.warn("clipping all velocities > 2.0 m/s")

    if not options.frames:
        xlabel = "time (s)"
    else:
        xlabel = "frame"

    for ax in subplot.itervalues():
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%s"))

    fixup_ax = FixupAxesWithTimeZone(tz).fixup_ax

    if "frame" in subplot:
        if time0 != 0.0:
            fixup_ax(subplot["frame"])
        else:
            subplot["frame"].set_xlabel(xlabel)

    if "x" in subplot:
        subplot["x"].set_ylim([-1, 1])
        subplot["x"].set_ylabel(r"x (m)")
        if time0 != 0.0:
            fixup_ax(subplot["x"])
        else:
            subplot["x"].set_xlabel(xlabel)

    if "y" in subplot:
        subplot["y"].set_ylim([-0.5, 1.5])
        subplot["y"].set_ylabel(r"y (m)")
        if time0 != 0.0:
            fixup_ax(subplot["y"])
        else:
            subplot["y"].set_xlabel(xlabel)

    max_z = None
    if options.stim_xml:
        file_timestamp = options.kalman_filename[4:19]
        stim_xml = xml_stimulus.xml_stimulus_from_filename(
            options.stim_xml, timestamp_string=file_timestamp
        )
        post_max_zs = []
        for post_num, post in enumerate(stim_xml.iterate_posts()):
            post_max_zs.append(
                max(post["verts"][0][2], post["verts"][1][2])
            )  # max post height
        if len(post_max_zs):
            max_z = min(post_max_zs)  # take shortest of posts

    if "z" in subplot:
        subplot["z"].set_ylim([0, 1])
        subplot["z"].set_ylabel(r"z (m)")
        if max_z is not None:
            subplot["z"].axhline(max_z, color="m")
        if time0 != 0.0:
            fixup_ax(subplot["z"])
        else:
            subplot["z"].set_xlabel(xlabel)

    for dir_var in ["dx", "dy", "dz"]:
        if dir_var in subplot:
            subplot[dir_var].set_ylabel(dir_var)
            if time0 != 0.0:
                fixup_ax(subplot[dir_var])
            else:
                subplot[dir_var].set_xlabel(xlabel)

    if "z_hist" in subplot:  # and flystate=='flying':
        Xz_all = np.hstack(Xz_all)
        bins = np.linspace(0, 0.8, 30)
        ax = subplot["z_hist"]
        ax.hist(Xz_all, bins=bins, orientation="horizontal")
        ax.set_xticks([])
        ax.set_yticks([])
        xlim = tuple(ax.get_xlim())  # matplotlib 0.98.3 returned np.array view
        ax.set_xlim((xlim[1], xlim[0]))
        ax.axhline(max_z, color="m")

    if "vel" in subplot:
        subplot["vel"].set_ylim([0, 2])
        subplot["vel"].set_ylabel(r"vel (m/s)")
        subplot["vel"].set_xlabel(xlabel)
        if time0 != 0.0:
            fixup_ax(subplot["vel"])
        else:
            subplot["vel"].set_xlabel(xlabel)

    if "xy_vel" in subplot:
        # subplot['xy_vel'].set_ylim([0,2])
        subplot["xy_vel"].set_ylabel(r"horiz vel (m/s)")
        subplot["xy_vel"].set_xlabel(xlabel)
        if time0 != 0.0:
            fixup_ax(subplot["xy_vel"])
        else:
            subplot["xy_vel"].set_xlabel(xlabel)

    if "accel" in subplot:
        subplot["accel"].set_ylabel(r"acceleration (m/(s^2))")
        subplot["accel"].set_xlabel(xlabel)
        if time0 != 0.0:
            fixup_ax(subplot["accel"])
        else:
            subplot["accel"].set_xlabel(xlabel)

    if "vel_hist" in subplot:
        ax = subplot["vel_hist"]
        bins = numpy.linspace(0, 2, 50)
        ax.set_title("excluding walking")
        pdf, bins, patches = ax.hist(all_vels, bins=bins, normed=True)
        ax.set_xlim(0, 2)
        ax.set_ylabel("probability density")
        ax.set_xlabel("velocity (m/s)")

    return line2obj_id


def doit(options=None,):
    import pylab  # do after matplotlib.use() call

    if options.up_dir is not None:
        options.up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        options.up_dir = None

    fig = pylab.figure()  # figsize=(6,4))
    figtitle = options.kalman_filename
    pylab.figtext(0, 0, figtitle)

    ax = None
    subplot = {}
    if options.show_3d_orientations:
        subplots = ["x", "y", "z", "dx", "dy", "dz"]
    else:
        subplots = ["x", "y", "z", "xy_vel", "vel", "accel", "frame"]

    if options.P55:
        subplots.append("P55")

    # subplots = ['x','y','z','vel','accel']
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots), 1, i + 1, sharex=ax)
        ax.grid(True)
        subplot[name] = ax

    fig.subplots_adjust(bottom=0.15)

    if 0:
        fig = pylab.figure()
        figtitle = options.kalman_filename
        pylab.figtext(0, 0, figtitle)

        ax = fig.add_subplot(1, 1, 1)
        subplot["vel_hist"] = ax
        ax.grid(True)

    line2obj_id = plot_timeseries(subplot=subplot, options=options,)

    class MyPickObj(object):
        def __init__(self, line2obj_id):
            self.line2obj_id = line2obj_id
            self.obj_ids = []

        def onpick(self, event):
            if isinstance(event.artist, matplotlib.lines.Line2D):
                thisline = event.artist
                obj_id = self.line2obj_id[thisline]
                if obj_id not in self.obj_ids:
                    self.obj_ids.append(obj_id)
                    self.obj_ids.sort()
                    print("picked", obj_id)
                    print("all:")
                    print(self.obj_ids)
                ## else:
                ##     print '(already had obj_id %d)'%obj_id

        def on_key_press(self, event):
            # print 'received key',repr(event.key)
            if event.key == "c":
                del self.obj_ids[:]
                print(self.obj_ids)
            if event.key == "l":
                del self.obj_ids[-1]
                print(self.obj_ids)
            # print 'all:'
            # print self.obj_ids

    if options.save_fig is not None:
        pylab.savefig(options.save_fig)
    else:
        pick_receiver = MyPickObj(line2obj_id)
        fig.canvas.mpl_connect("pick_event", pick_receiver.onpick)
        fig.canvas.mpl_connect("key_press_event", pick_receiver.on_key_press)
        pylab.show()


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    analysis_options.add_common_options(parser)

    parser.add_option(
        "--frames",
        action="store_true",
        help="plot horizontal axis in frame number (not seconds)",
        default=False,
    )

    parser.add_option(
        "--fuse",
        action="store_true",
        help="fuse object ids corresponding to a single fly (requires stim-xml fanout)",
        default=False,
    )

    parser.add_option(
        "--P55",
        action="store_true",
        help="plot P[5,5] component of covariance matrix",
        default=False,
    )

    parser.add_option(
        "--timestamp-file",
        type="string",
        help="file with data2d_distorted table to get timestamps",
    )

    parser.add_option(
        "--save-fig",
        type="string",
        default=None,
        help="path name of figure to save (exits script " "immediately after save)",
    )

    parser.add_option(
        "--ori-qual",
        type="float",
        default=None,
        help=("minimum orientation quality to use"),
    )

    parser.add_option(
        "--smooth-orientations",
        action="store_true",
        help="if displaying orientations, use smoothed data",
        default=False,
    )

    parser.add_option(
        "--show-3d-orientations",
        action="store_true",
        help="show orientation data?",
        default=False,
    )

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args):
        parser.print_help()
        return

    doit(options=options,)


if __name__ == "__main__":
    main()
