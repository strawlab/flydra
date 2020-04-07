# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20080112

from __future__ import division
from __future__ import with_statement
from __future__ import print_function
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
import collections
import flydra_core.reconstruct as reconstruct
import flydra_analysis.analysis.result_utils as result_utils
import matplotlib

rcParams = matplotlib.rcParams
rcParams["xtick.major.pad"] = 10
rcParams["ytick.major.pad"] = 10

from . import core_analysis
import flydra_analysis.a2.xml_stimulus as xml_stimulus
from . import analysis_options
from optparse import OptionParser
from . import densities  # from scikits.learn

import warnings


class Frames2Time:
    def __init__(self, frame0, fps):
        self.f0 = frame0
        self.fps = fps

    def __call__(self, farr):
        f = farr - self.f0
        f2 = f / self.fps
        return f2


class keep_axes_dimensions_if(object):
    def __init__(self, ax, mybool):
        self.ax = ax
        self.mybool = mybool

    def __enter__(self):
        if self.mybool:
            # matplotlib 0.98.3 returned np.array view
            self.xlim = tuple(self.ax.get_xlim())
            self.ylim = tuple(self.ax.get_ylim())

    def __exit__(self, etype, eval, etb):
        if self.mybool:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
        if etype:
            raise eval


def get_covariance(rowi):
    try:
        va = np.array(
            [
                [rowi["P00"], rowi["P01"], rowi["P02"]],
                [rowi["P01"], rowi["P11"], rowi["P12"]],
                [rowi["P02"], rowi["P12"], rowi["P22"]],
            ]
        )
    except IndexError:
        # no off-diagonal elements
        va = np.diag([rowi["P00"], rowi["P11"], rowi["P22"]])  # diagonal elements of P
    return va


def plot_top_and_side_views(
    subplot=None, options=None, obs_mew=None, scale=1.0, units="m",
):
    """
    inputs
    ------
    subplot - a dictionary of matplotlib axes instances with keys 'xy' and/or 'xz'
    fps - the framerate of the data
    """
    assert subplot is not None

    assert options is not None

    if not hasattr(options, "show_track_ends"):
        options.show_track_ends = False

    if not hasattr(options, "unicolor"):
        options.unicolor = False

    if not hasattr(options, "show_landing"):
        options.show_landing = False

    kalman_filename = options.kalman_filename
    fps = options.fps
    dynamic_model = options.dynamic_model
    use_kalman_smoothing = options.use_kalman_smoothing

    if not hasattr(options, "ellipsoids"):
        options.ellipsoids = False

    if not hasattr(options, "show_observations"):
        options.show_observations = False

    if not hasattr(options, "markersize"):
        options.markersize = 0.5

    if options.ellipsoids and use_kalman_smoothing:
        warnings.warn(
            "plotting ellipsoids while using Kalman smoothing does not reveal original error estimates"
        )

    assert kalman_filename is not None

    start = options.start
    stop = options.stop
    obj_only = options.obj_only

    if not use_kalman_smoothing:
        if dynamic_model is not None:
            print(
                "ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model options (--dynamic-model)",
                file=sys.stderr,
            )
            sys.exit(1)

    ca = core_analysis.get_global_CachingAnalyzer()

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
            kalman_filename
        )

    if not is_mat_file:
        mat_data = None

        if fps is None:
            fps = result_utils.get_fps(data_file, fail_on_error=False)

        if fps is None:
            fps = 100.0
            warnings.warn("Setting fps to default value of %f" % fps)
        reconstructor = reconstruct.Reconstructor(data_file)
    else:
        reconstructor = None

    if options.stim_xml:
        file_timestamp = data_file.filename[4:19]
        stim_xml = xml_stimulus.xml_stimulus_from_filename(
            options.stim_xml, timestamp_string=file_timestamp,
        )
        try:
            fanout = xml_stimulus.xml_fanout_from_filename(options.stim_xml)
        except xml_stimulus.WrongXMLTypeError:
            walking_start_stops = []
        else:
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
            stim_xml = fanout.get_stimulus_for_timestamp(
                timestamp_string=file_timestamp
            )
        if stim_xml.has_reconstructor():
            stim_xml.verify_reconstructor(reconstructor)
    else:
        walking_start_stops = []

    if dynamic_model is None:
        dynamic_model = extra.get("dynamic_model_name", None)

    if dynamic_model is None:
        if use_kalman_smoothing:
            warnings.warn(
                "no kalman smoothing will be performed because no "
                "dynamic model specified or found."
            )
            use_kalman_smoothing = False
    else:
        print('detected file loaded with dynamic model "%s"' % dynamic_model)
        if use_kalman_smoothing:
            if dynamic_model.startswith("EKF "):
                dynamic_model = dynamic_model[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model)

    subplots = subplot.keys()
    subplots.sort()  # ensure consistency across runs

    dt = 1.0 / fps

    if obj_only is not None:
        use_obj_ids = [i for i in use_obj_ids if i in obj_only]

    subplot["xy"].set_aspect("equal")
    subplot["xz"].set_aspect("equal")

    subplot["xy"].set_xlabel("x (%s)" % units)
    subplot["xy"].set_ylabel("y (%s)" % units)

    subplot["xz"].set_xlabel("x (%s)" % units)
    subplot["xz"].set_ylabel("z (%s)" % units)

    if options.stim_xml:
        stim_xml.plot_stim(
            subplot["xy"], projection=xml_stimulus.SimpleOrthographicXYProjection()
        )
        stim_xml.plot_stim(
            subplot["xz"], projection=xml_stimulus.SimpleOrthographicXZProjection()
        )

    allX = {}
    frame0 = None
    results = collections.defaultdict(list)
    for obj_id in use_obj_ids:
        line = None
        ellipse_lines = []
        MLE_line = None
        try:
            kalman_rows = ca.load_data(
                obj_id,
                data_file,
                use_kalman_smoothing=use_kalman_smoothing,
                dynamic_model_name=dynamic_model,
                frames_per_second=fps,
                up_dir=options.up_dir,
            )
        except core_analysis.ObjectIDDataError:
            continue

        if options.show_observations:
            kobs_rows = ca.load_dynamics_free_MLE_position(obj_id, data_file)

        frame = kalman_rows["frame"]
        if options.show_observations:
            frame_obs = kobs_rows["frame"]

        if (start is not None) or (stop is not None):
            valid_cond = numpy.ones(frame.shape, dtype=numpy.bool)
            if options.show_observations:
                valid_obs_cond = np.ones(frame_obs.shape, dtype=numpy.bool)

            if start is not None:
                valid_cond = valid_cond & (frame >= start)
                if options.show_observations:
                    valid_obs_cond = valid_obs_cond & (frame_obs >= start)

            if stop is not None:
                valid_cond = valid_cond & (frame <= stop)
                if options.show_observations:
                    valid_obs_cond = valid_obs_cond & (frame_obs <= stop)

            kalman_rows = kalman_rows[valid_cond]
            if options.show_observations:
                kobs_rows = kobs_rows[valid_obs_cond]
            if not len(kalman_rows):
                continue

        frame = kalman_rows["frame"]

        Xx = kalman_rows["x"]
        Xy = kalman_rows["y"]
        Xz = kalman_rows["z"]

        if options.max_z is not None:
            cond = Xz <= options.max_z

            frame = numpy.ma.masked_where(~cond, frame)
            Xx = numpy.ma.masked_where(~cond, Xx)
            Xy = numpy.ma.masked_where(~cond, Xy)
            Xz = numpy.ma.masked_where(~cond, Xz)
            with keep_axes_dimensions_if(subplot["xz"], options.stim_xml):
                subplot["xz"].axhline(options.max_z)

        kws = {"markersize": options.markersize}

        if options.unicolor:
            kws["color"] = "k"

        landing_idxs = []
        for walkstart, walkstop in walking_start_stops:
            if walkstart in frame:
                tmp_idx = numpy.nonzero(frame == walkstart)[0][0]
                landing_idxs.append(tmp_idx)

        with keep_axes_dimensions_if(subplot["xy"], options.stim_xml):
            (line,) = subplot["xy"].plot(
                Xx * scale, Xy * scale, ".", label="obj %d" % obj_id, **kws
            )
            kws["color"] = line.get_color()
            if options.ellipsoids:
                for i in range(len(Xx)):
                    rowi = kalman_rows[i]
                    mu = [rowi["x"], rowi["y"], rowi["z"]]
                    va = get_covariance(rowi)
                    ellx, elly = densities.gauss_ell(mu, va, [0, 1], 30, 0.39)
                    (ellipse_line,) = subplot["xy"].plot(
                        ellx * scale, elly * scale, color=kws["color"]
                    )
                    ellipse_lines.append(ellipse_line)
            if options.show_track_ends:
                subplot["xy"].plot(
                    [Xx[0] * scale, Xx[-1] * scale],
                    [Xy[0] * scale, Xy[-1] * scale],
                    "cd",
                    ms=6,
                    label="track end",
                )
            if options.show_obj_id:
                subplot["xy"].text(Xx[0] * scale, Xy[0] * scale, str(obj_id))
            if options.show_landing:
                for landing_idx in landing_idxs:
                    subplot["xy"].plot(
                        [Xx[landing_idx] * scale],
                        [Xy[landing_idx] * scale],
                        "rD",
                        ms=10,
                        label="landing",
                    )
            if options.show_observations:
                mykw = {}
                mykw.update(kws)
                mykw["markersize"] *= 5
                mykw["mew"] = obs_mew

                badcond = np.isnan(kobs_rows["x"])
                Xox = np.ma.masked_where(badcond, kobs_rows["x"])
                Xoy = np.ma.masked_where(badcond, kobs_rows["y"])

                (MLE_line,) = subplot["xy"].plot(
                    Xox * scale, Xoy * scale, "x", label="obj %d" % obj_id, **mykw
                )

        with keep_axes_dimensions_if(subplot["xz"], options.stim_xml):
            (line,) = subplot["xz"].plot(
                Xx * scale, Xz * scale, ".", label="obj %d" % obj_id, **kws
            )
            kws["color"] = line.get_color()
            if options.ellipsoids:
                for i in range(len(Xx)):
                    rowi = kalman_rows[i]
                    mu = [rowi["x"], rowi["y"], rowi["z"]]
                    va = get_covariance(rowi)
                    ellx, ellz = densities.gauss_ell(mu, va, [0, 2], 30, 0.39)
                    (ellipse_line,) = subplot["xz"].plot(
                        ellx * scale, ellz * scale, color=kws["color"]
                    )
                    ellipse_lines.append(ellipse_line)

            if options.show_track_ends:
                subplot["xz"].plot(
                    [Xx[0] * scale, Xx[-1] * scale],
                    [Xz[0] * scale, Xz[-1] * scale],
                    "cd",
                    ms=6,
                    label="track end",
                )
            if options.show_obj_id:
                subplot["xz"].text(Xx[0] * scale, Xz[0] * scale, str(obj_id))
            if options.show_landing:
                for landing_idx in landing_idxs:
                    subplot["xz"].plot(
                        [Xx[landing_idx] * scale],
                        [Xz[landing_idx] * scale],
                        "rD",
                        ms=10,
                        label="landing",
                    )
            if options.show_observations:
                mykw = {}
                mykw.update(kws)
                mykw["markersize"] *= 5
                mykw["mew"] = obs_mew

                badcond = np.isnan(kobs_rows["x"])
                Xox = np.ma.masked_where(badcond, kobs_rows["x"])
                Xoz = np.ma.masked_where(badcond, kobs_rows["z"])

                (MLE_line,) = subplot["xz"].plot(
                    Xox * scale, Xoz * scale, "x", label="obj %d" % obj_id, **mykw
                )
        results["lines"].append(line)
        results["ellipse_lines"].extend(ellipse_lines)
        results["MLE_line"].append(MLE_line)
    return results


def doit(options=None,):
    kalman_filename = options.kalman_filename
    import pylab  # do after matplotlib.use() call

    fig = pylab.figure(figsize=(5, 8))
    figtitle = kalman_filename.split(".")[0]
    pylab.figtext(0, 0, figtitle)

    ax = None
    subplot = {}
    subplots = ["xy", "xz"]
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots), 1, i + 1)  # ,sharex=ax)
        subplot[name] = ax

    if options.up_dir is not None:
        options.up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        options.up_dir = None

    plot_top_and_side_views(
        subplot=subplot, options=options,
    )

    pylab.show()


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    analysis_options.add_common_options(parser)
    parser.add_option("--ellipsoids", action="store_true", default=False)
    parser.add_option("--show-obj-id", action="store_true", default=False)
    parser.add_option("--show-observations", action="store_true", default=False)
    parser.add_option("--markersize", type="float", default=0.5)
    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args):
        parser.print_help()
        return

    doit(options=options,)


if __name__ == "__main__":
    main()
