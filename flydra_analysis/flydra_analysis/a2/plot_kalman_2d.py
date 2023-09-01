# see a2/save_movies_overlay.py
from __future__ import division, with_statement
from __future__ import print_function
import numpy
import numpy as np
import warnings
from numpy import nan, pi
import tables as PT
import tables.flavor

tables.flavor.restrict_flavors(keep=["numpy"])  # ensure pytables 2.x
import tables
import contextlib
import sys, os
from optparse import OptionParser
import matplotlib
import matplotlib.cm as cm
import matplotlib.collections as collections
import pylab

import flydra_core.reconstruct
import flydra_analysis.analysis.result_utils as result_utils
import flydra_core.kalman.flydra_kalman_utils
import flydra_analysis.a2.xml_stimulus as xml_stimulus
import flydra_analysis.a2.core_analysis as core_analysis

KalmanEstimatesVelOnly = flydra_core.kalman.flydra_kalman_utils.KalmanEstimatesVelOnly


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
    return fig.add_axes([left, bottom, w, h], autoscale_on=False)


class ShowIt(object):
    """
    handle interaction with the user

    It allows the user to indicate 2D points from multiple cameras
    that are gathered into a 2D point list. Once satisfied with the
    contents of this 2D point list, the best 3D intersection of the
    rays from each camera center to the 2D point may be
    calculated. Once such as 3D point is calculated, it is appended to
    the 3D point list, which may be displayed to the screen or saved
    to an .h5 file.

    The implementation is done as a set of key-bindings on top of the
    standard Matplotlib GUI.

    """

    def __init__(self):
        self.subplot_by_cam_id = {}
        self.reconstructor = None
        self.to_del = []
        self.cam_ids_and_points2d = []
        self.points3d = []

    def find_cam_id(self, ax):
        found = False
        for cam_id, axtest in self.subplot_by_cam_id.items():
            if ax is axtest:
                found = True
                break
        if not found:
            raise RuntimeError("event in unknown axes")
        return cam_id

    def on_key_press(self, event):
        """Process a key press

        hotkeys::

          ? - print help
          x - pick a new point and add it to the 2D point list
          i - intersect picked points in the 2D point list
          c - clear the 2D point list
          a - all intersected 3D points are printed to console
          h - save all intersected 3D points to 'points.h5'
        """

        print("received key", repr(event.key))

        if event.key == "c":
            del self.cam_ids_and_points2d[:]
            for ax, line in self.to_del:
                ax.lines.remove(line)
            self.to_del = []
            pylab.draw()

        elif event.key == "i":
            if self.reconstructor is None:
                return

            X = self.reconstructor.find3d(
                self.cam_ids_and_points2d,
                return_X_coords=True,
                return_line_coords=False,
            )
            self.points3d.append(X)

            print("maximum liklihood intersection:")
            print(repr(X))
            if 1:
                print("reprojection errors:")
                for (cam_id, value_tuple) in self.cam_ids_and_points2d:
                    newx, newy = self.reconstructor.find2d(cam_id, X, distorted=True)
                    # origx,origy=self.reconstructor.undistort(cam_id, value_tuple[:2] )
                    origx, origy = value_tuple[:2]
                    reproj_error = numpy.sqrt((newx - origx) ** 2 + (newy - origy) ** 2)
                    print("  %s: %.1f" % (cam_id, reproj_error))
                print()

            for cam_id, ax in self.subplot_by_cam_id.items():
                newx, newy = self.reconstructor.find2d(cam_id, X, distorted=True)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.plot([newx], [newy], "co", ms=5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            pylab.draw()

        elif event.key == "x":
            # new point -- project onto other images

            if not event.inaxes:
                print("not in axes -- nothing to do")
                return

            ax = event.inaxes  # the axes instance
            cam_id = self.find_cam_id(ax)
            print("click on", cam_id)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            (line,) = ax.plot([event.xdata], [event.ydata], "bx")
            self.to_del.append((ax, line))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if self.reconstructor is None:
                print("no reconstructor... cannot plot projection")
                return

            x, y = self.reconstructor.undistort(cam_id, [event.xdata, event.ydata])
            self.cam_ids_and_points2d.append((cam_id, (x, y)))
            line3d = self.reconstructor.get_projected_line_from_2d(cam_id, [x, y])

            cam_ids = self.subplot_by_cam_id.keys()
            cam_ids.sort()

            for other_cam_id in cam_ids:
                if other_cam_id == cam_id:
                    continue
                xs, ys = self.reconstructor.get_distorted_line_segments(
                    other_cam_id, line3d
                )  # these are distorted
                ax = self.subplot_by_cam_id[other_cam_id]
                # print xs
                # print ys
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                (line,) = ax.plot(xs, ys, "b-")
                self.to_del.append((ax, line))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            pylab.draw()
        elif event.key == "?":
            sys.stdout.write(
                """
%s
current list of 2D points
-------------------------
"""
                % self.on_key_press.__doc__
            )
            for cam_id, (x, y) in self.cam_ids_and_points2d:
                sys.stdout.write("%s: %s %s\n" % (cam_id, x, y))
            sys.stdout.write("\n")
        elif event.key == "a":

            def arrstr(arr):
                return "[ " + ", ".join([repr(elem) for elem in arr]) + " ]"

            sys.stdout.write("---------- 3D points so far\n")
            fd = sys.stdout
            fd.write("[\n  ")
            fd.write(",\n  ".join([arrstr(pt) for pt in self.points3d]))
            fd.write("\n]\n")
            sys.stdout.write("---------- \n")
        elif event.key == "h":
            fname = os.path.abspath("points.h5")
            if os.path.exists(fname):
                raise RuntimeError("will not overwrite file %s" % fname)
            if self.reconstructor is None:
                raise RuntimeError("will not save .h5 file without 3D data")

            with contextlib.closing(tables.open_file(fname, mode="w")) as h5file:
                self.reconstructor.save_to_h5file(h5file)
                ct = h5file.create_table  # shorthand
                root = h5file.root  # shorthand
                h5data3d = ct(
                    root, "kalman_estimates", KalmanEstimatesVelOnly, "3d data"
                )

                row = h5data3d.row
                for i, pt in enumerate(self.points3d):
                    row["obj_id"] = 0
                    row["frame"] = i
                    row["timestamp"] = i
                    row["x"], row["y"], row["z"] = pt
                    row["xvel"], row["yvel"], row["zvel"] = 0, 0, 0
                    row.append()
            sys.stdout.write(
                "saved %d points to file %s\n" % (len(self.points3d), fname)
            )

    def show_it(
        self,
        fig,
        filename,
        kalman_filename=None,
        frame_start=None,
        frame_stop=None,
        show_nth_frame=None,
        obj_only=None,
        reconstructor_filename=None,
        options=None,
    ):

        if show_nth_frame == 0:
            show_nth_frame = None

        results = result_utils.get_results(filename, mode="r")
        opened_kresults = False
        kresults = None
        if kalman_filename is not None:
            if os.path.abspath(kalman_filename) == os.path.abspath(filename):
                kresults = results
            else:
                kresults = PT.open_file(kalman_filename, mode="r")
                opened_kresults = True

            # copied from plot_timeseries_2d_3d.py
            ca = core_analysis.get_global_CachingAnalyzer()
            (
                xxobj_ids,
                xxuse_obj_ids,
                xxis_mat_file,
                xxdata_file,
                extra,
            ) = ca.initial_file_load(kalman_filename)
            fps = extra["frames_per_second"]
            dynamic_model_name = None
            if dynamic_model_name is None:
                dynamic_model_name = extra.get("dynamic_model_name", None)
                if dynamic_model_name is None:
                    dynamic_model_name = dynamic_models.DEFAULT_MODEL
                    warnings.warn(
                        'no dynamic model specified, using "%s"' % dynamic_model_name
                    )
                else:
                    print(
                        'detected file loaded with dynamic model "%s"'
                        % dynamic_model_name
                    )
                if dynamic_model_name.startswith("EKF "):
                    dynamic_model_name = dynamic_model_name[4:]
                print(
                    '  for smoothing, will use dynamic model "%s"' % dynamic_model_name
                )

        if hasattr(results.root, "images"):
            img_table = results.root.images
        else:
            img_table = None

        reconstructor_source = None
        if reconstructor_filename is None:
            if kresults is not None:
                reconstructor_source = kresults
            elif hasattr(results.root, "calibration"):
                reconstructor_source = results
            else:
                reconstructor_source = None
        else:
            if os.path.abspath(reconstructor_filename) == os.path.abspath(filename):
                reconstructor_source = results
            elif (kalman_filename is not None) and (
                os.path.abspath(reconstructor_filename)
                == os.path.abspath(kalman_filename)
            ):
                reconstructor_source = kresults
            else:
                reconstructor_source = reconstructor_filename

        if reconstructor_source is not None:
            self.reconstructor = flydra_core.reconstruct.Reconstructor(
                reconstructor_source
            )

        if options.stim_xml:
            file_timestamp = results.filename[4:19]
            stim_xml = xml_stimulus.xml_stimulus_from_filename(
                options.stim_xml, timestamp_string=file_timestamp
            )
            if self.reconstructor is not None:
                stim_xml.verify_reconstructor(self.reconstructor)

        if self.reconstructor is not None:
            self.reconstructor = self.reconstructor.get_scaled()

        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)

        data2d = results.root.data2d_distorted  # make sure we have 2d data table

        print("reading frames...")
        frames = data2d.read(field="frame")
        print("OK")

        if frame_start is not None:
            print("selecting frames after start")
            # after_start = data2d.get_where_list( 'frame>=frame_start')
            after_start = numpy.nonzero(frames >= frame_start)[0]
        else:
            after_start = None

        if frame_stop is not None:
            print("selecting frames before stop")
            # before_stop = data2d.get_where_list( 'frame<=frame_stop')
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
        frames = frames[use_idxs]  # data2d.read_coordinates( use_idxs, field='frame')
        if len(frames):
            print(
                "frame range: %d - %d (%d frames total)"
                % (frames[0], frames[-1], len(frames))
            )
        camns = data2d.read(field="camn")
        camns = camns[use_idxs]
        # camns = data2d.read_coordinates( use_idxs, field='camn')
        unique_camns = numpy.unique(camns)
        unique_cam_ids = list(set([camn2cam_id[camn] for camn in unique_camns]))
        unique_cam_ids.sort()
        print("%d cameras with data" % (len(unique_cam_ids),))

        # plot all cameras, not just those with data
        all_cam_ids = cam_id2camns.keys()
        all_cam_ids.sort()
        unique_cam_ids = all_cam_ids

        if len(unique_cam_ids) == 1:
            n_rows = 1
            n_cols = 1
        elif len(unique_cam_ids) <= 6:
            n_rows = 2
            n_cols = 3
        elif len(unique_cam_ids) <= 12:
            n_rows = 3
            n_cols = 4
        else:
            n_rows = 4
            n_cols = int(math.ceil(len(unique_cam_ids) / n_rows))

        for i, cam_id in enumerate(unique_cam_ids):
            ax = auto_subplot(fig, i, n_rows=n_rows, n_cols=n_cols)
            ax.set_title("%s: %s" % (cam_id, str(cam_id2camns[cam_id])))
            ##        ax.set_xticks([])
            ##        ax.set_yticks([])
            ax.this_minx = np.inf
            ax.this_maxx = -np.inf
            ax.this_miny = np.inf
            ax.this_maxy = -np.inf
            self.subplot_by_cam_id[cam_id] = ax

        for cam_id in unique_cam_ids:
            ax = self.subplot_by_cam_id[cam_id]
            if img_table is not None:
                bg_arr_h5 = getattr(img_table, cam_id)
                bg_arr = bg_arr_h5.read()
                ax.imshow(bg_arr.squeeze(), origin="lower", cmap=cm.pink)
                ax.set_autoscale_on(True)
                ax.autoscale_view()
                pylab.draw()
                ax.set_autoscale_on(False)

            if self.reconstructor is not None:
                if cam_id in self.reconstructor.get_cam_ids():
                    res = self.reconstructor.get_resolution(cam_id)
                    ax.set_xlim([0, res[0]])
                    ax.set_ylim([res[1], 0])

            if options.stim_xml is not None:
                stim_xml.plot_stim_over_distorted_image(ax, cam_id)
        for camn in unique_camns:
            cam_id = camn2cam_id[camn]
            ax = self.subplot_by_cam_id[cam_id]
            this_camn_idxs = use_idxs[camns == camn]

            xs = data2d.read_coordinates(this_camn_idxs, field="x")

            valid_idx = numpy.nonzero(~numpy.isnan(xs))[0]
            if not len(valid_idx):
                continue
            ys = data2d.read_coordinates(this_camn_idxs, field="y")
            if options.show_orientation:
                slope = data2d.read_coordinates(this_camn_idxs, field="slope")

            idx_first_valid = valid_idx[0]
            idx_last_valid = valid_idx[-1]
            tmp_frames = data2d.read_coordinates(this_camn_idxs, field="frame")

            ax.plot(
                [xs[idx_first_valid]], [ys[idx_first_valid]], "ro", label="first point"
            )

            ax.this_minx = min(np.min(xs[valid_idx]), ax.this_minx)
            ax.this_maxx = max(np.max(xs[valid_idx]), ax.this_maxx)

            ax.this_miny = min(np.min(ys[valid_idx]), ax.this_miny)
            ax.this_maxy = max(np.max(ys[valid_idx]), ax.this_maxy)

            ax.plot(xs[valid_idx], ys[valid_idx], "g.", label="all points")

            if options.show_orientation:
                angle = np.arctan(slope)
                r = 20.0
                dx = r * np.cos(angle)
                dy = r * np.sin(angle)
                x0 = xs - dx
                x1 = xs + dx
                y0 = ys - dy
                y1 = ys + dy
                segs = []
                for i in valid_idx:
                    segs.append(((x0[i], y0[i]), (x1[i], y1[i])))
                line_segments = collections.LineCollection(
                    segs, linewidths=[1], colors=[(0, 1, 0)],
                )
                ax.add_collection(line_segments)

            ax.plot(
                [xs[idx_last_valid]], [ys[idx_last_valid]], "bo", label="first point"
            )

            if show_nth_frame is not None:
                for i, f in enumerate(tmp_frames):
                    if f % show_nth_frame == 0:
                        ax.text(xs[i], ys[i], "%d" % (f,))

            if 0:
                for x, y, frame in zip(xs[::5], ys[::5], tmp_frames[::5]):
                    ax.text(x, y, "%d" % (frame,))

        fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        if options.autozoom:
            for cam_id in self.subplot_by_cam_id.keys():
                ax = self.subplot_by_cam_id[cam_id]
                ax.set_xlim((ax.this_minx - 10, ax.this_maxx + 10))
                ax.set_ylim((ax.this_miny - 10, ax.this_maxy + 10))

        if options.save_fig:
            for cam_id in self.subplot_by_cam_id.keys():
                ax = self.subplot_by_cam_id[cam_id]
                ax.set_xticks([])
                ax.set_yticks([])

        if kalman_filename is None:
            return

        if 0:
            # Do same as above for Kalman-filtered data

            kobs = kresults.root.ML_estimates
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

            obj_ids = kobs.read(field="obj_id")[k_use_idxs]
            obs_2d_idxs = kobs.read(field="obs_2d_idx")[k_use_idxs]
            kframes = kframes[k_use_idxs]

            kobs_2d = kresults.root.ML_estimates_2d_idxs
            xys_by_obj_id = {}
            for obj_id, kframe, obs_2d_idx in zip(obj_ids, kframes, obs_2d_idxs):
                if obj_only is not None:
                    if obj_id not in obj_only:
                        continue

                obs_2d_idx_find = int(
                    obs_2d_idx
                )  # XXX grr, why can't pytables do this?
                obj_id_save = int(obj_id)  # convert from possible numpy scalar
                xys_by_cam_id = xys_by_obj_id.setdefault(obj_id_save, {})
                kobs_2d_data = kobs_2d.read(
                    start=obs_2d_idx_find, stop=obs_2d_idx_find + 1
                )
                assert len(kobs_2d_data) == 1
                kobs_2d_data = kobs_2d_data[0]
                this_camns = kobs_2d_data[0::2]
                this_camn_idxs = kobs_2d_data[1::2]

                this_use_idxs = use_idxs[frames == kframe]

                d2d = data2d.read_coordinates(this_use_idxs)
                for this_camn, this_camn_idx in zip(this_camns, this_camn_idxs):
                    this_idxs_tmp = numpy.nonzero(d2d["camn"] == this_camn)[0]
                    this_camn_d2d = d2d[d2d["camn"] == this_camn]
                    found = False
                    for this_row in this_camn_d2d:  # XXX could be sped up
                        if this_row["frame_pt_idx"] == this_camn_idx:
                            found = True
                            break
                    if not found:
                        if 1:
                            print(
                                "WARNING:point not found in data -- 3D data starts before 2D I guess."
                            )
                            continue
                        else:
                            raise RuntimeError("point not found in data!?")
                    this_cam_id = camn2cam_id[this_camn]
                    xys = xys_by_cam_id.setdefault(this_cam_id, ([], []))
                    xys[0].append(this_row["x"])
                    xys[1].append(this_row["y"])

            for obj_id in xys_by_obj_id:
                xys_by_cam_id = xys_by_obj_id[obj_id]
                for cam_id, (xs, ys) in xys_by_cam_id.items():
                    ax = self.subplot_by_cam_id[cam_id]
                    ax.plot(xs, ys, "x-", label="obs: %d" % obj_id)
                    ax.text(xs[0], ys[0], "%d:" % (obj_id,))
                    ax.text(xs[-1], ys[-1], ":%d" % (obj_id,))

        if 1:
            # do for core_analysis smoothed (or not) data

            for obj_id in xxuse_obj_ids:
                try:
                    rows = ca.load_data(
                        obj_id,
                        kalman_filename,
                        use_kalman_smoothing=True,
                        frames_per_second=fps,
                        dynamic_model_name=dynamic_model_name,
                    )
                except core_analysis.NotEnoughDataToSmoothError:
                    warnings.warn(
                        "not enough data to smooth obj_id %d, skipping." % (obj_id,)
                    )
                if frame_start is not None:
                    c1 = rows["frame"] >= frame_start
                else:
                    c1 = np.ones((len(rows),), dtype=np.bool_)
                if frame_stop is not None:
                    c2 = rows["frame"] <= frame_stop
                else:
                    c2 = np.ones((len(rows),), dtype=np.bool_)
                valid = c1 & c2
                rows = rows[valid]
                if len(rows) > 1:
                    X3d = np.array(
                        (rows["x"], rows["y"], rows["z"], np.ones_like(rows["z"]))
                    ).T

                for cam_id in self.subplot_by_cam_id.keys():
                    ax = self.subplot_by_cam_id[cam_id]
                    newx, newy = self.reconstructor.find2d(cam_id, X3d, distorted=True)
                    ax.plot(newx, newy, "-", label="k: %d" % obj_id)

        results.close()
        if opened_kresults:
            kresults.close()


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

    parser.add_option(
        "-k",
        "--kalman-file",
        dest="kalman_filename",
        type="string",
        help="hdf5 file with kalman data to display KALMANFILE",
        metavar="KALMANFILE",
    )

    parser.add_option(
        "-r",
        "--reconstructor",
        dest="reconstructor_path",
        type="string",
        help="calibration/reconstructor path (if not specified, defaults to KALMANFILE)",
        metavar="RECONSTRUCTOR",
    )

    parser.add_option(
        "--start", type="int", help="first frame to plot", metavar="START"
    )

    parser.add_option("--stop", type="int", help="last frame to plot", metavar="STOP")

    parser.add_option(
        "--show-nth-frame",
        type="int",
        dest="show_nth_frame",
        help="show Nth frame number (0=none)",
    )

    parser.add_option(
        "--stim-xml",
        type="string",
        default=None,
        help="name of XML file with stimulus info",
    )

    parser.add_option(
        "--save-fig", type="string", help="save output to filename (disables gui)",
    )

    parser.add_option("--show-orientation", action="store_true", default=False)

    parser.add_option("--autozoom", action="store_true", default=False)

    parser.add_option("--obj-only", type="string")

    (options, args) = parser.parse_args()

    if options.filename is not None:
        args.append(options.filename)

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]

    fig = pylab.figure()
    fig.text(
        0.5,
        1.0,
        "Press '?' over this window to print help to console",
        verticalalignment="top",
        horizontalalignment="center",
    )
    showit = ShowIt()
    showit.show_it(
        fig,
        h5_filename,
        kalman_filename=options.kalman_filename,
        frame_start=options.start,
        frame_stop=options.stop,
        show_nth_frame=options.show_nth_frame,
        obj_only=options.obj_only,
        reconstructor_filename=options.reconstructor_path,
        options=options,
    )
    if options.save_fig is not None:
        print("saving to %s" % options.save_fig)
        pylab.savefig(options.save_fig)
    else:
        pylab.show()


if __name__ == "__main__":
    main()
