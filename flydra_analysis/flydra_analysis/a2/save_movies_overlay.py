# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20070112

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pkg_resources

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import os, sys, math

import numpy
import numpy as np
import tables as PT
from optparse import OptionParser
import flydra_core.reconstruct as reconstruct
import motmot.ufmf.ufmf as ufmf
import flydra_analysis.a2.utils as utils
import flydra_analysis.a2.aggdraw_coord_shifter as aggdraw_coord_shifter

PLOT = "image"

if PLOT == "image":
    import PIL.Image
    import aggdraw
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import flydra_analysis.analysis.result_utils as result_utils
import progressbar
from . import core_analysis

import warnings
import datetime


def ensure_minsize_image(arr, xxx_todo_changeme, fill=0):
    (h, w) = xxx_todo_changeme
    if (arr.shape[0] < h) or (arr.shape[1] < w):
        arr_new = numpy.ones((h, w), dtype=arr.dtype) * fill
        arr_new[: arr.shape[0], : arr.shape[1]] = arr
        arr = arr_new
    return arr


class KObsRowCacher:
    def __init__(self, h5):
        self.h5 = h5
        self.all_rows_obj_ids = h5.root.ML_estimates.read(field="obj_id")
        self.all_rows_frames = h5.root.ML_estimates.read(field="frame")
        self.cache = {}

    def get(self, obj_id):
        if obj_id in self.cache:
            return self.cache[obj_id]
        else:
            cond = self.all_rows_obj_ids == obj_id
            frames = self.all_rows_frames[cond]
            try:
                qualities = core_analysis.compute_ori_quality(self.h5, frames, obj_id)
            except Exception as err:
                print()
                print("len(frames)", len(frames))
                # this is probably missing data. no time to debug now.
                warnings.warn("ignoring pytables error %s" % err)
                qualities = np.zeros(frames.shape)
            results = (frames, qualities)
            self.cache[obj_id] = results
        return results


def doit(
    fmf_filename=None,
    h5_filename=None,
    kalman_filename=None,
    fps=None,
    use_kalman_smoothing=True,
    dynamic_model=None,
    start=None,
    stop=None,
    style="debug",
    blank=None,
    do_zoom=False,
    do_zoom_diff=False,
    up_dir=None,
    options=None,
):

    R = None  # initially set to none

    if options.flip_y and options.rotate_180:
        raise ValueError("can use flip_y or rotate_180, but not both")

    if do_zoom_diff and do_zoom:
        raise ValueError("can use do_zoom or do_zoom_diff, but not both")

    styles = ["debug", "pretty", "prettier", "blank", "prettier-MLE-slope"]
    if style not in styles:
        raise ValueError('style ("%s") is not one of %s' % (style, str(styles)))

    if options.debug_ori_pickle is not None:
        print("options.debug_ori_pickle", options.debug_ori_pickle)
        import pickle

        fd = open(options.debug_ori_pickle, mode="rb")
        used_camn_dict = pickle.load(fd)
        fd.close()

    if not use_kalman_smoothing:
        if (fps is not None) or (dynamic_model is not None):
            print(
                "ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting fps and dynamic model options (--fps and --dynamic-model)",
                file=sys.stderr,
            )
            sys.exit(1)

    if fmf_filename.endswith(".ufmf"):
        if options.ufmf_white_background:
            kwargs = dict(white_background=True, use_conventional_named_mean_fmf=False,)
            assert options.ufmf_abs_diff == False
        else:
            kwargs = dict(use_conventional_named_mean_fmf=True)
            if options.ufmf_abs_diff:
                kwargs["abs_diff"] = True
        fmf = ufmf.FlyMovieEmulator(fmf_filename, **kwargs)
    else:
        fmf = FMF.FlyMovie(fmf_filename)
    fmf_timestamps = fmf.get_all_timestamps()
    h5 = PT.open_file(h5_filename, mode="r")

    bg_fmf_filename = os.path.splitext(fmf_filename)[0] + "_mean.fmf"
    cmp_fmf_filename = os.path.splitext(fmf_filename)[0] + "_sumsqf.fmf"
    if not os.path.exists(cmp_fmf_filename):
        cmp_fmf_filename = (
            os.path.splitext(fmf_filename)[0] + "_mean2.fmf"
        )  # old version

    bg_OK = False
    if os.path.exists(bg_fmf_filename):
        bg_OK = True
        try:
            bg_fmf = FMF.FlyMovie(bg_fmf_filename)
            cmp_fmf = FMF.FlyMovie(cmp_fmf_filename)
        except FMF.InvalidMovieFileException as err:
            bg_OK = False

    if bg_OK:
        bg_fmf_timestamps = bg_fmf.get_all_timestamps()
        cmp_fmf_timestamps = cmp_fmf.get_all_timestamps()
        assert numpy.all(
            (bg_fmf_timestamps[1:] - bg_fmf_timestamps[:-1]) > 0
        )  # ascending
        assert numpy.all((bg_fmf_timestamps == cmp_fmf_timestamps))
        fmf2bg = bg_fmf_timestamps.searchsorted(fmf_timestamps, side="right") - 1
    else:
        print("Not loading background movie - it does not exist or it is invalid.")
        bg_fmf = None
        cmp_fmf = None
        bg_fmf_timestamps = None
        cmp_fmf_timestamps = None
        fmf2bg = None

    if fps is None:
        fps = result_utils.get_fps(h5)

    ca = core_analysis.get_global_CachingAnalyzer()

    if blank is not None:
        # use frame number as "blank image"
        fmf.seek(blank)
        blank_image, blank_timestamp = fmf.get_next_frame()
        fmf.seek(0)
    else:
        blank_image = 255 * numpy.ones(
            (fmf.get_height(), fmf.get_width()), dtype=numpy.uint8
        )

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
            kalman_filename
        )

        if dynamic_model is None:
            dynamic_model = extra["dynamic_model_name"]
            print('detected file loaded with dynamic model "%s"' % dynamic_model)
            if dynamic_model.startswith("EKF "):
                dynamic_model = dynamic_model[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model)

        if is_mat_file:
            raise ValueError(
                "cannot use .mat file for kalman_filename "
                "because it is missing the reconstructor "
                "and ability to get framenumbers"
            )

        R = reconstruct.Reconstructor(data_file)

        print("loading frame numbers for kalman objects (estimates)")
        kalman_rows = []
        if options.obj_only is not None:
            use_obj_ids = options.obj_only
        for obj_id in use_obj_ids:
            try:
                my_rows = ca.load_data(
                    obj_id,
                    data_file,
                    use_kalman_smoothing=use_kalman_smoothing,
                    dynamic_model_name=dynamic_model,
                    frames_per_second=fps,
                    return_smoothed_directions=options.smooth_orientations,
                    up_dir=up_dir,
                    min_ori_quality_required=options.ori_qual,
                )
            except core_analysis.NotEnoughDataToSmoothError:
                print("not enough data to smooth for obj_id %d, skipping..." % obj_id)
            else:
                kalman_rows.append(my_rows)

        if len(kalman_rows):
            kalman_rows = numpy.concatenate(kalman_rows)
            kalman_3d_frame = kalman_rows["frame"]

            print("loading frame numbers for kalman objects (observations)")
            kobs_rows = []
            for obj_id in use_obj_ids:
                my_rows = ca.load_dynamics_free_MLE_position(
                    obj_id, data_file, min_ori_quality_required=options.ori_qual,
                )
                kobs_rows.append(my_rows)
            kobs_rows = numpy.concatenate(kobs_rows)
            kobs_3d_frame = kobs_rows["frame"]
            print("loaded")
        else:
            print("WARNING: kalman filename specified, but objects found")
            kalman_filename = None

    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

    n = 0
    for cam_id in cam_id2camns.keys():
        if cam_id in fmf_filename:
            n += 1
            found_cam_id = cam_id
    if n != 1:
        print(
            "Could not automatically determine cam_id from fmf_filename. Exiting",
            file=sys.stderr,
        )
        h5.close()
        sys.exit(1)
    cam_id = found_cam_id
    my_camns = cam_id2camns[cam_id]

    if kalman_filename is not None:
        cam_center_meters = R.get_camera_center(cam_id)

    remote_timestamps = h5.root.data2d_distorted.read(field="timestamp")
    camns = h5.root.data2d_distorted.read(field="camn")
    # find rows for all camns for this cam_id
    all_camn_cond = None
    for camn in my_camns:
        cond = camn == camns
        if all_camn_cond is None:
            all_camn_cond = cond
        else:
            all_camn_cond = all_camn_cond | cond
    camn_idx = numpy.nonzero(all_camn_cond)[0]
    cam_remote_timestamps = remote_timestamps[camn_idx]
    cam_remote_timestamps_find = utils.FastFinder(cam_remote_timestamps)

    cur_bg_idx = None

    # find frame correspondence
    print("Finding frame correspondence... ", end=" ")
    sys.stdout.flush()
    frame_match_h5 = None
    first_match = None
    for fmf_fno, timestamp in enumerate(fmf_timestamps):
        timestamp_idx = numpy.nonzero(timestamp == remote_timestamps)[0]
        # print repr(timestamp), repr(timestamp_idx)
        idxs = numpy.intersect1d(camn_idx, timestamp_idx)
        if len(idxs):
            rows = h5.root.data2d_distorted.read_coordinates(idxs)
            frame_match_h5 = rows["frame"][0]
            if start is None:
                start = frame_match_h5
            if first_match is None:
                first_match = frame_match_h5
            if stop is not None:
                break

    if frame_match_h5 is None:
        print(
            "ERROR: no timestamp corresponding to .fmf '%s' for %s in '%s'"
            % (fmf_filename, cam_id, h5_filename),
            file=sys.stderr,
        )
        h5.close()
        sys.exit(1)

    print("done.")

    if stop is None:
        stop = frame_match_h5
        last_match = frame_match_h5
        print(
            "Frames in both the .fmf movie and the .h5 data file are in range %d - %d."
            % (first_match, last_match)
        )
    else:
        print(
            "Frames in both the .fmf movie and the .h5 data file start at %d."
            % (first_match,)
        )

    if 1:
        # if first_match is not None and last_match is not None:
        h5frames = h5.root.data2d_distorted.read(field="frame")[:]
        print("The .h5 file has frames %d - %d." % (h5frames.min(), h5frames.max()))
        del h5frames

    if start > stop:
        print(
            "ERROR: start (frame %d) is after stop (frame %d)!" % (start, stop),
            file=sys.stderr,
        )
        h5.close()
        sys.exit(1)

    fmf_frame2h5_frame = frame_match_h5 - fmf_fno

    if PLOT == "image":
        # colors from: http://jfly.iam.u-tokyo.ac.jp/color/index.html#pallet

        cb_orange = (230, 159, 0)
        cb_blue = (0, 114, 178)
        cb_vermillion = (213, 94, 0)
        cb_blue_green = (0, 158, 115)

        # Not from that pallette:
        cb_white = (255, 255, 255)

        font2d = aggdraw.Font(
            cb_blue, "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", size=20
        )
        pen2d = aggdraw.Pen(cb_blue, width=1)
        pen2d_bold = aggdraw.Pen(cb_blue, width=3)

        pen3d = aggdraw.Pen(cb_orange, width=2)

        pen3d_raw = aggdraw.Pen(cb_orange, width=1)
        font3d = aggdraw.Font(
            cb_orange, "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
        )

        pen_zoomed = pen3d
        font_zoomed = aggdraw.Font(
            cb_orange, "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", size=20
        )

        pen_obs = aggdraw.Pen(cb_blue_green, width=2)
        font_obs = aggdraw.Font(
            cb_blue_green, "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
        )

    print("loading frame information...")
    # step through .fmf file to get map of h5frame <-> fmfframe
    mymap = {}
    all_frame = h5.root.data2d_distorted.read(field="frame")
    cam_all_frame = all_frame[camn_idx]

    if not options.no_progress:
        widgets = [
            "stage 1 of 2: ",
            cam_id,
            " ",
            progressbar.Percentage(),
            " ",
            progressbar.Bar(),
            " ",
            progressbar.ETA(),
        ]

        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(fmf_timestamps)
        ).start()
    for fmf_fno, fmf_timestamp in enumerate(fmf_timestamps):
        if not options.no_progress:
            pbar.update(fmf_fno)
        # idxs = numpy.nonzero(cam_remote_timestamps==fmf_timestamp)[0]
        idxs = cam_remote_timestamps_find.get_idxs_of_equal(fmf_timestamp)
        if len(idxs):
            this_frame = cam_all_frame[idxs]
            real_h5_frame = int(this_frame[0])
            # we only should have one frame here
            assert numpy.all(real_h5_frame == this_frame)
            mymap[real_h5_frame] = fmf_fno
    if not options.no_progress:
        pbar.finish()
        print("done loading frame information.")

    kobs_row_cacher = KObsRowCacher(data_file)

    print("start, stop", start, stop)
    if not options.no_progress:
        widgets[0] = "stage 2 of 2: "
        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=(stop - start + 1)
        ).start()
    for h5_frame in range(start, stop + 1):
        if not options.no_progress:
            pbar.update(h5_frame - start)
        mainbrain_timestamp = numpy.nan
        idxs = []
        try:
            fmf_fno = mymap[h5_frame]
            fmf_timestamp = fmf_timestamps[fmf_fno]

            # get frame
            fmf.seek(fmf_fno)
            frame, fmf_timestamp2 = fmf.get_next_frame()
            assert (
                fmf_timestamp == fmf_timestamp2
            ), "timestamps of .fmf frames don't match"

            # get bg frame
            if fmf2bg is not None:
                bg_idx = fmf2bg[fmf_fno]
                if cur_bg_idx != bg_idx:
                    bg_frame, bg_timestamp = bg_fmf.get_frame(bg_idx)
                    cmp_frame, trash = cmp_fmf.get_frame(bg_idx)
                    cur_bg_idx = bg_idx

            timestamp_idx = numpy.nonzero(fmf_timestamp == remote_timestamps)[0]
            idxs = numpy.intersect1d(camn_idx, timestamp_idx)
            rows = None
            if len(idxs):
                rows = h5.root.data2d_distorted.read_coordinates(idxs)
                mainbrain_timestamp = rows["cam_received_timestamp"][0]

            del fmf_fno
            del fmf_timestamp2
        except KeyError as err:
            frame = blank_image

        if PLOT == "image":
            im = None

        if 1:
            # get 3D estimates data
            kalman_vert_images = []
            kalman_ori_verts_images = []  # for 3D orientation
            kalman_raw_ori_verts_images = []  # for 3D orientation

            if kalman_filename is not None:
                data_3d_idxs = numpy.nonzero(h5_frame == kalman_3d_frame)[0]
                these_3d_rows = kalman_rows[data_3d_idxs]
                line_length = 0.30  # 20 cm total
                for this_3d_row in these_3d_rows:
                    ori_qual_sufficient = True
                    if options.ori_qual is not None:
                        obj_id = this_3d_row["obj_id"]
                        this_obj_frames, qualities = kobs_row_cacher.get(obj_id)
                        tmp_cond = this_obj_frames == this_3d_row["frame"]
                        this_ori_qual = qualities[tmp_cond]
                        if this_ori_qual < options.ori_qual:
                            ori_qual_sufficient = False
                    vert = numpy.array(
                        [this_3d_row["x"], this_3d_row["y"], this_3d_row["z"]]
                    )
                    vert_image = R.find2d(cam_id, vert, distorted=True)
                    P = numpy.array(
                        [this_3d_row["P00"], this_3d_row["P11"], this_3d_row["P22"]]
                    )
                    Pmean = numpy.sqrt(numpy.sum(P ** 2))
                    Pmean_meters = numpy.sqrt(Pmean)
                    kalman_vert_images.append(
                        (vert_image, vert, this_3d_row["obj_id"], Pmean_meters)
                    )

                    if (
                        ori_qual_sufficient
                        and options.body_axis
                        or options.smooth_orientations
                    ):

                        for target, dir_x_name, dir_y_name, dir_z_name in [
                            (kalman_ori_verts_images, "dir_x", "dir_y", "dir_z"),
                            (
                                kalman_raw_ori_verts_images,
                                "rawdir_x",
                                "rawdir_y",
                                "rawdir_z",
                            ),
                        ]:
                            direction = numpy.array(
                                [
                                    this_3d_row[dir_x_name],
                                    this_3d_row[dir_y_name],
                                    this_3d_row[dir_z_name],
                                ]
                            )
                            start_frac, stop_frac = 0.3, 1.0
                            v1 = vert + (start_frac * direction * line_length)
                            v2 = vert + (stop_frac * direction * line_length)
                            u = v2 - v1

                            # plot several verts to deal with camera distortion
                            ori_verts = [
                                v1 + inc * u for inc in numpy.linspace(0, 1.0, 6)
                            ]
                            ori_verts_images = [
                                R.find2d(cam_id, ori_vert, distorted=True)
                                for ori_vert in ori_verts
                            ]
                            target.append(ori_verts_images)

            # get 3D observation data
            kobs_vert_images = []
            kobs_ori_verts_images_a = []  # for 3D orientation
            kobs_ori_verts_images_b = []  # for 3D orientation
            if kalman_filename is not None:
                data_3d_idxs = numpy.nonzero(h5_frame == kobs_3d_frame)[0]
                these_3d_rows = kobs_rows[data_3d_idxs]
                for this_3d_row in these_3d_rows:
                    ori_qual_sufficient = True
                    if options.ori_qual is not None:
                        obj_id = this_3d_row["obj_id"]
                        this_obj_frames, qualities = kobs_row_cacher.get(obj_id)
                        tmp_cond = this_obj_frames == this_3d_row["frame"]
                        this_ori_qual = qualities[tmp_cond]
                        if this_ori_qual < options.ori_qual:
                            ori_qual_sufficient = False
                    vert = numpy.array(
                        [this_3d_row["x"], this_3d_row["y"], this_3d_row["z"]]
                    )
                    if ori_qual_sufficient:
                        line_length = 0.16  # 16 cm total
                        hzline = numpy.array(
                            [
                                this_3d_row["hz_line0"],
                                this_3d_row["hz_line1"],
                                this_3d_row["hz_line2"],
                                this_3d_row["hz_line3"],
                                this_3d_row["hz_line4"],
                                this_3d_row["hz_line5"],
                            ]
                        )
                        direction = reconstruct.line_direction(hzline)
                        for (start_frac, stop_frac, target) in [
                            (-1.0, -0.5, kobs_ori_verts_images_a),
                            (1.0, 0.5, kobs_ori_verts_images_b),
                        ]:
                            v1 = vert + (start_frac * direction * line_length)
                            v2 = vert + (stop_frac * direction * line_length)
                            u = v2 - v1

                            # plot several verts to deal with camera distortion
                            ori_verts = [
                                v1 + inc * u for inc in numpy.linspace(0, 1.0, 5)
                            ]

                            ori_verts_images = [
                                R.find2d(cam_id, ori_vert, distorted=True)
                                for ori_vert in ori_verts
                            ]
                            target.append(ori_verts_images)

                    vert_image = R.find2d(cam_id, vert, distorted=True)
                    obs_2d_idx = this_3d_row["obs_2d_idx"]
                    kobs_2d_data = data_file.root.ML_estimates_2d_idxs[int(obs_2d_idx)]

                    # parse VLArray
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]
                    this_cam_ids = [camn2cam_id[this_camn] for this_camn in this_camns]
                    obs_info = (this_cam_ids, this_camn_idxs)

                    kobs_vert_images.append(
                        (vert_image, vert, this_3d_row["obj_id"], obs_info)
                    )

            if do_zoom_diff or do_zoom:
                fg = frame.astype(numpy.float32)
                zoom_fgs = []
                obj_ids = []
                this2ds = []
                if do_zoom_diff:
                    zoom_luminance_scale = 7.0
                    zoom_luminance_offset = 127
                    zoom_scaled_black = -zoom_luminance_offset / zoom_luminance_scale

                    # Zoomed difference image for this frame
                    bg = bg_frame.astype(numpy.float32)  # running_mean
                    cmp = cmp_frame.astype(
                        numpy.float32
                    )  # running_sumsqf, (already float32, probably)
                    cmp = numpy.sqrt(cmp - bg ** 2)  # approximates standard deviation
                    diff_im = fg - bg

                    zoom_diffs = []
                    zoom_absdiffs = []
                    zoom_bgs = []
                    zoom_cmps = []

                    maxabsdiff = []
                    maxcmp = []
                    meancmp = []
                plotted_pt_nos = set()
                radius = 10
                h, w = fg.shape
                for (xy, XYZ, obj_id, Pmean_meters) in kalman_vert_images:
                    x, y = xy
                    if (0 <= x <= w) and (0 <= y <= h):
                        minx = max(0, x - radius)
                        maxx = min(w, minx + (2 * radius))
                        miny = max(0, y - radius)
                        maxy = min(h, miny + (2 * radius))

                        zoom_fg = fg[miny:maxy, minx:maxx]
                        zoom_fg = ensure_minsize_image(
                            zoom_fg, (2 * radius, 2 * radius)
                        )
                        zoom_fgs.append(zoom_fg)

                        if do_zoom_diff:
                            zoom_diff = diff_im[miny:maxy, minx:maxx]
                            zoom_diff = ensure_minsize_image(
                                zoom_diff,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_diffs.append(zoom_diff)
                            zoom_absdiffs.append(abs(zoom_diff))

                            zoom_bg = bg[miny:maxy, minx:maxx]
                            zoom_bg = ensure_minsize_image(
                                zoom_bg,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_bgs.append(zoom_bg)

                            zoom_cmp = cmp[miny:maxy, minx:maxx]
                            zoom_cmp = ensure_minsize_image(
                                zoom_cmp,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_cmps.append(zoom_cmp)

                            maxabsdiff.append(abs(zoom_diff).max())
                            maxcmp.append(zoom_cmp.max())
                            meancmp.append(numpy.mean(zoom_cmp))

                        obj_ids.append(obj_id)

                        this2d = []
                        for pt_no in range(len(rows["x"])):
                            row = rows[pt_no]
                            x2d = row["x"]
                            y2d = row["y"]
                            area = row["area"]
                            slope = row["slope"]
                            eccentricity = row["eccentricity"]
                            try:
                                cur_val = row["cur_val"]
                                mean_val = row["mean_val"]
                                nstd_val = row["nstd_val"]
                            except IndexError:
                                # older format didn't save these
                                cur_val = None
                                mean_val = None
                                nstd_val = None
                            if (minx <= x2d <= maxx) and (miny <= y2d <= maxy):
                                this2d.append(
                                    (
                                        x2d - minx,
                                        y2d - miny,
                                        pt_no,
                                        area,
                                        slope,
                                        eccentricity,
                                        cur_val,
                                        mean_val,
                                        nstd_val,
                                    )
                                )
                                plotted_pt_nos.add(int(pt_no))
                        this2ds.append(this2d)
                all_pt_nos = set(range(len(rows)))
                missing_pt_nos = list(
                    all_pt_nos - plotted_pt_nos
                )  # those not plotted yet due to being kalman objects
                for pt_no in missing_pt_nos:
                    this_row = rows[pt_no]
                    (x2d, y2d) = this_row["x"], this_row["y"]
                    area = this_row["area"]
                    slope = this_row["slope"]
                    eccentricity = this_row["eccentricity"]
                    try:
                        cur_val = this_row["cur_val"]
                        mean_val = this_row["mean_val"]
                        nstd_val = this_row["nstd_val"]
                    except IndexError:
                        # older format didn't save these
                        cur_val = None
                        mean_val = None
                        nstd_val = None
                    x = x2d
                    y = y2d

                    if (0 <= x <= w) and (0 <= y <= h):
                        minx = max(0, x - radius)
                        maxx = min(w, minx + (2 * radius))
                        miny = max(0, y - radius)
                        maxy = min(h, miny + (2 * radius))

                        zoom_fg = fg[miny:maxy, minx:maxx]

                        zoom_fg = ensure_minsize_image(
                            zoom_fg, (2 * radius, 2 * radius)
                        )
                        zoom_fgs.append(zoom_fg)

                        if do_zoom_diff:
                            zoom_diff = diff_im[miny:maxy, minx:maxx]
                            zoom_diff = ensure_minsize_image(
                                zoom_diff,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_diffs.append(zoom_diff)
                            zoom_absdiffs.append(abs(zoom_diff))

                            zoom_bg = bg[miny:maxy, minx:maxx]
                            zoom_bg = ensure_minsize_image(
                                zoom_bg,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_bgs.append(zoom_bg)

                            zoom_cmp = cmp[miny:maxy, minx:maxx]
                            zoom_cmp = ensure_minsize_image(
                                zoom_cmp,
                                (2 * radius, 2 * radius),
                                fill=zoom_scaled_black,
                            )
                            zoom_cmps.append(zoom_cmp)

                            maxabsdiff.append(abs(zoom_diff).max())
                            maxcmp.append(zoom_cmp.max())
                            meancmp.append(numpy.mean(zoom_cmp))

                        obj_ids.append(None)

                        this2d = []
                        if 1:
                            if (minx <= x2d <= maxx) and (miny <= y2d <= maxy):
                                this2d.append(
                                    (
                                        x2d - minx,
                                        y2d - miny,
                                        pt_no,
                                        area,
                                        slope,
                                        eccentricity,
                                        cur_val,
                                        mean_val,
                                        nstd_val,
                                    )
                                )
                                plotted_pt_nos.add(int(pt_no))
                        this2ds.append(this2d)

                if len(zoom_fgs):
                    top_offset = 5
                    left_offset = 30
                    fgrow = numpy.hstack(zoom_fgs)
                    blackrow = numpy.zeros((top_offset, fgrow.shape[1]))
                    if do_zoom_diff:
                        scale = zoom_luminance_scale
                        offset = zoom_luminance_offset

                        diffrow = numpy.hstack(zoom_diffs) * scale + offset
                        cmprow = numpy.hstack(zoom_cmps) * scale + offset

                        bgrow = numpy.hstack(zoom_bgs)
                        absdiffrow = numpy.hstack(zoom_absdiffs) * scale + offset

                        row_ims = [
                            blackrow,
                            diffrow,
                            cmprow,
                            absdiffrow,
                            blackrow,
                            fgrow,
                            bgrow,
                        ]
                        labels = [
                            None,
                            "diff (s.)",
                            "std (s.)",
                            "absdiff (s.)",
                            None,
                            "raw",
                            "bg",
                        ]

                    else:
                        row_ims = [
                            blackrow,
                            fgrow,
                        ]
                        labels = [
                            None,
                            "raw",
                        ]

                    rightpart = numpy.vstack(row_ims)
                    leftpart = numpy.zeros((rightpart.shape[0], left_offset))
                    rightborder = numpy.zeros((rightpart.shape[0], 50))
                    newframe = numpy.hstack([leftpart, rightpart, rightborder])

                    newframe = numpy.clip(newframe, 0, 255)
                    im = newframe.astype(numpy.uint8)  # scale and offset
                    im = PIL.Image.frombuffer(
                        "L", (im.shape[1], im.shape[0]), im
                    )
                    w, h = im.size
                    rescale_factor = 5
                    im = im.resize((rescale_factor * w, rescale_factor * h))
                    im = im.convert("RGB")
                    if 1:
                        draw = aggdraw.Draw(im)
                    else:
                        if options.flip_y:
                            xform = aggdraw_coord_shifter.XformFlipY(
                                ymax=(im.size[1] - 1)
                            )
                        elif options.rotate_180:
                            xform = aggdraw_coord_shifter.XformRotate180(
                                xmax=(im.size[0] - 1), ymax=(im.size[1] - 1)
                            )
                        else:
                            xform = aggdraw_coord_shifter.XformIdentity()
                        draw = aggdraw_coord_shifter.CoordShiftDraw(
                            im, xform
                        )  # zoomed image
                        im = draw.get_image()

                    cumy = 0
                    absdiffy = None
                    cmpy = None
                    for j in range(len(labels)):
                        label = labels[j]
                        row_im = row_ims[j]
                        draw.text(
                            (rescale_factor * (0), rescale_factor * cumy),
                            label,
                            font_zoomed,
                        )
                        if do_zoom_diff and label == "absdiff (s.)":
                            absdiffy = cumy
                        elif do_zoom and label == "raw":
                            absdiffy = cumy
                        if label == "std (s.)":
                            cmpy = cumy
                        cumy += row_im.shape[0]

                    for i, obj_id in enumerate(obj_ids):
                        if obj_id is not None:
                            draw.text(
                                (rescale_factor * (i * 2 * radius + left_offset), 0),
                                "obj %d" % (obj_id,),
                                font_zoomed,
                            )

                    if do_zoom_diff:
                        for (
                            i,
                            (this_maxabsdiff, this_maxcmp, this_meancmp),
                        ) in enumerate(zip(maxabsdiff, maxcmp, meancmp)):

                            draw.text(
                                (
                                    rescale_factor * (i * 2 * radius + left_offset) + 3,
                                    rescale_factor * (absdiffy),
                                ),
                                "max %.0f" % (this_maxabsdiff,),
                                font_zoomed,
                            )

                            draw.text(
                                (
                                    rescale_factor * (i * 2 * radius + left_offset) + 3,
                                    rescale_factor * (cmpy),
                                ),
                                "max %.0f, mean %.1f" % (this_maxcmp, this_meancmp),
                                font_zoomed,
                            )

                    radius_pt = 3
                    for i, this2d in enumerate(this2ds):
                        for (
                            x2d,
                            y2d,
                            pt_no,
                            area,
                            slope,
                            eccentricity,
                            cur_val,
                            mean_val,
                            nstd_val,
                        ) in this2d:
                            xloc = rescale_factor * (x2d + i * 2 * radius + left_offset)
                            xloc1 = xloc - radius_pt
                            xloc2 = xloc + radius_pt
                            yloc = rescale_factor * (absdiffy + y2d)

                            if (
                                eccentricity < R.minimum_eccentricity
                                or area < options.area_threshold_for_orientation
                            ):
                                # only draw circle if not drawing slope line
                                draw.ellipse(
                                    [xloc1, yloc - radius_pt, xloc2, yloc + radius_pt],
                                    pen_zoomed,
                                )
                            else:
                                # draw slope line
                                direction = numpy.array([1, slope])
                                direction = direction / numpy.sqrt(
                                    numpy.sum(direction ** 2)
                                )  # normalize
                                pos = numpy.array([xloc, yloc])
                                if style == "debug":
                                    for sign in [-1, 1]:
                                        p1 = pos + sign * (
                                            eccentricity * 10 * direction
                                        )
                                        p2 = pos + sign * (
                                            R.minimum_eccentricity * 10 * direction
                                        )
                                        draw.line(
                                            [p1[0], p1[1], p2[0], p2[1]], pen_zoomed
                                        )
                                elif style == "pretty":
                                    radius = 20
                                    vec = direction * radius
                                    for sign in [-1, 1]:
                                        p1 = pos + sign * (1.2 * vec)
                                        p2 = pos + sign * (0.5 * vec)
                                        draw.line(
                                            [p1[0], p1[1], p2[0], p2[1]], pen_zoomed
                                        )
                            if style == "debug":
                                if cur_val is None:
                                    draw.text(
                                        (xloc, rescale_factor * (absdiffy + y2d)),
                                        "pt %d (%.1f, %.1f)"
                                        % (pt_no, area, eccentricity),
                                        font_zoomed,
                                    )
                                else:
                                    draw.text(
                                        (xloc, rescale_factor * (absdiffy + y2d)),
                                        "pt %d (%.1f, %.1f, %d, %d, %d)"
                                        % (
                                            pt_no,
                                            area,
                                            eccentricity,
                                            cur_val,
                                            mean_val,
                                            nstd_val,
                                        ),
                                        font_zoomed,
                                    )
                    draw.flush()

                    dirname = "zoomed_%s_movies" % os.path.splitext(h5_filename)[0]
                    fname = os.path.join(
                        dirname, "zoom_diff_%(cam_id)s_%(h5_frame)07d.png" % locals()
                    )
                    # dirname = os.path.abspath(os.path.split(fname)[0])
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    im.save(fname)

            # full image
            if PLOT == "image":
                assert fmf.format == "MONO8"
                imframe = np.clip(frame, 0, 255).astype(np.uint8)
                im = PIL.Image.frombuffer(
                    "L", (imframe.shape[1], imframe.shape[0]), imframe
                )
                im = im.convert("RGB")
                # draw = aggdraw.Draw(im)
                if options.flip_y:
                    xform = aggdraw_coord_shifter.XformFlipY(
                        ymax=(imframe.shape[0] - 1)
                    )
                elif options.rotate_180:
                    xform = aggdraw_coord_shifter.XformRotate180(
                        xmax=(imframe.shape[1] - 1), ymax=(imframe.shape[0] - 1)
                    )
                else:
                    xform = aggdraw_coord_shifter.XformIdentity()
                draw = aggdraw_coord_shifter.CoordShiftDraw(im, xform)
                im = draw.get_image()

                if style == "debug":
                    try:
                        strtime = datetime.datetime.fromtimestamp(mainbrain_timestamp)
                    except:
                        strtime = "<no 2d data timestamp>"
                    # draw.text( (0,0), 'frame %d, %s timestamp %s - %s'%(
                    #    h5_frame, cam_id, repr(fmf_timestamp), strtime), font2d )
                    draw.text_noxform(
                        (0, 0),
                        "frame %d, %s timestamp %s - %s"
                        % (h5_frame, cam_id, repr(fmf_timestamp), strtime),
                        font2d,
                    )

                # plot extracted data for full image
                if len(idxs):
                    for (
                        pt_no,
                        (camn, frame_pt_idx, x, y, area, slope, eccentricity),
                    ) in enumerate(
                        zip(
                            rows["camn"],
                            rows["frame_pt_idx"],
                            rows["x"],
                            rows["y"],
                            rows["area"],
                            rows["slope"],
                            rows["eccentricity"],
                        )
                    ):

                        if style == "debug":
                            radius = numpy.sqrt(area / (2 * numpy.pi))
                            draw.ellipse(
                                [x - radius, y - radius, x + radius, y + radius], pen2d
                            )

                            pos = numpy.array([x, y])
                            tmp_str = "pt %d (area %.1f, ecc %.1f)" % (
                                pt_no,
                                area,
                                eccentricity,
                            )
                            tmpw, tmph = draw.textsize(tmp_str, font2d)
                            # draw.text( (x+5,y-tmph-1), tmp_str, font2d )
                            draw.text_smartshift(
                                (x + 5, y - tmph - 1), (x, y), tmp_str, font2d
                            )
                        elif style == "pretty":
                            radius = 30
                            draw.ellipse(
                                [x - radius, y - radius, x + radius, y + radius], pen2d
                            )

                        # plot slope line
                        if (
                            options.body_axis
                            or options.smooth_orientations
                            or options.show_slope_2d
                        ):
                            if (
                                (R is None)
                                or (not eccentricity < R.minimum_eccentricity)
                            ) and (area >= options.area_threshold_for_orientation):
                                direction = numpy.array([1, slope])
                                direction = direction / numpy.sqrt(
                                    numpy.sum(direction ** 2)
                                )  # normalize
                                if style == "debug":
                                    pos = numpy.array([x, y])
                                    for sign in [-1, 1]:
                                        p1 = pos + sign * (
                                            eccentricity * 10 * direction
                                        )
                                        if R is None:
                                            p2 = pos + sign * (1.0 * 10 * direction)
                                        else:
                                            p2 = pos + sign * (
                                                R.minimum_eccentricity * 10 * direction
                                            )

                                        use_pen = pen2d
                                        if options.debug_ori_pickle is not None:
                                            try:
                                                tmplist = used_camn_dict[h5_frame]
                                            except KeyError:
                                                pass
                                            else:
                                                for (used_camn, ufpi) in tmplist:
                                                    if (
                                                        camn == used_camn
                                                        and frame_pt_idx == ufpi
                                                    ):
                                                        use_pen = pen2d_bold
                                                        break

                                        draw.line([p1[0], p1[1], p2[0], p2[1]], use_pen)
                                elif style == "pretty":
                                    vec = direction * radius
                                    pos = numpy.array([x, y])
                                    for sign in [-1, 1]:
                                        p1 = pos + sign * (1.2 * vec)
                                        p2 = pos + sign * (0.5 * vec)
                                        draw.line([p1[0], p1[1], p2[0], p2[1]], pen2d)

                for (xy, XYZ, obj_id, Pmean_meters) in kalman_vert_images:
                    if style in ["debug", "pretty", "prettier"]:
                        radius = 20
                        x, y = xy
                        X, Y, Z = XYZ
                        draw.ellipse(
                            [x - radius, y - radius, x + radius, y + radius], pen3d
                        )
                        for ori_verts_images in kalman_ori_verts_images:
                            ori_verts_images = numpy.array(ori_verts_images)
                            draw.line(ori_verts_images.flatten(), pen3d)
                        if style == "debug":
                            for raw_ori_verts_images in kalman_raw_ori_verts_images:
                                raw_ori_verts_images = numpy.array(raw_ori_verts_images)
                                draw.line(raw_ori_verts_images.flatten(), pen3d_raw)

                    if style == "debug":
                        pt_dist_meters = numpy.sqrt(
                            (X - cam_center_meters[0]) ** 2
                            + (Y - cam_center_meters[1]) ** 2
                            + (Z - cam_center_meters[2]) ** 2
                        )
                        ## draw.text( (x+5,y), 'obj %d (%.3f, %.3f, %.3f +- ~%f), dist %.2f'%(
                        ##     obj_id,X,Y,Z,Pmean_meters,pt_dist_meters),
                        ##            font3d )
                        draw.text_smartshift(
                            (x + 5, y),
                            (x, y),
                            "obj %d (%.3f, %.3f, %.3f +- ~%f), dist %.2f"
                            % (obj_id, X, Y, Z, Pmean_meters, pt_dist_meters),
                            font3d,
                        )

                if style == "debug":
                    for (xy, XYZ, obj_id, obs_info) in kobs_vert_images:
                        radius = 9
                        x, y = xy
                        X, Y, Z = XYZ
                        draw.ellipse(
                            [x - radius, y - radius, x + radius, y + radius], pen_obs
                        )
                        pt_dist_meters = numpy.sqrt(
                            (X - cam_center_meters[0]) ** 2
                            + (Y - cam_center_meters[1]) ** 2
                            + (Z - cam_center_meters[2]) ** 2
                        )
                        ## draw.text( (x+5,y), 'obj %d (%.3f, %.3f, %.3f), dist %.2f'%(obj_id,X,Y,Z,pt_dist_meters), font_obs )
                        draw.text_smartshift(
                            (x + 5, y),
                            (x, y),
                            "obj %d (%.3f, %.3f, %.3f), dist %.2f"
                            % (obj_id, X, Y, Z, pt_dist_meters),
                            font_obs,
                        )
                        (this_cam_ids, this_camn_idxs) = obs_info
                        for i, (obs_cam_id, pt_no) in enumerate(zip(*obs_info)):
                            ## draw.text( (x+15,y+(i+1)*10),
                            ##            '%s pt %d'%(obs_cam_id,pt_no), font_obs )
                            draw.text_smartshift(
                                (x + 15, y + (i + 1) * 10),
                                (x, y),
                                "%s pt %d" % (obs_cam_id, pt_no),
                                font_obs,
                            )

                if style in ["debug", "prettier-MLE-slope"]:
                    for kobs_ori_verts_images in [
                        kobs_ori_verts_images_a,
                        kobs_ori_verts_images_b,
                    ]:
                        for ori_verts_images in kobs_ori_verts_images:
                            ori_verts_images = numpy.array(ori_verts_images)
                            draw.line(ori_verts_images.flatten(), pen_obs)

                draw.flush()

        if 1:
            dirname = "full_%s_movies" % os.path.splitext(h5_filename)[0]
            fname = os.path.join(
                dirname, "smo_%(cam_id)s_%(h5_frame)07d.png" % locals()
            )
            if not os.path.exists(dirname):
                try:
                    os.makedirs(dirname)
                except OSError:
                    print(
                        "could not make directory %s: race condition "
                        "or permission problem?" % (dirname,)
                    )

            # print 'saving',fname
            if PLOT == "image":
                if im is not None:
                    im.save(fname)

    if not options.no_progress:
        pbar.finish()

    h5.close()


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "--fmf",
        dest="fmf_filename",
        type="string",
        help=".fmf (or .ufmf) filename (REQUIRED)",
    )

    parser.add_option(
        "--h5",
        dest="h5_filename",
        type="string",
        help=".h5 file with data2d_distorted (REQUIRED)",
    )

    parser.add_option(
        "--kalman",
        dest="kalman_filename",
        type="string",
        help=".h5 file with kalman data and 3D reconstructor",
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
        "--blank",
        dest="blank",
        type="int",
        help="frame number of FMF file (fmf-reference) of blank image to use when no image",
    )

    parser.add_option(
        "--disable-kalman-smoothing",
        action="store_false",
        dest="use_kalman_smoothing",
        default=True,
        help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)",
    )

    parser.add_option(
        "--fps",
        dest="fps",
        type="float",
        help="frames per second (used for Kalman filtering/smoothing)",
    )

    parser.add_option(
        "--dynamic-model", type="string", dest="dynamic_model", default=None,
    )

    parser.add_option(
        "--style", dest="style", type="string", default="debug",
    )

    parser.add_option(
        "--zoom",
        action="store_true",
        default=False,
        help="save zoomed image (not well tested)",
    )

    parser.add_option(
        "--smooth-orientations",
        action="store_true",
        help="use slow quaternion-based smoother if body axis data is available",
        default=False,
    )

    parser.add_option(
        "--body-axis",
        action="store_true",
        help="use body axis data if available",
        default=False,
    )

    parser.add_option(
        "--show-slope-2d",
        action="store_true",
        help="show 2D body axis data if available",
        default=False,
    )

    parser.add_option(
        "--zoom-diff",
        action="store_true",
        default=False,
        help="save zoomed difference image (not well tested)",
    )

    parser.add_option("--flip-y", action="store_true", default=False)

    parser.add_option("--rotate-180", action="store_true", default=False)

    parser.add_option("--no-progress", action="store_true", default=False)

    parser.add_option("--ufmf-white-background", action="store_true", default=False)

    parser.add_option("--ufmf-abs-diff", action="store_true", default=False)

    parser.add_option("--obj-only", type="string")

    parser.add_option("--up-dir", type="string")

    parser.add_option("--debug-ori-pickle", type="string")

    parser.add_option(
        "--area-threshold-for-orientation",
        type="float",
        default=0.0,
        help="minimum area to display orientation",
    )

    parser.add_option(
        "--ori-qual",
        type="float",
        default=None,
        help=("minimum orientation quality to use"),
    )

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args) or (options.fmf_filename is None):
        parser.print_help()
        return

    if options.up_dir is not None:
        up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        up_dir = None

    doit(
        fmf_filename=options.fmf_filename,
        h5_filename=options.h5_filename,
        kalman_filename=options.kalman_filename,
        fps=options.fps,
        dynamic_model=options.dynamic_model,
        use_kalman_smoothing=options.use_kalman_smoothing,
        start=options.start,
        stop=options.stop,
        style=options.style,
        blank=options.blank,
        do_zoom=options.zoom,
        do_zoom_diff=options.zoom_diff,
        options=options,
        up_dir=up_dir,
    )


if __name__ == "__main__":
    main()
