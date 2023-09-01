from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import
import motmot.ufmf.ufmf as ufmf_mod
import sys, os, tempfile, re, contextlib, warnings
from optparse import OptionParser
import flydra_analysis.a2.auto_discover_ufmfs as auto_discover_ufmfs
import flydra_analysis.a2.auto_discover_movies as auto_discover_movies
import numpy as np
import tables
import flydra_analysis.a2.utils as utils
import flydra_analysis.analysis.result_utils as result_utils
import subprocess, collections
import flydra_analysis.a2.ufmf_tools as ufmf_tools
import flydra_analysis.a2.core_analysis as core_analysis
import flydra_core.kalman.dynamic_models as dynamic_models
from flydra_analysis.a2.orientation_ekf_fitter import compute_ori_quality
import flydra_core.reconstruct as reconstruct
import flydra_core.geom as geom
import cherrypy  # ubuntu: install python-cherrypy3
from . import benu

from .tables_tools import open_file_safe


def get_config_defaults():
    # keep in sync with usage in main() below
    what = {
        "show_2d_position": False,
        "show_2d_orientation": False,
        "show_3d_MLE_position": False,
        "show_3d_smoothed_position": False,
        "show_3d_raw_orientation": False,
        "show_3d_raw_chosen_orientation": False,
        "show_3d_smoothed_orientation": False,
        "show_3d_obj_position_text": False,
        "minimum_display_orientation_quality": 0,
        "zoom_obj": None,
        "zoom_orig_pixels": 50,
        "zoom_factor": 5,
        "white_background": False,
        "max_resolution": None,
        "border_pixels": 0,
        "obj_labels": False,
        "linewidth": 1.0,
        "show_cam_id": False,
        "image_manipulation": "raw",
    }
    default = collections.defaultdict(dict)
    default["what to show"] = what
    return default


def montage(fnames, title, target):
    def get_tile(N):
        rows = int(np.ceil(np.sqrt(float(N))))
        cols = rows
        return "%dx%d" % (rows, cols)

    tile = get_tile(len(fnames))
    imnames = " ".join(fnames)

    CMD = (
        "gm montage %s -mode Concatenate -tile %s -bordercolor white "
        "-title '%s' "
        "%s" % (imnames, tile, title, target)
    )
    # print CMD
    subprocess.check_call(CMD, shell=True)


def load_3d_raw_data(kalman_filename, require_qual=True, **kwargs):
    with open_file_safe(kalman_filename, mode="r") as kh5:
        ca = core_analysis.get_global_CachingAnalyzer()
        all_obj_ids, obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
            kalman_filename
        )
        allrows = []
        if require_qual:
            allqualrows = []
        this_kw = {
            "min_ori_quality_required": kwargs["min_ori_quality_required"],
            "ori_quality_smooth_len": kwargs["ori_quality_smooth_len"],
        }
        for obj_id in obj_ids:
            rows = ca.load_dynamics_free_MLE_position(obj_id, data_file=kh5, **this_kw)
            allrows.append(rows)
            if require_qual:
                if np.any(~np.isnan(rows["hz_line0"])):
                    qualrows = compute_ori_quality(
                        data_file, rows["frame"], obj_id, smooth_len=0
                    )
                else:
                    qualrows = np.zeros_like(rows["hz_line0"])
                allqualrows.append(qualrows)
    data3d = np.concatenate(allrows)
    if require_qual:
        dataqual3d = np.concatenate(allqualrows)
        return data3d, dataqual3d
    else:
        return data3d


def is_obj_in_frame_range(obj_id, all_obj_ids, frames, start=None, stop=None):
    valid_cond = np.ones(frames.shape, dtype=np.bool_)
    if start is not None:
        valid_cond = valid_cond & (frames >= start)
    if stop is not None:
        valid_cond = valid_cond & (frames <= stop)
    valid_obj_ids = all_obj_ids[valid_cond]
    return bool(np.sum(valid_obj_ids == obj_id))


def load_3d_data(kalman_filename, start=None, stop=None, require_qual=True, **kwargs):
    with open_file_safe(kalman_filename, mode="r") as kh5:
        ca = core_analysis.get_global_CachingAnalyzer()
        all_obj_ids, obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
            kalman_filename
        )
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
                    'detected file loaded with dynamic model "%s"' % dynamic_model_name
                )
            if dynamic_model_name.startswith("EKF "):
                dynamic_model_name = dynamic_model_name[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model_name)
        allrows = []
        if require_qual:
            allqualrows = []
        for obj_id in obj_ids:
            if not is_obj_in_frame_range(
                obj_id, all_obj_ids, extra["frames"], start=start, stop=stop
            ):
                # obj_id not in range of frames that we're analyzing now
                continue
            try:
                rows = ca.load_data(
                    obj_id,
                    kalman_filename,
                    use_kalman_smoothing=True,
                    frames_per_second=fps,
                    dynamic_model_name=dynamic_model_name,
                    return_smoothed_directions=True,
                    **kwargs
                )
            except core_analysis.NotEnoughDataToSmoothError:
                warnings.warn(
                    "not enough data to smooth obj_id %d, skipping." % (obj_id,)
                )
                continue
            allrows.append(rows)
            if require_qual:
                qualrows = compute_ori_quality(
                    data_file, rows["frame"], obj_id, smooth_len=0
                )
                allqualrows.append(qualrows)
    data3d = np.concatenate(allrows)
    if require_qual:
        dataqual3d = np.concatenate(allqualrows)
        return data3d, dataqual3d
    else:
        return data3d


def make_montage(
    h5_filename,
    cfg_filename=None,
    ufmf_dir=None,
    dest_dir=None,
    save_ogv_movie=False,
    no_remove=False,
    max_n_frames=None,
    start=None,
    stop=None,
    movie_fnames=None,
    movie_cam_ids=None,
    caminfo_h5_filename=None,
    colormap=None,
    kalman_filename=None,
    candidate_index=0,
    nth_frame=1,
    verbose=False,
    reconstructor=None,
    **kwargs
):
    config = get_config_defaults()
    if cfg_filename is not None:
        loaded_cfg = cherrypy.lib.reprconf.as_dict(cfg_filename)
        for section in loaded_cfg:
            config[section].update(loaded_cfg.get(section, {}))
    else:
        warnings.warn("no configuration file specified -- using defaults")

    orientation_3d_line_length = 0.1

    if (
        config["what to show"]["show_3d_smoothed_position"]
        or config["what to show"]["show_3d_MLE_position"]
        or config["what to show"]["show_3d_raw_orientation"]
        or config["what to show"]["show_3d_raw_chosen_orientation"]
        or config["what to show"]["show_3d_smoothed_orientation"]
        or config["what to show"]["show_3d_obj_position_text"]
    ):
        if kalman_filename is None:
            raise ValueError("need kalman filename to show requested 3D data")

    if config["what to show"]["obj_labels"]:
        if kalman_filename is None:
            raise ValueError("need kalman filename to show object labels")

    if kalman_filename is not None:
        if (
            config["what to show"]["show_3d_smoothed_orientation"]
            or config["what to show"]["show_3d_raw_orientation"]
            or config["what to show"]["show_3d_raw_chosen_orientation"]
        ):
            need_quality_data = True
        else:
            need_quality_data = False

        if need_quality_data:
            # need data about quality of tracking
            data3d, dataqual_3d = load_3d_data(
                kalman_filename, start=start, stop=stop, require_qual=True, **kwargs
            )
        else:
            data3d = load_3d_data(
                kalman_filename, start=start, stop=stop, require_qual=False, **kwargs
            )
            dataqual_3d = None

        if (
            config["what to show"]["show_3d_MLE_position"]
            or config["what to show"]["show_3d_raw_orientation"]
        ):
            if need_quality_data:
                data_raw_3d, dataqual_raw_3d = load_3d_raw_data(
                    kalman_filename, **kwargs
                )
            else:
                data_raw_3d = load_3d_raw_data(
                    kalman_filename, require_qual=False, **kwargs
                )
                dataqual_raw_3d = None
        else:
            data_raw_3d, dataqual_raw_3d = None, None
        if reconstructor is None:
            R = reconstruct.Reconstructor(kalman_filename)
        else:
            R = reconstruct.Reconstructor(reconstructor)
    else:
        data3d = R = data_raw_3d = None
        dataqual_raw_3d = None
        dataqual_3d = None

    min_ori_qual = config["what to show"]["minimum_display_orientation_quality"]

    if movie_fnames is None:
        # This works based on UUIDs
        movie_fnames = auto_discover_movies.find_movies(
            h5_filename,
            ufmf_dir=ufmf_dir,
            candidate_index=candidate_index,
            verbose=verbose,
        )
        if verbose:
            print("autodiscovery: found movie_fnames: %r" % (movie_fnames,))
    else:
        if verbose:
            print("autodiscovery: movie_fnames specified, not finding movies")

    if len(movie_fnames) == 0:
        if verbose:
            print("autodiscovery: no FMF files found, looking for ufmfs")
        movie_fnames = auto_discover_ufmfs.find_ufmfs(
            h5_filename, ufmf_dir=ufmf_dir, careful=True, verbose=verbose,
        )
    else:
        if verbose:
            print("autodiscovery: prefixing directory")
        if ufmf_dir is not None:
            if verbose:
                print(
                    "autodiscovery: prefixing movie names with directory %r"
                    % (ufmf_dir,)
                )
            movie_fnames = [os.path.join(ufmf_dir, f) for f in movie_fnames]

    if len(movie_fnames) == 0:
        raise ValueError("no input movies -- nothing to do")
    elif verbose:
        print("movie_fnames:", movie_fnames)

    if dest_dir is None:
        dest_dir = os.curdir
    else:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    # get name of data

    datetime_str = os.path.splitext(os.path.split(h5_filename)[-1])[0]
    if datetime_str.startswith("DATA"):
        datetime_str = datetime_str[4:19]

    workaround_ffmpeg2theora_bug = True

    if caminfo_h5_filename is None:
        caminfo_h5_filename = h5_filename

    if caminfo_h5_filename is not None:
        with open_file_safe(caminfo_h5_filename, mode="r") as h5:
            camn2cam_id, tmp = result_utils.get_caminfo_dicts(h5)
            del tmp
    else:
        camn2cam_id = None

    blank_images = {}

    all_frame_montages = []
    for frame_enum, (frame_dict, frame) in enumerate(
        ufmf_tools.iterate_frames(
            h5_filename,
            movie_fnames,
            movie_cam_ids=movie_cam_ids,
            white_background=config["what to show"]["white_background"],
            max_n_frames=max_n_frames,
            start=start,
            stop=stop,
            rgb8_if_color=True,
            camn2cam_id=camn2cam_id,
        )
    ):

        if frame_enum % nth_frame != 0:
            continue

        tracker_data = frame_dict["tracker_data"]
        global_data = frame_dict["global_data"]

        if data3d is not None:
            this_frame_3d_data = data3d[data3d["frame"] == frame]
            if dataqual_3d is None:
                this_frame_dataqual = None
            else:
                this_frame_dataqual = dataqual_3d[data3d["frame"] == frame]
        else:
            this_frame_3d_data = None
            this_frame_dataqual = None

        if data_raw_3d is not None:
            this_frame_raw_3d_data = data_raw_3d[data_raw_3d["frame"] == frame]
            if dataqual_raw_3d is None:
                this_frame_raw_dataqual = None
            else:
                this_frame_raw_dataqual = dataqual_raw_3d[data_raw_3d["frame"] == frame]
        else:
            this_frame_raw_3d_data = None
            this_frame_raw_dataqual = None

        if config["what to show"]["zoom_obj"]:
            zoom_cond_3d = (
                this_frame_3d_data["obj_id"] == config["what to show"]["zoom_obj"]
            )
            if np.sum(zoom_cond_3d) == 0:
                # object not in this frame
                this_frame_this_obj_3d_data = None
            else:
                this_frame_this_obj_3d_data = this_frame_3d_data[zoom_cond_3d]

        if (frame_enum % 100) == 0:
            print("%s: frame %d" % (datetime_str, frame))

        saved_fnames = []
        for movie_idx, ufmf_fname in enumerate(movie_fnames):
            try:
                frame_data = frame_dict[ufmf_fname]
            except KeyError:
                # no data saved (frame skip on Prosilica camera?)
                if movie_cam_ids is not None:
                    cam_id = movie_cam_ids[movie_idx]
                else:
                    cam_id = ufmf_tools.get_cam_id_from_ufmf_fname(ufmf_fname)
                camn = None
                if cam_id not in blank_images:
                    im_w, im_h = global_data["width_heights"][cam_id]
                    image = np.empty((im_h, im_w), dtype=np.uint8)
                    image.fill(255)
                    blank_images[cam_id] = image
                image = blank_images[cam_id]
                mean_image = None
            else:
                cam_id = frame_data["cam_id"]
                camn = frame_data["camn"]
                image = frame_data["image"]
                if config["what to show"]["image_manipulation"] == "absdiff":
                    mean_image = frame_data["mean"]
                del frame_data
            save_fname = "tmp_frame%07d_%s.png" % (frame, cam_id)
            save_fname_path = os.path.join(dest_dir, save_fname)

            pixel_aspect = config[cam_id].get("pixel_aspect", 1)
            transform = config[cam_id].get("transform", "orig")

            border_pixels = config["what to show"]["border_pixels"]

            if config["what to show"]["max_resolution"] is not None:
                b2 = border_pixels * 2
                fix_w, fix_h = config["what to show"]["max_resolution"]
                fix_aspect = (fix_w - b2) / float(fix_h - b2)
                desire_aspect = image.shape[1] / float(image.shape[0] * pixel_aspect)
                if desire_aspect >= fix_aspect:
                    # image is wider than resolution given
                    device_w = fix_w - b2
                    device_h = (fix_w - b2) / desire_aspect
                    device_x = border_pixels
                    device_y = (fix_h - device_h + border_pixels) / 2.0
                else:
                    # image is taller than resolution given
                    device_h = fix_h - b2
                    device_w = (fix_h - b2) * desire_aspect
                    device_y = border_pixels
                    device_x = (fix_w - device_w + border_pixels) / 2.0
                user_rect = (0, 0, image.shape[1], image.shape[0])
            elif config["what to show"]["zoom_obj"]:
                if border_pixels != 0:
                    raise NotImplementedError()
                device_x = 0
                device_y = 0
                device_w = (
                    config["what to show"]["zoom_orig_pixels"]
                    * config["what to show"]["zoom_factor"]
                )
                device_h = device_w
                fix_w = device_w
                fix_h = device_h

                if this_frame_this_obj_3d_data is not None:
                    X = np.array(
                        [
                            this_frame_this_obj_3d_data["x"],
                            this_frame_this_obj_3d_data["y"],
                            this_frame_this_obj_3d_data["z"],
                            np.ones_like(this_frame_this_obj_3d_data["x"]),
                        ]
                    ).T
                    xarr, yarr = R.find2d(cam_id, X, distorted=True)
                    assert len(xarr) == 1
                    x = xarr[0]
                    y = yarr[0]
                    r = config["what to show"]["zoom_orig_pixels"] * 0.5
                    user_rect = (x - r, y - r, r * 2, r * 2)
                else:
                    # we're not tracking object -- don't draw anything
                    user_rect = (-1000, -1000, 10, 10)
            else:
                device_x = border_pixels
                device_y = border_pixels
                device_w = image.shape[1]
                device_h = int(
                    image.shape[0] * pixel_aspect
                )  # compensate for pixel_aspect
                fix_w = device_w + 2 * border_pixels
                fix_h = device_h + 2 * border_pixels
                user_rect = (0, 0, image.shape[1], image.shape[0])

            canv = benu.Canvas(save_fname_path, fix_w, fix_h)
            device_rect = (device_x, device_y, device_w, device_h)
            with canv.set_user_coords(device_rect, user_rect, transform=transform):
                if config["what to show"]["image_manipulation"] == "raw":
                    canv.imshow(image, 0, 0, cmap=colormap)
                if config["what to show"]["image_manipulation"] == "absdiff":
                    if mean_image is not None:
                        adsdiff_image = abs(
                            image.astype(np.int16) - mean_image.astype(np.int16)
                        )
                        scaled_show = np.clip((5 * adsdiff_image) + 127, 0, 255).astype(
                            np.uint8
                        )
                        canv.imshow(scaled_show, 0, 0, cmap=colormap)
                if config["what to show"]["show_2d_position"] and camn is not None:
                    cond = tracker_data["camn"] == camn
                    this_cam_data = tracker_data[cond]
                    xarr = np.atleast_1d(this_cam_data["x"])
                    yarr = np.atleast_1d(this_cam_data["y"])
                    canv.scatter(
                        xarr,
                        yarr,
                        color_rgba=(0, 0, 0, 1),
                        radius=10,
                        markeredgewidth=config["what to show"]["linewidth"],
                    )
                    # draw shadow
                    canv.scatter(
                        xarr + config["what to show"]["linewidth"],
                        yarr + config["what to show"]["linewidth"],
                        color_rgba=(1, 1, 1, 1),
                        radius=10,
                        markeredgewidth=config["what to show"]["linewidth"],
                    )
                if config["what to show"]["show_2d_orientation"] and camn is not None:
                    cond = tracker_data["camn"] == camn
                    this_cam_data = tracker_data[cond]
                    xarr = np.atleast_1d(this_cam_data["x"])
                    yarr = np.atleast_1d(this_cam_data["y"])
                    slope = np.atleast_1d(this_cam_data["slope"])
                    thetaarr = np.arctan(slope)
                    line_len = 30.0
                    xinc = np.cos(thetaarr) * line_len
                    yinc = np.sin(thetaarr) * line_len / float(pixel_aspect)
                    for x, y, xi, yi in zip(xarr, yarr, xinc, yinc):
                        xarr = np.array([x - xi, x + xi])
                        yarr = np.array([y - yi, y + yi])
                        if np.any(np.isnan(xarr)) or np.any(np.isnan(yarr)):
                            continue
                        canv.plot(
                            xarr,
                            yarr,
                            color_rgba=(0, 1, 0, 0.4),
                            linewidth=config["what to show"]["linewidth"],
                        )
                if (
                    config["what to show"]["show_3d_smoothed_position"]
                    and camn is not None
                ):
                    if len(this_frame_3d_data):
                        X = np.array(
                            [
                                this_frame_3d_data["x"],
                                this_frame_3d_data["y"],
                                this_frame_3d_data["z"],
                                np.ones_like(this_frame_3d_data["x"]),
                            ]
                        ).T
                        xarr, yarr = R.find2d(cam_id, X, distorted=True)
                        canv.scatter(
                            xarr,
                            yarr,
                            color_rgba=(0, 1, 1, 1),
                            radius=10,
                            markeredgewidth=config["what to show"]["linewidth"],
                        )

                if config["what to show"]["show_3d_MLE_position"] and camn is not None:
                    if len(this_frame_raw_3d_data):
                        X = np.array(
                            [
                                this_frame_raw_3d_data["x"],
                                this_frame_raw_3d_data["y"],
                                this_frame_raw_3d_data["z"],
                                np.ones_like(this_frame_raw_3d_data["x"]),
                            ]
                        ).T
                        xarr, yarr = R.find2d(cam_id, X, distorted=True)
                        canv.scatter(
                            xarr,
                            yarr,
                            color_rgba=(0.2, 0.2, 0.5, 1),
                            radius=8,
                            markeredgewidth=config["what to show"]["linewidth"],
                        )
                        # draw shadow
                        canv.scatter(
                            xarr + config["what to show"]["linewidth"],
                            yarr + config["what to show"]["linewidth"],
                            color_rgba=(0.7, 0.7, 1, 1),  # blue
                            radius=8,
                            markeredgewidth=config["what to show"]["linewidth"],
                        )

                if (
                    config["what to show"]["show_3d_raw_orientation"]
                    and camn is not None
                ):
                    if len(this_frame_raw_3d_data):
                        hzs = np.array(
                            [
                                this_frame_raw_3d_data["hz_line0"],
                                this_frame_raw_3d_data["hz_line1"],
                                this_frame_raw_3d_data["hz_line2"],
                                this_frame_raw_3d_data["hz_line3"],
                                this_frame_raw_3d_data["hz_line4"],
                                this_frame_raw_3d_data["hz_line5"],
                            ]
                        ).T
                        Xs = np.array(
                            [
                                this_frame_raw_3d_data["x"],
                                this_frame_raw_3d_data["y"],
                                this_frame_raw_3d_data["z"],
                            ]
                        ).T
                        cam_center = R.get_camera_center(cam_id)[:, 0]
                        for (X, hz, this_dataqual) in zip(
                            Xs, hzs, this_frame_raw_dataqual
                        ):
                            if this_dataqual < min_ori_qual:
                                continue
                            cam_ray = geom.line_from_points(
                                geom.ThreeTuple(cam_center), geom.ThreeTuple(X)
                            )
                            raw_ori_line = geom.line_from_HZline(hz)
                            X_ = raw_ori_line.get_my_point_closest_to_line(cam_ray)

                            ld = raw_ori_line.direction()
                            dmag = abs(ld)
                            du = ld * (1.0 / dmag)  # unit length direction (normalize)

                            length = 0.5  # arbitrary, 0.5 meters
                            N = 100  # n segments (to deal with distortion)

                            X0 = X_.vals + du.vals * -length / 2.0
                            X = (
                                X0[:, np.newaxis]
                                + np.linspace(0, length, N)[np.newaxis, :]
                                * du.vals[:, np.newaxis]
                            )
                            Xh = np.vstack((X, np.ones_like(X[0, np.newaxis, :]))).T
                            xarr, yarr = R.find2d(cam_id, Xh, distorted=True)
                            canv.plot(
                                xarr,
                                yarr,
                                color_rgba=(0, 0, 1, 1),  # blue
                                linewidth=config["what to show"]["linewidth"],
                            )

                if (
                    config["what to show"]["show_3d_smoothed_orientation"]
                    and camn is not None
                ):
                    if len(this_frame_3d_data):
                        for (row, ori_qual) in zip(
                            this_frame_3d_data, this_frame_dataqual
                        ):
                            if ori_qual < min_ori_qual:
                                continue
                            X0 = np.array(
                                [row["x"], row["y"], row["z"], np.ones_like(row["x"])]
                            ).T
                            dx = np.array(
                                [
                                    row["dir_x"],
                                    row["dir_y"],
                                    row["dir_z"],
                                    np.zeros_like(row["x"]),
                                ]
                            ).T
                            X1 = X0 + dx * orientation_3d_line_length
                            if np.any(np.isnan(X1)):
                                continue
                            pts = np.vstack([X0, X1])
                            xarr, yarr = R.find2d(cam_id, pts, distorted=True)
                            canv.plot(
                                xarr,
                                yarr,
                                color_rgba=(1, 0, 0, 1),  # red
                                linewidth=config["what to show"]["linewidth"],
                            )

                if (
                    config["what to show"]["show_3d_raw_chosen_orientation"]
                    and camn is not None
                ):
                    if len(this_frame_3d_data):
                        for (row, ori_qual) in zip(
                            this_frame_3d_data, this_frame_dataqual
                        ):
                            if ori_qual < min_ori_qual:
                                continue
                            X0 = np.array(
                                [row["x"], row["y"], row["z"], np.ones_like(row["x"])]
                            ).T
                            dx = np.array(
                                [
                                    row["rawdir_x"],
                                    row["rawdir_y"],
                                    row["rawdir_z"],
                                    np.zeros_like(row["x"]),
                                ]
                            ).T
                            X1 = X0 + dx * orientation_3d_line_length
                            if np.any(np.isnan(X1)):
                                continue
                            pts = np.vstack([X0, X1])
                            xarr, yarr = R.find2d(cam_id, pts, distorted=True)
                            canv.plot(
                                xarr,
                                yarr,
                                color_rgba=(1, 159.0 / 255, 0, 1),  # orange
                                linewidth=config["what to show"]["linewidth"],
                            )

                if config["what to show"]["obj_labels"] and camn is not None:
                    if len(this_frame_3d_data):
                        X = np.array(
                            [
                                this_frame_3d_data["x"],
                                this_frame_3d_data["y"],
                                this_frame_3d_data["z"],
                                np.ones_like(this_frame_3d_data["x"]),
                            ]
                        ).T
                        xarr, yarr = R.find2d(cam_id, X, distorted=True)
                        for i in range(len(xarr)):
                            obj_id = this_frame_3d_data["obj_id"][i]
                            canv.text(
                                "%d" % obj_id,
                                xarr[i],
                                yarr[i],
                                font_size=14,
                                color_rgba=(1, 0, 0, 1),
                            )

                if (
                    config["what to show"]["show_3d_obj_position_text"]
                    and camn is not None
                ):
                    if len(this_frame_3d_data):
                        X = np.array(
                            [
                                this_frame_3d_data["x"],
                                this_frame_3d_data["y"],
                                this_frame_3d_data["z"],
                                np.ones_like(this_frame_3d_data["x"]),
                            ]
                        ).T
                        xarr, yarr = R.find2d(cam_id, X, distorted=True)
                        for i in range(len(xarr)):
                            canv.text(
                                "(%.1f, %.1f, %.1f) mm"
                                % (
                                    X[i, 0] * 1000.0,
                                    X[i, 1] * 1000.0,
                                    X[i, 2] * 1000.0,
                                ),
                                xarr[i] + 10,
                                yarr[i],
                                font_size=14,
                                color_rgba=(0, 1, 1, 1),
                            )

                if config["what to show"]["show_cam_id"]:
                    canv.text(
                        "%s" % cam_id, 0, 20, font_size=14, color_rgba=(1, 0, 0, 1)
                    )

                if workaround_ffmpeg2theora_bug:
                    # first frame should get a colored pixel so that
                    # ffmpeg doesn't interpret the whole move as grayscale
                    canv.plot(
                        [0, 1], [0, 1], color_rgba=(1, 0, 0, 0.1),
                    )
                    workaround_ffmpeg2theora_bug = False  # Now we already did it.

            canv.save()
            saved_fnames.append(save_fname_path)

        target = os.path.join(
            dest_dir, "movie%s_frame%07d.png" % (datetime_str, frame_enum + 1)
        )
        # All cameras saved for this frame, make montage
        title = "%s frame %d" % (datetime_str, frame)
        montage(saved_fnames, title, target)
        all_frame_montages.append(target)
        if not no_remove:
            for fname in saved_fnames:
                os.unlink(fname)
    print("%s: %d frames montaged" % (datetime_str, len(all_frame_montages),))

    if save_ogv_movie:
        orig_dir = os.path.abspath(os.curdir)
        os.chdir(dest_dir)
        try:
            CMD = "ffmpeg2theora -v 10 movie%s_frame%%07d.png -o movie%s.ogv" % (
                datetime_str,
                datetime_str,
            )
            subprocess.check_call(CMD, shell=True)
        finally:
            os.chdir(orig_dir)

        if not no_remove:
            for fname in all_frame_montages:
                os.unlink(fname)


def main():
    # keep default config file in sync with get_config_defaults() above
    usage = """%prog DATAFILE2D.h5 [options]

The default configuration corresponds to a config file:

[what to show]
show_2d_position = False
show_2d_orientation = False
show_3d_MLE_position = False
show_3d_smoothed_position = False
show_3d_raw_orientation = False
show_3d_raw_chosen_orientation = False
show_3d_smoothed_orientation = False
show_3d_obj_position_text = False
minimum_display_orientation_quality = 0
white_background =  False
max_resolution = None
border_pixels = 0
zoom_obj = None
zoom_orig_pixels = 50
zoom_factor = 5
obj_labels = False
linewidth = 1.0
show_cam_id = False
image_manipulation = 'raw'

Config files may also have sections such as:

[cam7_1]
pixel_aspect=2 # each pixel is twice as wide as tall
transform='rot 180' # rotate the image 180 degrees (See transform
                    # keyword argument of
                    # :meth:`flydra_analysis.a2.benu.Canvas.set_user_coords`
                    # for all possible transforms.)

"""

    parser = OptionParser(usage)

    parser.add_option(
        "-k",
        "--kalman-file",
        dest="kalman_filename",
        type="string",
        help=".h5 file with 3D kalman data (and reconstructor if not given)",
    )

    parser.add_option(
        "-R",
        "--reconstructor",
        dest="reconstructor",
        type="string",
        help="reconstructor used for computing 2D coordinates",
    )

    parser.add_option(
        "--dest-dir",
        type="string",
        help="destination directory to save resulting files",
    )

    parser.add_option("--ufmf-dir", type="string", help="directory with .ufmf files")

    parser.add_option("--config", type="string", help="configuration file name")

    parser.add_option(
        "--max-n-frames",
        type="int",
        default=None,
        help="maximum number of frames to save",
    )

    parser.add_option("--start", type="int", default=None, help="start frame")

    parser.add_option("--stop", type="int", default=None, help="stop frame")

    parser.add_option(
        "--ogv", action="store_true", default=False, help="export .ogv video"
    )

    parser.add_option(
        "-n",
        "--no-remove",
        action="store_true",
        default=False,
        help="don't remove intermediate images",
    )

    parser.add_option(
        "--movie-fnames",
        type="string",
        default=None,
        help=(
            "names of movie files (separator is %r; "
            "don't autodiscover from .h5)" % (os.pathsep,)
        ),
    )

    parser.add_option(
        "--movie-cam-ids",
        type="string",
        default=None,
        help=(
            "cam_ids of movie files (separator is %r; "
            "don't autodiscover from .h5)" % (os.pathsep,)
        ),
    )

    parser.add_option(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose mode (help understand autodiscovery)",
    )

    parser.add_option("--colormap", type="string", default=None)

    parser.add_option(
        "--caminfo-h5-filename",
        type="string",
        help="path of h5 file from which to load caminfo",
    )

    parser.add_option(
        "--candidate",
        type="int",
        default=0,
        help="when multiple auto-discovered movie options, the index to take",
    )

    parser.add_option("--nth-frame", type="int", default=1, help="save every Nth frame")

    core_analysis.add_options_to_parser(parser)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        return

    movie_fnames = options.movie_fnames
    if movie_fnames is not None:
        movie_fnames = movie_fnames.split(os.pathsep)

    movie_cam_ids = options.movie_cam_ids
    if movie_cam_ids is not None:
        movie_cam_ids = movie_cam_ids.split(os.pathsep)

    reconstructor = options.reconstructor

    h5_filename = args[0]
    kwargs = core_analysis.get_options_kwargs(options)
    make_montage(
        h5_filename,
        kalman_filename=options.kalman_filename,
        cfg_filename=options.config,
        ufmf_dir=options.ufmf_dir,
        dest_dir=options.dest_dir,
        save_ogv_movie=options.ogv,
        no_remove=options.no_remove,
        max_n_frames=options.max_n_frames,
        start=options.start,
        stop=options.stop,
        movie_fnames=movie_fnames,
        movie_cam_ids=movie_cam_ids,
        caminfo_h5_filename=options.caminfo_h5_filename,
        colormap=options.colormap,
        candidate_index=options.candidate,
        verbose=options.verbose,
        nth_frame=options.nth_frame,
        reconstructor=reconstructor,
        **kwargs
    )
