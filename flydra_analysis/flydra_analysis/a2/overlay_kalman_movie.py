from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import pkg_resources
import motmot.imops.imops as imops
from optparse import OptionParser
import flydra_core.reconstruct
import motmot.ufmf.ufmf as ufmf_mod
import motmot.FlyMovieFormat.FlyMovieFormat as fmf_mod
import flydra_analysis.a2.core_analysis as core_analysis
import warnings, os
import numpy as np
from . import benu

green = (0, 1, 0, 1)

# from http://jfly.iam.u-tokyo.ac.jp/color/index.html
orange = (0.9, 0.6, 0, 1)
sky_blue = (0.35, 0.70, 0.90, 1)


def doit(
    movie_fname=None,
    reconstructor_fname=None,
    h5_fname=None,
    cam_id=None,
    dest_dir=None,
    transform=None,
    start=None,
    stop=None,
    h5start=None,
    h5stop=None,
    show_obj_ids=False,
    obj_only=None,
    image_format=None,
    subtract_frame=None,
    save_framelist_fname=None,
):

    if dest_dir is None:
        dest_dir = os.curdir

    if movie_fname is None:
        raise NotImplementedError("")

    if image_format is None:
        image_format = "png"

    if cam_id is None:
        raise NotImplementedError("")

    if movie_fname.lower().endswith(".fmf"):
        movie = fmf_mod.FlyMovie(movie_fname)
    else:
        movie = ufmf_mod.FlyMovieEmulator(movie_fname)

    if start is None:
        start = 0

    if stop is None:
        stop = movie.get_n_frames() - 1

    ca = core_analysis.get_global_CachingAnalyzer()
    (obj_ids, unique_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(
        h5_fname
    )
    if obj_only is not None:
        unique_obj_ids = obj_only

    dynamic_model_name = extra["dynamic_model_name"]
    if dynamic_model_name.startswith("EKF"):
        dynamic_model_name = dynamic_model_name[4:]

    if reconstructor_fname is None:
        reconstructor = flydra_core.reconstruct.Reconstructor(data_file)
    else:
        reconstructor = flydra_core.reconstruct.Reconstructor(reconstructor_fname)

    fix_w = movie.get_width()
    fix_h = movie.get_height()
    is_color = imops.is_coding_color(movie.get_format())

    if subtract_frame is not None:
        if not subtract_frame.endswith(".fmf"):
            raise NotImplementedError("only fmf supported for --subtract-frame")
        tmp_fmf = fmf_mod.FlyMovie(subtract_frame)

        if is_color:
            tmp_frame, tmp_timestamp = tmp_fmf.get_next_frame()
            subtract_frame = imops.to_rgb8(tmp_fmf.get_format(), tmp_frame)
            subtract_frame = subtract_frame.astype(
                np.float32
            )  # force upconversion to float
        else:
            tmp_frame, tmp_timestamp = tmp_fmf.get_next_frame()
            subtract_frame = imops.to_mono8(tmp_fmf.get_format(), tmp_frame)
            subtract_frame = subtract_frame.astype(
                np.float32
            )  # force upconversion to float

    if save_framelist_fname is not None:
        save_framelist_fd = open(save_framelist_fname, mode="w")

    movie_fno_count = 0
    for movie_fno in range(start, stop + 1):
        movie.seek(movie_fno)
        image, timestamp = movie.get_next_frame()
        h5_frame = extra["time_model"].timestamp2framestamp(timestamp)
        if h5start is not None:
            if h5_frame < h5start:
                continue
        if h5stop is not None:
            if h5_frame > h5stop:
                continue
        if is_color:
            image = imops.to_rgb8(movie.get_format(), image)
        else:
            image = imops.to_mono8(movie.get_format(), image)
        if subtract_frame is not None:
            new_image = np.clip(image - subtract_frame, 0, 255)
            image = new_image.astype(np.uint8)
        warnings.warn("not implemented: interpolating data")
        h5_frame = int(round(h5_frame))
        if save_framelist_fname is not None:
            save_framelist_fd.write("%d\n" % h5_frame)

        movie_fno_count += 1
        if 0:
            # save starting from frame 1
            save_fname_path = os.path.splitext(movie_fname)[0] + "_frame%06d.%s" % (
                movie_fno_count,
                image_format,
            )
        else:
            # frame is frame in movie file
            save_fname_path = os.path.splitext(movie_fname)[0] + "_frame%06d.%s" % (
                movie_fno,
                image_format,
            )
        save_fname_path = os.path.join(dest_dir, save_fname_path)
        if transform in ["rot 90", "rot -90"]:
            device_rect = (0, 0, fix_h, fix_w)
            canv = benu.Canvas(save_fname_path, fix_h, fix_w)
        else:
            device_rect = (0, 0, fix_w, fix_h)
            canv = benu.Canvas(save_fname_path, fix_w, fix_h)
        user_rect = (0, 0, image.shape[1], image.shape[0])
        show_points = []
        with canv.set_user_coords(device_rect, user_rect, transform=transform):
            canv.imshow(image, 0, 0)
            for obj_id in unique_obj_ids:
                try:
                    data = ca.load_data(
                        obj_id,
                        data_file,
                        frames_per_second=extra["frames_per_second"],
                        dynamic_model_name=dynamic_model_name,
                    )
                except core_analysis.NotEnoughDataToSmoothError:
                    continue
                cond = data["frame"] == h5_frame
                idxs = np.nonzero(cond)[0]
                if not len(idxs):
                    continue  # no data at this frame for this obj_id
                assert len(idxs) == 1
                idx = idxs[0]
                row = data[idx]

                # circle over data point
                xyz = row["x"], row["y"], row["z"]
                x2d, y2d = reconstructor.find2d(cam_id, xyz, distorted=True)
                radius = 10
                canv.scatter(
                    [x2d], [y2d], color_rgba=green, markeredgewidth=3, radius=radius
                )

                if 1:
                    # z line to XY plane through origin
                    xyz0 = row["x"], row["y"], 0
                    x2d_z0, y2d_z0 = reconstructor.find2d(cam_id, xyz0, distorted=True)
                    warnings.warn("not distorting Z line")
                    if 1:
                        xdist = x2d - x2d_z0
                        ydist = y2d - y2d_z0
                        dist = np.sqrt(xdist ** 2 + ydist ** 2)
                        start_frac = radius / dist
                        if radius > dist:
                            start_frac = 0
                        x2d_r = x2d - xdist * start_frac
                        y2d_r = y2d - ydist * start_frac
                    else:
                        x2d_r = x2d
                        y2d_r = y2d
                    canv.plot(
                        [x2d_r, x2d_z0], [y2d_r, y2d_z0], color_rgba=green, linewidth=3
                    )
                if show_obj_ids:
                    show_points.append((obj_id, x2d, y2d))
        for show_point in show_points:
            obj_id, x2d, y2d = show_point
            x, y = canv.get_transformed_point(
                x2d, y2d, device_rect, user_rect, transform=transform
            )
            canv.text(
                "obj_id %d" % obj_id, x, y, color_rgba=(0, 1, 0, 1), font_size=20,
            )
        canv.save()


def main():
    # keep default config file in sync with get_config_defaults() above
    usage = """%prog DATAFILE3D.h5 [options]"""
    parser = OptionParser(usage)
    parser.add_option(
        "--dest-dir",
        type="string",
        help="destination directory to save resulting files",
    )
    parser.add_option(
        "--movie-fname",
        type="string",
        default=None,
        help="name of .ufmf of .fmf movie file (don't autodiscover from .h5)",
    )
    parser.add_option(
        "-r", "--reconstructor", type="string", help="calibration/reconstructor path"
    )
    parser.add_option(
        "--cam-id",
        type="string",
        default=None,
        help="cam_id of movie file (don't autodiscover from .h5)",
    )
    parser.add_option(
        "--start",
        type="int",
        default=None,
        help="first frame of the movie (not .h5) file to export",
    )
    parser.add_option(
        "--stop",
        type="int",
        default=None,
        help="last frame of the movie (not .h5) file to export",
    )
    parser.add_option(
        "--h5start",
        type="int",
        default=None,
        help="first frame of the .h5 file to export",
    )
    parser.add_option(
        "--h5stop",
        type="int",
        default=None,
        help="last frame of the .h5 file to export",
    )
    parser.add_option(
        "--transform", type="string", default=None, help="how to orient the movie file"
    )
    parser.add_option(
        "--show-obj-ids", action="store_true", default=False, help="show object ids"
    )
    parser.add_option("--obj-only", type="string")
    parser.add_option("--image-format", type="string", default="png")
    parser.add_option("--subtract-frame", type="string")
    parser.add_option("--save-framelist", type="string")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        return

    h5_fname = args[0]

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    doit(
        movie_fname=options.movie_fname,
        reconstructor_fname=options.reconstructor,
        h5_fname=h5_fname,
        cam_id=options.cam_id,
        dest_dir=options.dest_dir,
        transform=options.transform,
        start=options.start,
        stop=options.stop,
        h5start=options.h5start,
        h5stop=options.h5stop,
        show_obj_ids=options.show_obj_ids,
        obj_only=options.obj_only,
        image_format=options.image_format,
        save_framelist_fname=options.save_framelist,
        subtract_frame=options.subtract_frame,
    )


if __name__ == "__main__":
    main()
