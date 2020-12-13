from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import
import os, tempfile, errno
from optparse import OptionParser
import flydra_analysis.a2.auto_discover_ufmfs as auto_discover_ufmfs
import numpy as np
import flydra_analysis.analysis.result_utils as result_utils
import scipy.misc
import flydra_analysis.a2.ufmf_tools as ufmf_tools
import scipy.ndimage
import flydra_core.data_descriptions
from .tables_tools import open_file_safe
import cherrypy  # ubuntu: install python-cherrypy3
import collections, shutil


def get_config_defaults():
    default = {
        "pixel_aspect": 1,
        # In order of operations:
        "absdiff_max_frac_thresh": 0.4,
        "min_absdiff": 5,
        "area_minimum_threshold": 0,
    }
    result = collections.defaultdict(dict)
    result["default"] = default
    return result


def mkdir_p(path):
    # From
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def retrack_movies(
    h5_filename,
    output_h5_filename=None,
    max_n_frames=None,
    start=None,
    stop=None,
    ufmf_dir=None,
    cfg_filename=None,
    ufmf_filenames=None,
    save_debug_images=False,
):
    # We do a deferred import so test runners can import this python script
    # without depending on these, which depend on Intel IPP.

    import motmot.FastImage.FastImage as FastImage
    import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis


    # 2D data format for PyTables:
    Info2D = flydra_core.data_descriptions.Info2D

    if ufmf_filenames is None:
        ufmf_filenames = auto_discover_ufmfs.find_ufmfs(
            h5_filename, ufmf_dir=ufmf_dir, careful=True
        )
    print("ufmf_filenames: %r" % ufmf_filenames)
    if len(ufmf_filenames) == 0:
        raise RuntimeError("nothing to do (autodetection of .ufmf files failed)")

    if ufmf_dir is not None:
        if (not ufmf_filenames[0].startswith("/")) and (
            not os.path.isfile(ufmf_filenames[0])
        ):
            # filenames are not absolute and are not present, convert
            ufmf_filenames = [os.path.join(ufmf_dir, fname) for fname in ufmf_filenames]
        else:
            raise RuntimeError("ufmf_dir given but ufmf_filenames exist without it")

    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)

    # get name of data
    config = get_config_defaults()
    if cfg_filename is not None:
        loaded_cfg = cherrypy.lib.reprconf.as_dict(cfg_filename)
        for section in loaded_cfg:
            config[section].update(loaded_cfg.get(section, {}))
    default_camcfg = config["default"]
    for cam_id in config.keys():
        if cam_id == "default":
            continue
        # ensure default key/value pairs in each cam_id
        for key, value in default_camcfg.items():
            if key not in config[cam_id]:
                config[cam_id][key] = value

    datetime_str = os.path.splitext(os.path.split(h5_filename)[-1])[0]
    datetime_str = datetime_str[4:19]

    retrack_cam_ids = [ufmf_tools.get_cam_id_from_ufmf_fname(f) for f in ufmf_filenames]

    with open_file_safe(h5_filename, mode="r") as h5:

        # Find camns in original data
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

        retrack_camns = []
        for cam_id in retrack_cam_ids:
            retrack_camns.extend(cam_id2camns[cam_id])
        all_camns = camn2cam_id.keys()

        # Save results to temporary file. Copy to real location on success.
        tmpdir = tempfile.mkdtemp()
        tmp_output_h5_filename = os.path.join(tmpdir, "retrack.h5")

        with open_file_safe(
            tmp_output_h5_filename, mode="w", delete_on_error=True
        ) as output_h5:

            out_data2d = output_h5.create_table(
                output_h5.root,
                "data2d_distorted",
                Info2D,
                "2d data",
                expectedrows=h5.root.data2d_distorted.nrows,
            )

            # Are there any camns in original h5 that are not being retracked?
            if len(set(all_camns) - set(retrack_camns)):
                # Yes.

                # OK, exclude all camns to be retracked...
                orig_data2d = h5.root.data2d_distorted[:]  # read all data
                for camn in retrack_camns:
                    delete_cond = orig_data2d["camn"] == camn
                    save_cond = ~delete_cond
                    orig_data2d = orig_data2d[save_cond]

                # And save original data for untouched camns
                out_data2d.append(orig_data2d)

            for input_node in h5.root._f_iter_nodes():
                if input_node._v_name not in [
                    "data2d_distorted",
                    "kalman_estimates",
                    "ML_estimates",
                    "ML_estimates_2d_idxs",
                ]:
                    print("copying", input_node._v_name)
                    # copy everything from source to dest
                    input_node._f_copy(output_h5.root, recursive=True)

            fpc = realtime_image_analysis.FitParamsClass()  # allocate FitParamsClass

            count = 0
            iterate_frames = ufmf_tools.iterate_frames  # shorten notation
            for frame_enum, (frame_dict, frame) in enumerate(
                iterate_frames(
                    h5_filename,
                    ufmf_filenames,
                    max_n_frames=max_n_frames,
                    start=start,
                    stop=stop,
                )
            ):

                if (frame_enum % 100) == 0:
                    print("%s: frame %d" % (datetime_str, frame))

                for ufmf_fname in ufmf_filenames:
                    try:
                        frame_data = frame_dict[ufmf_fname]
                    except KeyError:
                        # no data saved (frame skip on Prosilica camera?)
                        continue
                    count += 1
                    camn = frame_data["camn"]
                    cam_id = frame_data["cam_id"]
                    camcfg = config.get(cam_id, default_camcfg)
                    image = frame_data["image"]
                    cam_received_timestamp = frame_data["cam_received_timestamp"]
                    timestamp = frame_data["timestamp"]

                    detected_points = True
                    obj_slices = None
                    if len(frame_data["regions"]) == 0:
                        # no data this frame -- go to next camera or frame
                        detected_points = False
                    if detected_points:
                        # print frame,cam_id,len(frame_data['regions'])
                        absdiff_im = abs(frame_data["mean"].astype(np.float32) - image)
                        thresh_val = (
                            np.max(absdiff_im) * camcfg["absdiff_max_frac_thresh"]
                        )
                        thresh_val = max(camcfg["min_absdiff"], thresh_val)
                        thresh_im = absdiff_im > thresh_val
                        labeled_im, n_labels = scipy.ndimage.label(thresh_im)
                        if not n_labels:
                            detected_points = False
                        else:
                            obj_slices = scipy.ndimage.find_objects(labeled_im)
                    detection = out_data2d.row
                    if detected_points:
                        height, width = image.shape
                        if save_debug_images:
                            xarr = []
                            yarr = []
                        frame_pt_idx = 0
                        detected_points = False  # possible not to find any below

                        for i in range(n_labels):
                            y_slice, x_slice = obj_slices[i]
                            # limit pixel operations to covering rectangle
                            this_labeled_im = labeled_im[y_slice, x_slice]
                            this_label_im = this_labeled_im == (i + 1)

                            # calculate area (number of binarized pixels)
                            xsum = np.sum(this_label_im, axis=0)
                            pixel_area = np.sum(xsum)
                            if pixel_area < camcfg["area_minimum_threshold"]:
                                continue

                            # calculate center
                            xpos = np.arange(x_slice.start, x_slice.stop, x_slice.step)
                            ypos = np.arange(y_slice.start, y_slice.stop, y_slice.step)

                            xmean = np.sum((xsum * xpos)) / np.sum(xsum)
                            ysum = np.sum(this_label_im, axis=1)
                            ymean = np.sum((ysum * ypos)) / np.sum(ysum)

                            if 1:
                                if camcfg["pixel_aspect"] == 1:
                                    this_fit_im = this_label_im
                                elif camcfg["pixel_aspect"] == 2:
                                    this_fit_im = np.repeat(this_label_im, 2, axis=0)
                                else:
                                    raise ValueError("unknown pixel_aspect")

                                fast_foreground = FastImage.asfastimage(
                                    this_fit_im.astype(np.uint8)
                                )

                                fail_fit = False
                                try:
                                    (
                                        x0_roi,
                                        y0_roi,
                                        weighted_area,
                                        slope,
                                        eccentricity,
                                    ) = fpc.fit(fast_foreground)
                                except realtime_image_analysis.FitParamsError as err:
                                    fail_fit = True
                                    print(
                                        "frame %d, ufmf %s: fit failed"
                                        % (frame, ufmf_fname)
                                    )
                                    print(err)
                                else:
                                    if camcfg["pixel_aspect"] == 2:
                                        y0_roi *= 0.5
                                    xmean = x_slice.start + x0_roi
                                    ymean = y_slice.start + y0_roi
                                    del weighted_area  # don't leave room for confusion
                            else:
                                fail_fit = True

                            if fail_fit:
                                slope = np.nan
                                eccentricity = np.nan

                            detection["camn"] = camn
                            detection["frame"] = frame
                            detection["timestamp"] = timestamp
                            detection["cam_received_timestamp"] = cam_received_timestamp
                            detection["x"] = xmean
                            detection["y"] = ymean
                            detection["area"] = pixel_area
                            detection["slope"] = slope
                            detection["eccentricity"] = eccentricity
                            detection["frame_pt_idx"] = frame_pt_idx
                            # XXX These are not yet implemented:
                            detection["cur_val"] = 0
                            detection["mean_val"] = np.nan
                            detection["sumsqf_val"] = np.nan
                            frame_pt_idx += 1
                            if save_debug_images:
                                xarr.append(xmean)
                                yarr.append(ymean)
                            detection.append()
                            detected_points = True

                        if save_debug_images:
                            save_dir = "debug"
                            mkdir_p(save_dir)
                            save_fname = "debug_%s_%d.png" % (cam_id, frame)
                            save_fname_path = os.path.join(save_dir, save_fname)
                            print("saving", save_fname_path)
                            from . import benu

                            canv = benu.Canvas(save_fname_path, width, height)
                            maxlabel = np.max(labeled_im)
                            fact = int(np.floor(255.0 / maxlabel))
                            canv.imshow((labeled_im * fact).astype(np.uint8), 0, 0)
                            canv.scatter(
                                xarr, yarr, color_rgba=(0, 1, 0, 1), radius=10,
                            )
                            canv.save()

                    if not detected_points:
                        # If no point was tracked for this frame,
                        # still save timestamp.
                        detection["camn"] = camn
                        detection["frame"] = frame
                        detection["timestamp"] = timestamp
                        detection["cam_received_timestamp"] = cam_received_timestamp
                        detection["x"] = np.nan
                        detection["y"] = np.nan
                        detection["area"] = np.nan
                        detection["slope"] = np.nan
                        detection["eccentricity"] = np.nan
                        detection["frame_pt_idx"] = 0
                        detection["cur_val"] = 0
                        detection["mean_val"] = np.nan
                        detection["sumsqf_val"] = np.nan
                        detection.append()
            if count == 0:
                raise RuntimeError("no frames processed")

    # move to correct location
    shutil.move(tmp_output_h5_filename, output_h5_filename)


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    parser.add_option("--ufmf-dir", type="string", help="directory with .ufmf files")

    parser.add_option(
        "--max-n-frames",
        type="int",
        default=None,
        help="maximum number of frames to save",
    )

    parser.add_option("--start", type="int", default=None, help="start frame")

    parser.add_option("--stop", type="int", default=None, help="stop frame")

    parser.add_option(
        "--h5", type="string", help="filename for input .h5 file with data2d_distorted"
    )

    parser.add_option(
        "--output-h5",
        type="string",
        help="filename for output .h5 file with data2d_distorted",
    )

    parser.add_option("--config", type="string", help="configuration file name")

    parser.add_option(
        "--ufmfs",
        type="string",
        help=("sequence of .ufmf filenames " "(e.g. 'cam1.ufmf:cam2.ufmf')"),
    )

    parser.add_option(
        "--save-images", action="store_true", default=False, help=("save images")
    )

    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.print_help()
        return

    if options.h5 is None:
        raise ValueError("--h5 option must be specified")

    if options.output_h5 is None:
        raise ValueError("--output-h5 option must be specified")

    if options.ufmfs is None:
        ufmf_filenames = None
    else:
        ufmf_filenames = options.ufmfs.split(os.pathsep)

    retrack_movies(
        options.h5,
        cfg_filename=options.config,
        ufmf_dir=options.ufmf_dir,
        max_n_frames=options.max_n_frames,
        start=options.start,
        stop=options.stop,
        output_h5_filename=options.output_h5,
        ufmf_filenames=ufmf_filenames,
        save_debug_images=options.save_images,
    )
