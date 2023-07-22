from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])
import os, sys, math, contextlib, collections, warnings

import pkg_resources
import numpy as np
import tables as PT
from optparse import OptionParser
import flydra_core.reconstruct as reconstruct
import motmot.ufmf.ufmf as ufmf
import motmot.imops.imops as imops
import flydra_analysis.a2.utils as utils
import flydra_analysis.analysis.result_utils as result_utils
from . import core_analysis
import scipy.ndimage

import cairo
from . import benu
import adskalman.adskalman

from .tables_tools import clear_col, open_file_safe

font_size = 14


def shift_image(im, xy):
    def mapping(x):
        return (x[0] + xy[1], x[1] + xy[0])

    result = scipy.ndimage.geometric_transform(im, mapping, im.shape, order=0)
    return result


def get_cam_id_from_filename(filename, all_cam_ids):
    # guess cam_id
    n = 0
    found_cam_id = None
    for cam_id in all_cam_ids:
        if cam_id in filename:
            n += 1
            if found_cam_id is not None:
                raise ValueError("cam_id found more than once in filename")
            found_cam_id = cam_id
    return found_cam_id


def plot_image_subregion(
    raw_im,
    mean_im,
    absdiff_im,
    roiradius,
    fname,
    user_coords,
    scale=4.0,
    view="orig",
    extras=None,
):
    if extras is None:
        extras = {}
    output_ext = os.path.splitext(fname)[1].lower()

    roisize = 2 * roiradius
    imtypes = ["raw", "absdiff", "mean"]
    margin = 10
    square_edge = roisize * scale
    width = int(round(len(imtypes) * square_edge + (len(imtypes) + 1) * margin))
    height = int(round(square_edge + 2 * margin))
    if output_ext == ".pdf":
        output_surface = cairo.PDFSurface(fname, width, height)
    elif output_ext == ".svg":
        output_surface = cairo.SVGSurface(fname, width, height)
    elif output_ext == ".png":
        output_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    else:
        raise ValueError("unknown output extension %s" % output_ext)

    ctx = cairo.Context(output_surface)

    # fill with white
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()

    user_l, user_b, user_r, user_t = user_coords

    # setup transform
    #   calculate image boundary (user coords)

    for im_idx, im in enumerate(imtypes):
        if im == "raw":
            display_im = raw_im
        elif im == "mean":
            display_im = mean_im
        elif im == "absdiff":
            display_im = np.clip(5 * absdiff_im, 0, 255)
        # set transform - make a patch of the cairo
        # device be addressed with our image space
        # coords
        device_l = (im_idx + 1) * margin + im_idx * square_edge
        device_b = margin

        ctx.identity_matrix()  # reset
        if view == "orig":
            matrix = cairo.Matrix(
                xx=scale,
                yx=0,
                xy=0,
                yy=scale,
                x0=(device_l - scale * user_l),
                y0=(device_b - scale * user_b),
            )
        elif view == "rot -90":
            matrix = cairo.Matrix(
                xx=0,
                yx=scale,
                xy=scale,
                yy=0,
                x0=(device_l - scale * user_b),
                y0=(device_b - scale * user_l),
            )
        elif view == "rot 180":
            matrix = cairo.Matrix(
                xx=-scale,
                yx=0,
                xy=0,
                yy=-scale,
                x0=(device_l + scale * user_r),
                y0=(device_b + scale * user_t),
            )
        else:
            raise ValueError("unknown view '%s'" % view)
        ctx.set_matrix(matrix)
        ## print 'device_l-user_l, device_b-user_b',device_l-user_l, device_b-user_b
        ## #ctx.translate(device_l-user_l, device_b-user_b)
        ## if scale!= 1.0:
        ##     ctx.scale( scale, scale )
        ##     #raise NotImplementedError('')
        ## ctx.translate(device_l-user_l, device_b-user_b)
        ## #print 'square_edge/roisize, square_edge/roisize',square_edge/roisize, square_edge/roisize
        ## #ctx.scale( roisize/square_edge, square_edge/roisize)

        if 1:
            in_surface = benu.numpy2cairo(display_im.astype(np.uint8))
            ctx.rectangle(user_l, user_b, display_im.shape[1], display_im.shape[0])
            if 1:
                ctx.save()
                ctx.set_source_surface(in_surface, user_l, user_b)
                ctx.paint()
                ctx.restore()
            else:
                ctx.set_source_rgb(0, 0.3, 0)
                ctx.fill()

        if 0:
            ctx.move_to(user_l, user_b)

            ctx.line_to(user_r, user_b)
            ctx.line_to(user_r, user_t)
            ctx.line_to(user_l, user_t)
            ctx.line_to(user_l, user_b)
            ctx.close_path()
            ctx.set_source_rgb(0, 1, 0)
            ctx.fill()

            ctx.move_to(user_l + 5, user_b + 5)

            ctx.line_to(user_r - 40, user_b + 5)
            ctx.line_to(user_r - 40, user_t - 40)
            ctx.line_to(user_l + 5, user_t - 40)
            ctx.line_to(user_l + 5, user_b + 5)
            ctx.close_path()
            ctx.set_source_rgb(0, 0, 1)
            ctx.fill()

    if output_ext == ".png":
        output_surface.write_to_png(fname)
    else:
        ctx.show_page()
        output_surface.finish()


def flatten_image_stack(image_framenumbers, ims, im_coords, camn_pt_no_array, N=None):
    """take a stack of several images and flatten by finding min pixel"""
    if N is None:
        raise ValueError("N must be specified")
    assert np.all((image_framenumbers[1:] - image_framenumbers[:-1]) > 0)
    all_framenumbers = np.arange(
        image_framenumbers[0], image_framenumbers[-1] + 1, dtype=np.int64
    )

    assert N % 2 == 1
    offset = N // 2

    results = []
    for center_fno in range(offset, len(all_framenumbers) - offset):
        center_fno += all_framenumbers[0]
        center_idx = np.searchsorted(image_framenumbers, center_fno, side="right") - 1
        camn_pt_no = camn_pt_no_array[center_idx]
        orig_idxs_in_average = []
        ims_to_average = []
        coords_to_average = []
        for fno in range(center_fno - offset, center_fno + offset + 1):
            idx = np.searchsorted(image_framenumbers, fno, side="right") - 1
            if image_framenumbers[idx] == fno:
                orig_idxs_in_average.append(idx)
                ims_to_average.append(ims[idx])
                coords_to_average.append(im_coords[idx])

        n_images = len(coords_to_average)
        if 1:

            # XXX this is not very efficient.
            to_av = np.array(ims_to_average)
            ## print 'fno %d: min %.1f max %.1f'%(center_fno, to_av.min(), to_av.max())
            # av_im = np.mean( to_av, axis=0 )

            if to_av.shape == (0,):
                av_im = np.zeros(
                    (2, 2), dtype=np.uint8
                )  # just create a small blank image
                mean_lowerleft = np.array([np.nan, np.nan])
            else:
                av_im = np.min(to_av, axis=0)
                coords_to_average = np.array(coords_to_average)
                mean_lowerleft = np.mean(coords_to_average[:, :2], axis=0)
            results.append(
                (
                    center_fno,
                    av_im,
                    mean_lowerleft,
                    camn_pt_no,
                    center_idx,
                    orig_idxs_in_average,
                )
            )
    return results


def clip_and_math(raw_image, mean_image, xy, roiradius, maxsize):
    roisize = 2 * roiradius
    x, y = xy
    l = max(x - roiradius, 0)
    b = max(y - roiradius, 0)
    r = l + roisize
    t = b + roisize
    maxwidth, maxheight = maxsize
    if r > maxwidth:
        r = maxwidth
        l = r - roisize
    if t > maxheight:
        t = maxheight
        b = t - roisize

    raw_im = raw_image[b:t, l:r]
    mean_im = mean_image[b:t, l:r]
    absdiff_im = abs(mean_im.astype(np.float32) - raw_im)
    if absdiff_im.ndim == 3:
        # convert to grayscale
        absdiff_im = np.mean(absdiff_im, axis=2)

    return (l, b, r, t), raw_im, mean_im, absdiff_im


def doit(
    h5_filename=None,
    output_h5_filename=None,
    ufmf_filenames=None,
    kalman_filename=None,
    start=None,
    stop=None,
    view=None,
    erode=0,
    save_images=False,
    save_image_dir=None,
    intermediate_thresh_frac=None,
    final_thresh=None,
    stack_N_images=None,
    stack_N_images_min=None,
    old_sync_timestamp_source=False,
    do_rts_smoothing=True,
):
    """

    Copy all data in .h5 file (specified by h5_filename) to a new .h5
    file in which orientations are set based on image analysis of
    .ufmf files. Tracking data to associate 2D points from subsequent
    frames is read from the .h5 kalman file specified by
    kalman_filename.

    """

    # We do a deferred import so test runners can import this python script
    # without depending on these, which depend on Intel IPP.

    import motmot.FastImage.FastImage as FastImage
    import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis

    if view is None:
        view = ["orig" for f in ufmf_filenames]
    else:
        assert len(view) == len(ufmf_filenames)

    if intermediate_thresh_frac is None or final_thresh is None:
        raise ValueError("intermediate_thresh_frac and final_thresh must be " "set")

    filename2view = dict(zip(ufmf_filenames, view))

    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
        kalman_filename
    )
    try:
        ML_estimates_2d_idxs = data_file.root.ML_estimates_2d_idxs[:]
    except tables.exceptions.NoSuchNodeError as err1:
        # backwards compatibility
        try:
            ML_estimates_2d_idxs = data_file.root.kalman_observations_2d_idxs[:]
        except tables.exceptions.NoSuchNodeError as err2:
            raise err1

    if os.path.exists(output_h5_filename):
        raise RuntimeError("will not overwrite old file '%s'" % output_h5_filename)
    with open_file_safe(
        output_h5_filename, delete_on_error=True, mode="w"
    ) as output_h5:
        if save_image_dir is not None:
            if not os.path.exists(save_image_dir):
                os.mkdir(save_image_dir)

        with open_file_safe(h5_filename, mode="r") as h5:

            fps = result_utils.get_fps(h5, fail_on_error=True)

            for input_node in h5.root._f_iter_nodes():
                # copy everything from source to dest
                input_node._f_copy(output_h5.root, recursive=True)
            print("done copying")

            # Clear values in destination table that we may overwrite.
            dest_table = output_h5.root.data2d_distorted
            for colname in [
                "x",
                "y",
                "area",
                "slope",
                "eccentricity",
                "cur_val",
                "mean_val",
                "sumsqf_val",
            ]:
                if colname == "cur_val":
                    fill_value = 0
                else:
                    fill_value = np.nan
                clear_col(dest_table, colname, fill_value=fill_value)
            dest_table.flush()
            print("done clearing")

            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

            cam_id2fmfs = collections.defaultdict(list)
            cam_id2view = {}
            for ufmf_filename in ufmf_filenames:
                fmf = ufmf.FlyMovieEmulator(
                    ufmf_filename,
                    # darken=-50,
                    allow_no_such_frame_errors=True,
                )
                timestamps = fmf.get_all_timestamps()

                cam_id = get_cam_id_from_filename(fmf.filename, cam_id2camns.keys())
                cam_id2fmfs[cam_id].append(
                    (fmf, result_utils.Quick1DIndexer(timestamps))
                )

                cam_id2view[cam_id] = filename2view[fmf.filename]

            # associate framenumbers with timestamps using 2d .h5 file
            data2d = h5.root.data2d_distorted[:]  # load to RAM
            data2d_idxs = np.arange(len(data2d))
            h5_framenumbers = data2d["frame"]
            h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

            fpc = realtime_image_analysis.FitParamsClass()  # allocate FitParamsClass

            for obj_id_enum, obj_id in enumerate(use_obj_ids):
                print("object %d of %d" % (obj_id_enum, len(use_obj_ids)))

                # get all images for this camera and this obj_id

                obj_3d_rows = ca.load_dynamics_free_MLE_position(obj_id, data_file)

                this_obj_framenumbers = collections.defaultdict(list)
                if save_images:
                    this_obj_raw_images = collections.defaultdict(list)
                    this_obj_mean_images = collections.defaultdict(list)
                this_obj_absdiff_images = collections.defaultdict(list)
                this_obj_morphed_images = collections.defaultdict(list)
                this_obj_morph_failures = collections.defaultdict(list)
                this_obj_im_coords = collections.defaultdict(list)
                this_obj_com_coords = collections.defaultdict(list)
                this_obj_camn_pt_no = collections.defaultdict(list)

                for this_3d_row in obj_3d_rows:
                    # iterate over each sample in the current camera
                    framenumber = this_3d_row["frame"]
                    if start is not None:
                        if not framenumber >= start:
                            continue
                    if stop is not None:
                        if not framenumber <= stop:
                            continue
                    h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)

                    frame2d = data2d[h5_2d_row_idxs]
                    frame2d_idxs = data2d_idxs[h5_2d_row_idxs]

                    obs_2d_idx = this_3d_row["obs_2d_idx"]
                    kobs_2d_data = ML_estimates_2d_idxs[int(obs_2d_idx)]

                    # Parse VLArray.
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]

                    # Now, for each camera viewing this object at this
                    # frame, extract images.
                    for camn, camn_pt_no in zip(this_camns, this_camn_idxs):

                        # find 2D point corresponding to object
                        cam_id = camn2cam_id[camn]

                        movie_tups_for_this_camn = cam_id2fmfs[cam_id]
                        cond = (frame2d["camn"] == camn) & (
                            frame2d["frame_pt_idx"] == camn_pt_no
                        )
                        idxs = np.nonzero(cond)[0]
                        assert len(idxs) == 1
                        idx = idxs[0]

                        orig_data2d_rownum = frame2d_idxs[idx]

                        if not old_sync_timestamp_source:
                            # Change the next line to 'timestamp' for old
                            # data (before May/June 2009 -- the switch to
                            # fview_ext_trig)
                            frame_timestamp = frame2d[idx]["cam_received_timestamp"]
                        else:
                            # previous version
                            frame_timestamp = frame2d[idx]["timestamp"]
                        found = None
                        for fmf, fmf_timestamp_qi in movie_tups_for_this_camn:
                            fmf_fnos = fmf_timestamp_qi.get_idxs(frame_timestamp)
                            if not len(fmf_fnos):
                                continue
                            assert len(fmf_fnos) == 1

                            # should only be one .ufmf with this frame and cam_id
                            assert found is None

                            fmf_fno = fmf_fnos[0]
                            found = (fmf, fmf_fno)
                        if found is None:
                            print(
                                "no image data for frame timestamp %s cam_id %s"
                                % (repr(frame_timestamp), cam_id)
                            )
                            continue
                        fmf, fmf_fno = found
                        image, fmf_timestamp = fmf.get_frame(fmf_fno)
                        mean_image = fmf.get_mean_for_timestamp(fmf_timestamp)
                        coding = fmf.get_format()
                        if imops.is_coding_color(coding):
                            image = imops.to_rgb8(coding, image)
                            mean_image = imops.to_rgb8(coding, mean_image)
                        else:
                            image = imops.to_mono8(coding, image)
                            mean_image = imops.to_mono8(coding, mean_image)

                        xy = (
                            int(round(frame2d[idx]["x"])),
                            int(round(frame2d[idx]["y"])),
                        )
                        maxsize = (fmf.get_width(), fmf.get_height())

                        # Accumulate cropped images. Note that the region
                        # of the full image that the cropped image
                        # occupies changes over time as the tracked object
                        # moves. Thus, averaging these cropped-and-shifted
                        # images is not the same as simply averaging the
                        # full frame.

                        roiradius = 25
                        warnings.warn(
                            "roiradius hard-coded to %d: could be set "
                            "from 3D tracking" % roiradius
                        )
                        tmp = clip_and_math(image, mean_image, xy, roiradius, maxsize)
                        im_coords, raw_im, mean_im, absdiff_im = tmp

                        max_absdiff_im = absdiff_im.max()
                        intermediate_thresh = intermediate_thresh_frac * max_absdiff_im
                        absdiff_im[absdiff_im <= intermediate_thresh] = 0

                        if erode > 0:
                            morphed_im = scipy.ndimage.grey_erosion(
                                absdiff_im, size=erode
                            )
                            ## morphed_im = scipy.ndimage.binary_erosion(absdiff_im>1).astype(np.float32)*255.0
                        else:
                            morphed_im = absdiff_im

                        y0_roi, x0_roi = scipy.ndimage.center_of_mass(morphed_im)
                        x0 = im_coords[0] + x0_roi
                        y0 = im_coords[1] + y0_roi

                        if 1:
                            morphed_im_binary = morphed_im > 0
                            labels, n_labels = scipy.ndimage.label(morphed_im_binary)
                            morph_fail_because_multiple_blobs = False

                            if n_labels > 1:
                                x0, y0 = np.nan, np.nan
                                # More than one blob -- don't allow image.
                                if 1:
                                    # for min flattening
                                    morphed_im = np.empty(
                                        morphed_im.shape, dtype=np.uint8
                                    )
                                    morphed_im.fill(255)
                                    morph_fail_because_multiple_blobs = True
                                else:
                                    # for mean flattening
                                    morphed_im = np.zeros_like(morphed_im)
                                    morph_fail_because_multiple_blobs = True

                        this_obj_framenumbers[camn].append(framenumber)
                        if save_images:
                            this_obj_raw_images[camn].append((raw_im, im_coords))
                            this_obj_mean_images[camn].append(mean_im)
                        this_obj_absdiff_images[camn].append(absdiff_im)
                        this_obj_morphed_images[camn].append(morphed_im)
                        this_obj_morph_failures[camn].append(
                            morph_fail_because_multiple_blobs
                        )
                        this_obj_im_coords[camn].append(im_coords)
                        this_obj_com_coords[camn].append((x0, y0))
                        this_obj_camn_pt_no[camn].append(orig_data2d_rownum)
                        if 0:
                            fname = "obj%05d_%s_frame%07d_pt%02d.png" % (
                                obj_id,
                                cam_id,
                                framenumber,
                                camn_pt_no,
                            )
                            plot_image_subregion(
                                raw_im,
                                mean_im,
                                absdiff_im,
                                roiradius,
                                fname,
                                im_coords,
                                view=filename2view[fmf.filename],
                            )

                # Now, all the frames from all cameras for this obj_id
                # have been gathered. Do a camera-by-camera analysis.
                for camn in this_obj_absdiff_images:
                    cam_id = camn2cam_id[camn]
                    image_framenumbers = np.array(this_obj_framenumbers[camn])
                    if save_images:
                        raw_images = this_obj_raw_images[camn]
                        mean_images = this_obj_mean_images[camn]
                    absdiff_images = this_obj_absdiff_images[camn]
                    morphed_images = this_obj_morphed_images[camn]
                    morph_failures = np.array(this_obj_morph_failures[camn])
                    im_coords = this_obj_im_coords[camn]
                    com_coords = this_obj_com_coords[camn]
                    camn_pt_no_array = this_obj_camn_pt_no[camn]

                    all_framenumbers = np.arange(
                        image_framenumbers[0], image_framenumbers[-1] + 1
                    )

                    com_coords = np.array(com_coords)
                    if do_rts_smoothing:
                        # Perform RTS smoothing on center-of-mass coordinates.

                        # Find first good datum.
                        fgnz = np.nonzero(~np.isnan(com_coords[:, 0]))
                        com_coords_smooth = np.empty(com_coords.shape, dtype=np.float64)
                        com_coords_smooth.fill(np.nan)

                        if len(fgnz[0]):
                            first_good = fgnz[0][0]

                            RTS_com_coords = com_coords[first_good:, :]

                            # Setup parameters for Kalman filter.
                            dt = 1.0 / fps
                            A = np.array(
                                [
                                    [1, 0, dt, 0],  # process update
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ],
                                dtype=np.float64,
                            )
                            C = np.array(
                                [[1, 0, 0, 0], [0, 1, 0, 0]],  # observation matrix
                                dtype=np.float64,
                            )
                            Q = 0.1 * np.eye(4)  # process noise
                            R = 1.0 * np.eye(2)  # observation noise
                            initx = np.array(
                                [RTS_com_coords[0, 0], RTS_com_coords[0, 1], 0, 0],
                                dtype=np.float64,
                            )
                            initV = 2 * np.eye(4)
                            initV[0, 0] = 0.1
                            initV[1, 1] = 0.1
                            y = RTS_com_coords
                            xsmooth, Vsmooth = adskalman.adskalman.kalman_smoother(
                                y, A, C, Q, R, initx, initV
                            )
                            com_coords_smooth[first_good:] = xsmooth[:, :2]

                        # Now shift images

                        image_shift = com_coords_smooth - com_coords
                        bad_cond = np.isnan(image_shift[:, 0])
                        # broadcast zeros to places where no good tracking
                        image_shift[bad_cond, 0] = 0
                        image_shift[bad_cond, 1] = 0

                        shifted_morphed_images = [
                            shift_image(im, xy)
                            for im, xy in zip(morphed_images, image_shift)
                        ]

                        results = flatten_image_stack(
                            image_framenumbers,
                            shifted_morphed_images,
                            im_coords,
                            camn_pt_no_array,
                            N=stack_N_images,
                        )
                    else:
                        results = flatten_image_stack(
                            image_framenumbers,
                            morphed_images,
                            im_coords,
                            camn_pt_no_array,
                            N=stack_N_images,
                        )

                    # The variable fno (the first element of the results
                    # tuple) is guaranteed to be contiguous and to span
                    # the range from the first to last frames available.

                    for (
                        fno,
                        av_im,
                        lowerleft,
                        orig_data2d_rownum,
                        orig_idx,
                        orig_idxs_in_average,
                    ) in results:

                        # Clip image to reduce moment arms.
                        av_im[av_im <= final_thresh] = 0

                        fail_fit = False
                        fast_av_im = FastImage.asfastimage(av_im.astype(np.uint8))
                        try:
                            (x0_roi, y0_roi, area, slope, eccentricity) = fpc.fit(
                                fast_av_im
                            )
                        except realtime_image_analysis.FitParamsError as err:
                            fail_fit = True

                        this_morph_failures = morph_failures[orig_idxs_in_average]
                        n_failed_images = np.sum(this_morph_failures)
                        n_good_images = stack_N_images - n_failed_images
                        if n_good_images >= stack_N_images_min:
                            n_images_is_acceptable = True
                        else:
                            n_images_is_acceptable = False

                        if fail_fit:
                            x0_roi = np.nan
                            y0_roi = np.nan
                            area, slope, eccentricity = np.nan, np.nan, np.nan

                        if not n_images_is_acceptable:
                            x0_roi = np.nan
                            y0_roi = np.nan
                            area, slope, eccentricity = np.nan, np.nan, np.nan

                        x0 = x0_roi + lowerleft[0]
                        y0 = y0_roi + lowerleft[1]

                        if 1:
                            for row in dest_table.iterrows(
                                start=orig_data2d_rownum, stop=orig_data2d_rownum + 1
                            ):

                                row["x"] = x0
                                row["y"] = y0
                                row["area"] = area
                                row["slope"] = slope
                                row["eccentricity"] = eccentricity
                                row.update()  # save data

                        if save_images:
                            # Display debugging images
                            fname = "av_obj%05d_%s_frame%07d.png" % (
                                obj_id,
                                cam_id,
                                fno,
                            )
                            if save_image_dir is not None:
                                fname = os.path.join(save_image_dir, fname)

                            raw_im, raw_coords = raw_images[orig_idx]
                            mean_im = mean_images[orig_idx]
                            absdiff_im = absdiff_images[orig_idx]
                            morphed_im = morphed_images[orig_idx]
                            raw_l, raw_b = raw_coords[:2]

                            imh, imw = raw_im.shape[:2]
                            n_ims = 5

                            if 1:
                                # increase contrast
                                contrast_scale = 2.0
                                av_im_show = np.clip(av_im * contrast_scale, 0, 255)

                            margin = 10
                            scale = 3

                            # calculate the orientation line
                            yintercept = y0 - slope * x0
                            xplt = np.array(
                                [
                                    lowerleft[0] - 5,
                                    lowerleft[0] + av_im_show.shape[1] + 5,
                                ]
                            )
                            yplt = slope * xplt + yintercept
                            if 1:
                                # only send non-nan values to plot
                                plt_good = ~np.isnan(xplt) & ~np.isnan(yplt)
                                xplt = xplt[plt_good]
                                yplt = yplt[plt_good]

                            top_row_width = scale * imw * n_ims + (1 + n_ims) * margin
                            SHOW_STACK = True
                            if SHOW_STACK:
                                n_stack_rows = 4
                                rw = scale * imw * stack_N_images + (1 + n_ims) * margin
                                row_width = max(top_row_width, rw)
                                col_height = (
                                    n_stack_rows * scale * imh
                                    + (n_stack_rows + 1) * margin
                                )
                                stack_margin = 20
                            else:
                                row_width = top_row_width
                                col_height = scale * imh + 2 * margin
                                stack_margin = 0

                            canv = benu.Canvas(
                                fname,
                                row_width,
                                col_height + stack_margin,
                                color_rgba=(1, 1, 1, 1),
                            )

                            if SHOW_STACK:
                                for (stacki, s_orig_idx) in enumerate(
                                    orig_idxs_in_average
                                ):

                                    (s_raw_im, s_raw_coords) = raw_images[s_orig_idx]
                                    s_raw_l, s_raw_b = s_raw_coords[:2]
                                    s_imh, s_imw = s_raw_im.shape[:2]
                                    user_rect = (s_raw_l, s_raw_b, s_imw, s_imh)

                                    x_display = (stacki + 1) * margin + (
                                        scale * imw
                                    ) * stacki
                                    for show in ["raw", "absdiff", "morphed"]:
                                        if show == "raw":
                                            y_display = scale * imh + 2 * margin
                                        elif show == "absdiff":
                                            y_display = 2 * scale * imh + 3 * margin
                                        elif show == "morphed":
                                            y_display = 3 * scale * imh + 4 * margin
                                        display_rect = (
                                            x_display,
                                            y_display + stack_margin,
                                            scale * raw_im.shape[1],
                                            scale * raw_im.shape[0],
                                        )

                                        with canv.set_user_coords(
                                            display_rect,
                                            user_rect,
                                            transform=cam_id2view[cam_id],
                                        ):

                                            if show == "raw":
                                                s_im = s_raw_im.astype(np.uint8)
                                            elif show == "absdiff":
                                                tmp = absdiff_images[s_orig_idx]
                                                s_im = tmp.astype(np.uint8)
                                            elif show == "morphed":
                                                tmp = morphed_images[s_orig_idx]
                                                s_im = tmp.astype(np.uint8)

                                            canv.imshow(s_im, s_raw_l, s_raw_b)
                                            sx0, sy0 = com_coords[s_orig_idx]
                                            X = [sx0]
                                            Y = [sy0]
                                            # the raw coords in red
                                            canv.scatter(
                                                X, Y, color_rgba=(1, 0.5, 0.5, 1)
                                            )

                                            if do_rts_smoothing:
                                                sx0, sy0 = com_coords_smooth[s_orig_idx]
                                                X = [sx0]
                                                Y = [sy0]
                                                # the RTS smoothed coords in green
                                                canv.scatter(
                                                    X, Y, color_rgba=(0.5, 1, 0.5, 1)
                                                )

                                            if s_orig_idx == orig_idx:
                                                boxx = np.array(
                                                    [
                                                        s_raw_l,
                                                        s_raw_l,
                                                        s_raw_l + s_imw,
                                                        s_raw_l + s_imw,
                                                        s_raw_l,
                                                    ]
                                                )
                                                boxy = np.array(
                                                    [
                                                        s_raw_b,
                                                        s_raw_b + s_imh,
                                                        s_raw_b + s_imh,
                                                        s_raw_b,
                                                        s_raw_b,
                                                    ]
                                                )
                                                canv.plot(
                                                    boxx,
                                                    boxy,
                                                    color_rgba=(0.5, 1, 0.5, 1),
                                                )
                                        if show == "morphed":
                                            canv.text(
                                                "morphed %d" % (s_orig_idx - orig_idx,),
                                                display_rect[0],
                                                (
                                                    display_rect[1]
                                                    + display_rect[3]
                                                    + stack_margin
                                                    - 20
                                                ),
                                                font_size=font_size,
                                                color_rgba=(1, 0, 0, 1),
                                            )

                            # Display raw_im
                            display_rect = (
                                margin,
                                margin,
                                scale * raw_im.shape[1],
                                scale * raw_im.shape[0],
                            )
                            user_rect = (raw_l, raw_b, imw, imh)
                            with canv.set_user_coords(
                                display_rect, user_rect, transform=cam_id2view[cam_id],
                            ):
                                canv.imshow(raw_im.astype(np.uint8), raw_l, raw_b)
                                canv.plot(
                                    xplt, yplt, color_rgba=(0, 1, 0, 0.5)
                                )  # the orientation line
                            canv.text(
                                "raw",
                                display_rect[0],
                                display_rect[1] + display_rect[3],
                                font_size=font_size,
                                color_rgba=(0.5, 0.5, 0.9, 1),
                                shadow_offset=1,
                            )

                            # Display mean_im
                            display_rect = (
                                2 * margin + (scale * imw),
                                margin,
                                scale * mean_im.shape[1],
                                scale * mean_im.shape[0],
                            )
                            user_rect = (raw_l, raw_b, imw, imh)
                            with canv.set_user_coords(
                                display_rect, user_rect, transform=cam_id2view[cam_id],
                            ):
                                canv.imshow(mean_im.astype(np.uint8), raw_l, raw_b)
                            canv.text(
                                "mean",
                                display_rect[0],
                                display_rect[1] + display_rect[3],
                                font_size=font_size,
                                color_rgba=(0.5, 0.5, 0.9, 1),
                                shadow_offset=1,
                            )

                            # Display absdiff_im
                            display_rect = (
                                3 * margin + (scale * imw) * 2,
                                margin,
                                scale * absdiff_im.shape[1],
                                scale * absdiff_im.shape[0],
                            )
                            user_rect = (raw_l, raw_b, imw, imh)
                            absdiff_clip = np.clip(absdiff_im * contrast_scale, 0, 255)
                            with canv.set_user_coords(
                                display_rect, user_rect, transform=cam_id2view[cam_id],
                            ):
                                canv.imshow(absdiff_clip.astype(np.uint8), raw_l, raw_b)
                            canv.text(
                                "absdiff",
                                display_rect[0],
                                display_rect[1] + display_rect[3],
                                font_size=font_size,
                                color_rgba=(0.5, 0.5, 0.9, 1),
                                shadow_offset=1,
                            )

                            # Display morphed_im
                            display_rect = (
                                4 * margin + (scale * imw) * 3,
                                margin,
                                scale * morphed_im.shape[1],
                                scale * morphed_im.shape[0],
                            )
                            user_rect = (raw_l, raw_b, imw, imh)
                            morphed_clip = np.clip(morphed_im * contrast_scale, 0, 255)
                            with canv.set_user_coords(
                                display_rect, user_rect, transform=cam_id2view[cam_id],
                            ):
                                canv.imshow(morphed_clip.astype(np.uint8), raw_l, raw_b)
                            if 0:
                                canv.text(
                                    "morphed",
                                    display_rect[0],
                                    display_rect[1] + display_rect[3],
                                    font_size=font_size,
                                    color_rgba=(0.5, 0.5, 0.9, 1),
                                    shadow_offset=1,
                                )

                            # Display time-averaged absdiff_im
                            display_rect = (
                                5 * margin + (scale * imw) * 4,
                                margin,
                                scale * av_im_show.shape[1],
                                scale * av_im_show.shape[0],
                            )
                            user_rect = (
                                lowerleft[0],
                                lowerleft[1],
                                av_im_show.shape[1],
                                av_im_show.shape[0],
                            )
                            with canv.set_user_coords(
                                display_rect, user_rect, transform=cam_id2view[cam_id],
                            ):
                                canv.imshow(
                                    av_im_show.astype(np.uint8),
                                    lowerleft[0],
                                    lowerleft[1],
                                )
                                canv.plot(
                                    xplt, yplt, color_rgba=(0, 1, 0, 0.5)
                                )  # the orientation line
                            canv.text(
                                "stacked/flattened",
                                display_rect[0],
                                display_rect[1] + display_rect[3],
                                font_size=font_size,
                                color_rgba=(0.5, 0.5, 0.9, 1),
                                shadow_offset=1,
                            )

                            canv.text(
                                "%s frame % 7d: eccentricity % 5.1f, min N images %d, actual N images %d"
                                % (
                                    cam_id,
                                    fno,
                                    eccentricity,
                                    stack_N_images_min,
                                    n_good_images,
                                ),
                                0,
                                15,
                                font_size=font_size,
                                color_rgba=(0.6, 0.7, 0.9, 1),
                                shadow_offset=1,
                            )
                            canv.save()

                # Save results to new table
                if 0:
                    recarray = np.rec.array(
                        list_of_rows_of_data2d, dtype=Info2DCol_description
                    )
                    dest_table.append(recarray)
                    dest_table.flush()
            dest_table.attrs.has_ibo_data = True
        data_file.close()


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "--ufmfs",
        type="string",
        help=("sequence of .ufmf filenames " "(e.g. 'cam1.ufmf:cam2.ufmf')"),
    )

    parser.add_option("--view", type="string", help="how to view .ufmf files")

    parser.add_option(
        "--h5", type="string", help=".h5 file with data2d_distorted (REQUIRED)"
    )

    parser.add_option(
        "--output-h5",
        type="string",
        help="filename for output .h5 file with data2d_distorted",
    )

    parser.add_option(
        "--kalman",
        dest="kalman_filename",
        type="string",
        help=".h5 file with kalman data and 3D reconstructor",
    )

    parser.add_option(
        "--start", type="int", default=None, help="frame number to begin analysis on"
    )

    parser.add_option(
        "--stop", type="int", default=None, help="frame number to end analysis on"
    )

    parser.add_option(
        "--erode", type="int", default=0, help="amount of erosion to perform"
    )

    parser.add_option(
        "--intermediate-thresh-frac",
        type="float",
        default=0.5,
        help=(
            "accumublate pixels greater than this fraction "
            "times brightest absdiff pixel"
        ),
    )

    parser.add_option(
        "--final-thresh",
        type="int",
        default=7,
        help=(
            "clip final image to reduce moment arms before " "extracting orientation"
        ),
    )

    parser.add_option(
        "--stack-N-images",
        type="int",
        default=5,
        help=("preferred number of images to accumulate " "before reducing"),
    )

    parser.add_option(
        "--stack-N-images-min",
        type="int",
        default=5,
        help=("minimum number of images to accumulate " "before reducing"),
    )

    parser.add_option("--save-images", action="store_true", default=False)

    parser.add_option(
        "--no-rts-smoothing",
        action="store_false",
        dest="do_rts_smoothing",
        default=True,
    )

    parser.add_option("--save-image-dir", type="string", default=None)

    parser.add_option(
        "--old-sync-timestamp-source",
        action="store_true",
        default=False,
        help="use data2d['timestamp'] to find matching ufmf frame",
    )

    (options, args) = parser.parse_args()

    if options.ufmfs is None:
        raise ValueError("--ufmfs option must be specified")

    if options.h5 is None:
        raise ValueError("--h5 option must be specified")

    if options.output_h5 is None:
        raise ValueError("--output-h5 option must be specified")

    if options.kalman_filename is None:
        raise ValueError("--kalman option must be specified")

    ufmf_filenames = options.ufmfs.split(os.pathsep)
    ## print 'ufmf_filenames',ufmf_filenames
    ## print 'options.h5',options.h5

    if options.view is not None:
        view = eval(options.view)
    else:
        view = ["orig"] * len(ufmf_filenames)

    doit(
        ufmf_filenames=ufmf_filenames,
        h5_filename=options.h5,
        kalman_filename=options.kalman_filename,
        start=options.start,
        stop=options.stop,
        view=view,
        output_h5_filename=options.output_h5,
        erode=options.erode,
        save_images=options.save_images,
        save_image_dir=options.save_image_dir,
        intermediate_thresh_frac=options.intermediate_thresh_frac,
        final_thresh=options.final_thresh,
        stack_N_images=options.stack_N_images,
        stack_N_images_min=options.stack_N_images_min,
        old_sync_timestamp_source=options.old_sync_timestamp_source,
        do_rts_smoothing=options.do_rts_smoothing,
    )


if __name__ == "__main__":
    main()
