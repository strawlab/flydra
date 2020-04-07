from __future__ import with_statement
from __future__ import absolute_import
import motmot.ufmf.ufmf as ufmf_mod
import motmot.FlyMovieFormat.FlyMovieFormat as fmf_mod
import sys, os, tempfile, re, contextlib, warnings, time
from optparse import OptionParser
import flydra_analysis.a2.auto_discover_ufmfs as auto_discover_ufmfs
import numpy as np
import tables
import flydra_analysis.a2.utils as utils
import flydra_analysis.analysis.result_utils as result_utils
import scipy.misc
import subprocess
import motmot.imops.imops as imops

from .tables_tools import open_file_safe

ufmf_fname_regex = re.compile(r"small_([0-9]+)_([0-9]+)_(.*)")


def get_cam_id_from_ufmf_fname(ufmf_fname):
    cam_id = None

    # cam_id in fname (2008 flydra approach) -----------------
    a = os.path.split(ufmf_fname)[-1]
    b, mode = os.path.splitext(a)
    mode = mode[1:]
    assert mode in ["ufmf", "fmf"]
    matchobj = ufmf_fname_regex.search(b)
    if matchobj is not None:
        date, datetime_str, cam_id = matchobj.groups()

    # cam_id in path (2013 flydra approach ) -----------------------
    if cam_id is None:
        base1, filepart = os.path.split(ufmf_fname)
        base2, maybe_camid = os.path.split(base1)
        base3, datetime_str = os.path.split(base2)
        try:
            struct_time = time.strptime(datetime_str, "%Y%m%d_%H%M%S")
            approx_start = time.mktime(struct_time)
        except RuntimeError:
            pass
        except ValueError:
            datetime_str2 = filepart
            struct_time = time.strptime(datetime_str2, "%Y%m%d_%H%M%S." + mode)
            # if the above succeeded, it was a timestamp. thus we have 2013-08 version.
            cam_id = maybe_camid
        else:
            # if the parent directory is datetime, this is likely the camid
            if os.path.splitext(filepart)[0] == datetime_str:
                cam_id = maybe_camid

    if cam_id is None:
        raise ValueError("could not guess cam_id from filename %s" % (ufmf_fname,))

    return cam_id


frames_by_fmf = {}


def lru_cache_get_frame(f, idx):
    global frames_by_fmf

    # get cache (or create it)
    try:
        cached = frames_by_fmf[f]
    except KeyError:
        cached = {
            "idx": None,
        }
        frames_by_fmf[f] = cached

    if cached["idx"] != idx:
        # cache miss
        f.seek(idx)
        return_value = f.get_next_frame()
        cached["idx"] = idx
        cached["return_value"] = return_value

    return cached["return_value"]


def fill_more_for(extra, image_ts):
    if "bg_tss" not in extra:
        return None
    more = {}
    bg_tss = extra["bg_tss"]
    bg_fmf = extra["bg_fmf"]
    idxs = np.nonzero((bg_tss >= image_ts))[0]
    if len(idxs) < 1:
        return None
    idx = idxs[0]
    image, image_ts = lru_cache_get_frame(bg_fmf, idx)
    more["mean"] = image
    return more


def iterate_frames(
    h5_filename,
    ufmf_fnames,  # or fmfs
    white_background=False,
    max_n_frames=None,
    start=None,
    stop=None,
    rgb8_if_color=False,
    movie_cam_ids=None,
    camn2cam_id=None,
):
    """yield frame-by-frame data"""

    # First pass over .ufmf files: get intersection of timestamps
    first_ufmf_ts = -np.inf
    last_ufmf_ts = np.inf
    ufmfs = {}
    cam_ids = []
    global_data = {"width_heights": {}}
    for movie_idx, ufmf_fname in enumerate(ufmf_fnames):
        if movie_cam_ids is not None:
            cam_id = movie_cam_ids[movie_idx]
        else:
            cam_id = get_cam_id_from_ufmf_fname(ufmf_fname)
        cam_ids.append(cam_id)
        kwargs = {}
        extra = {}
        if ufmf_fname.lower().endswith(".fmf"):
            ufmf = fmf_mod.FlyMovie(ufmf_fname)
            bg_fmf_filename = os.path.splitext(ufmf_fname)[0] + "_mean.fmf"
            if os.path.exists(bg_fmf_filename):
                extra["bg_fmf"] = fmf_mod.FlyMovie(bg_fmf_filename)
                extra["bg_tss"] = extra["bg_fmf"].get_all_timestamps()
                extra["bg_fmf"].seek(0)

        else:
            ufmf = ufmf_mod.FlyMovieEmulator(
                ufmf_fname, white_background=white_background, **kwargs
            )

        global_data["width_heights"][cam_id] = (ufmf.get_width(), ufmf.get_height())
        tss = ufmf.get_all_timestamps()
        ufmf.seek(0)
        ufmfs[ufmf_fname] = (ufmf, cam_id, tss, extra)
        min_ts = np.min(tss)
        max_ts = np.max(tss)
        if min_ts > first_ufmf_ts:
            first_ufmf_ts = min_ts
        if max_ts < last_ufmf_ts:
            last_ufmf_ts = max_ts

    assert first_ufmf_ts < last_ufmf_ts, ".ufmf files don't all overlap in time"

    ufmf_fnames.sort()
    cam_ids.sort()

    with open_file_safe(h5_filename, mode="r") as h5:
        if camn2cam_id is None:
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        parsed = result_utils.read_textlog_header(h5)
        flydra_version = parsed.get("flydra_version", None)
        if flydra_version is not None and flydra_version >= "0.4.45":
            # camnode.py saved timestamps into .ufmf file given by
            # time.time() (camn_receive_timestamp). Compare with
            # mainbrain's data2d_distorted column
            # 'cam_received_timestamp'.
            old_camera_timestamp_source = False
            timestamp_name = "cam_received_timestamp"
        else:
            # camnode.py saved timestamps into .ufmf file given by
            # camera driver. Compare with mainbrain's data2d_distorted
            # column 'timestamp'.
            old_camera_timestamp_source = True
            timestamp_name = "timestamp"

        h5_data = h5.root.data2d_distorted[:]

    if 1:
        # narrow search to local region of .h5
        cond = (first_ufmf_ts <= h5_data[timestamp_name]) & (
            h5_data[timestamp_name] <= last_ufmf_ts
        )
        narrow_h5_data = h5_data[cond]

        narrow_camns = narrow_h5_data["camn"]
        narrow_timestamps = narrow_h5_data[timestamp_name]

        # Find the camn for each .ufmf file
        cam_id2camn = {}
        for cam_id in cam_ids:
            cam_id_camn_already_found = False
            for ufmf_fname in ufmfs.keys():
                (ufmf, test_cam_id, tss, extra) = ufmfs[ufmf_fname]
                if cam_id != test_cam_id:
                    continue
                assert not cam_id_camn_already_found
                cam_id_camn_already_found = True

                umin = np.min(tss)
                umax = np.max(tss)
                cond = (umin <= narrow_timestamps) & (narrow_timestamps <= umax)
                ucamns = narrow_camns[cond]
                ucamns = np.unique(ucamns)
                camns = []
                for camn in ucamns:
                    if camn2cam_id[camn] == cam_id:
                        camns.append(camn)

                assert len(camns) < 2, "can't handle multiple camns per cam_id"
                if len(camns):
                    cam_id2camn[cam_id] = camns[0]

        ff = utils.FastFinder(narrow_h5_data["frame"])
        unique_frames = list(np.unique(narrow_h5_data["frame"]))
        unique_frames.sort()
        unique_frames = np.array(unique_frames)
        if start is not None:
            unique_frames = unique_frames[unique_frames >= start]
        if stop is not None:
            unique_frames = unique_frames[unique_frames <= stop]

        if max_n_frames is not None:
            unique_frames = unique_frames[:max_n_frames]
        for frame_enum, frame in enumerate(unique_frames):
            narrow_idxs = ff.get_idxs_of_equal(frame)

            # trim data under consideration to just this frame
            this_h5_data = narrow_h5_data[narrow_idxs]
            this_camns = this_h5_data["camn"]
            this_tss = this_h5_data[timestamp_name]

            # a couple more checks
            if np.any(this_tss < first_ufmf_ts):
                continue
            if np.any(this_tss >= last_ufmf_ts):
                break

            per_frame_dict = {}
            for ufmf_fname in ufmf_fnames:
                ufmf, cam_id, tss, extra = ufmfs[ufmf_fname]
                if cam_id not in cam_id2camn:
                    continue
                camn = cam_id2camn[cam_id]
                this_camn_cond = this_camns == camn
                this_cam_h5_data = this_h5_data[this_camn_cond]
                this_camn_tss = this_cam_h5_data[timestamp_name]
                if not len(this_camn_tss):
                    # no h5 data for this cam_id at this frame
                    continue
                this_camn_ts = np.unique(this_camn_tss)
                assert len(this_camn_ts) == 1
                this_camn_ts = this_camn_ts[0]

                if isinstance(ufmf, ufmf_mod.FlyMovieEmulator):
                    is_real_ufmf = True
                else:
                    is_real_ufmf = False

                # optimistic: get next frame. it's probably the one we want
                try:
                    if is_real_ufmf:
                        image, image_ts, more = ufmf.get_next_frame(_return_more=True)
                    else:
                        image, image_ts = ufmf.get_next_frame()
                        more = fill_more_for(extra, image_ts)
                except ufmf_mod.NoMoreFramesException:
                    image_ts = None
                if this_camn_ts != image_ts:
                    # It was not the frame we wanted. Find it.
                    ufmf_frame_idxs = np.nonzero(tss == this_camn_ts)[0]
                    if len(ufmf_frame_idxs) == 0 and old_camera_timestamp_source:
                        warnings.warn(
                            "low-precision timestamp comparison in "
                            "use due to outdated .ufmf timestamp "
                            "saving."
                        )
                        # 2.5 msec precision required
                        ufmf_frame_idxs = np.nonzero(abs(tss - this_camn_ts) < 0.0025)[
                            0
                        ]
                    assert len(ufmf_frame_idxs) == 1
                    ufmf_frame_no = ufmf_frame_idxs[0]
                    if is_real_ufmf:
                        image, image_ts, more = ufmf.get_frame(
                            ufmf_frame_no, _return_more=True
                        )
                    else:
                        image, image_ts = ufmf.get_frame(ufmf_frame_no)
                        more = fill_more_for(extra, image_ts)

                    del ufmf_frame_no, ufmf_frame_idxs
                coding = ufmf.get_format()
                if imops.is_coding_color(coding):
                    if rgb8_if_color:
                        image = imops.to_rgb8(coding, image)
                    else:
                        warnings.warn("color image not converted to color")
                per_frame_dict[ufmf_fname] = {
                    "image": image,
                    "cam_id": cam_id,
                    "camn": camn,
                    "timestamp": this_cam_h5_data["timestamp"][0],
                    "cam_received_timestamp": this_cam_h5_data[
                        "cam_received_timestamp"
                    ][0],
                    "ufmf_frame_timestamp": this_cam_h5_data[timestamp_name][0],
                }
                if more is not None:
                    per_frame_dict[ufmf_fname].update(more)
            per_frame_dict["tracker_data"] = this_h5_data
            per_frame_dict[
                "global_data"
            ] = global_data  # on every iteration, pass our global data
            yield (per_frame_dict, frame)
