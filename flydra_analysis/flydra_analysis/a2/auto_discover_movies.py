from __future__ import print_function
import os, glob, time
from .auto_discover_ufmfs import get_h5_start_stop
import tables
import numpy as np

DEFAULT_MOVIE_SUBDIR = "~/FLYDRA_MOVIES"


def get_uuid(filename):
    with tables.open_file(filename, mode="r") as h5file_raw:
        table_experiment = h5file_raw.root.experiment_info
        result = table_experiment.read(field="uuid")
    uuids = np.unique(result)
    if len(uuids) == 0:
        return None
    assert len(uuids) == 1
    uuid = uuids[0]
    return uuid


def find_movies(h5_fname, ufmf_dir=None, verbose=False, candidate_index=0):
    """find movies (.ufmf or .fmf) which are in canonical location"""
    h5_start, h5_stop = get_h5_start_stop(h5_fname)
    uuid = get_uuid(h5_fname)
    if verbose:
        print("h5_start, h5_stop", h5_start, h5_stop)
    if ufmf_dir is None:
        test_path = os.path.expanduser(DEFAULT_MOVIE_SUBDIR)
    else:
        test_path = ufmf_dir
    if verbose:
        print("find_movies test_path: %r" % (test_path,))

    if uuid is not None:
        uuid_path = os.path.join(test_path, uuid)
        if verbose:
            print("find_movies: looking for uuid path", uuid_path)

        if os.path.exists(uuid_path):
            # new code path - movies saved in "<DEFAULT_MOVIE_SUBDIR>/<uuid>/<cam_id>/<time>.fmf"
            if verbose:
                print("find_movies: finding by uuid")
            return find_movies_uuid(
                h5_fname, verbose=verbose, pick_number=candidate_index
            )
        if verbose:
            print("find_movies: no uuid path found. It would be %r" % (uuid_path,))
    else:
        if verbose:
            print("find_movies: no uuid")

    maybe_dirnames = glob.glob(os.path.join(test_path, "*"))

    candidates = []

    for maybe_dirname in maybe_dirnames:
        if not os.path.isdir(maybe_dirname):
            continue  # not a directory
        basename = os.path.split(maybe_dirname)[-1]

        struct_time = time.strptime(basename, "%Y%m%d_%H%M%S")
        approx_start = time.mktime(struct_time)
        if verbose:
            print("  option: %s: %s" % (maybe_dirname, approx_start))

        if (h5_start <= approx_start) and (h5_stop >= approx_start):
            if verbose:
                print("    valid!")
            candidates.append((maybe_dirname, basename))

    if len(candidates) == 0:
        return []

    candidate = candidates[candidate_index]
    (dirname, basename) = candidate

    if verbose and len(candidates) > 1:
        print(
            "%d candidate movies available, choosing %s/%s"
            % (len(candidates), dirname, basename)
        )

    cam_ids = os.listdir(dirname)
    results = []
    mode = None
    for cam_id in cam_ids:
        camdir = os.path.join(dirname, cam_id)
        camfiles = os.listdir(camdir)
        if mode == "fmf" or mode == "ufmf":
            fname = basename + "." + mode
            assert fname in camfiles
            results.append(os.path.join(camdir, fname))
        else:
            fname = basename + ".fmf"
            if fname in camfiles:
                mode = "fmf"
                results.append(os.path.join(camdir, fname))
            else:
                fname = basename + ".ufmf"
                if fname in camfiles:
                    mode = "ufmf"
                    results.append(os.path.join(camdir, fname))
    if verbose:
        for r in results:
            print(r)
    return results


def find_movies_uuid(h5_fname, pick_number=None, verbose=True):
    uuid = get_uuid(h5_fname)

    test_path = os.path.expanduser(DEFAULT_MOVIE_SUBDIR)

    candidates = []

    cam_ids = glob.glob(os.path.join(test_path, uuid, "*"))
    if verbose:
        print("putative cam_ids", cam_ids)
    for cam_id in cam_ids:
        cam_dir = os.path.join(os.path.join(test_path, uuid, cam_id))

        mean_fmf = glob.glob(os.path.join(cam_dir, "*_mean.fmf"))
        mean_ufmf = glob.glob(os.path.join(cam_dir, "*_mean.ufmf"))
        mean_files = mean_fmf + mean_ufmf

        total_length = len(mean_files)
        if total_length == 0:
            if verbose:
                print("no data for putative cam_id", cam_id)
            continue
        if not total_length == 1:
            if verbose:
                print(
                    "found %d mean files in dir %s: %s"
                    % (total_length, cam_dir, mean_files)
                )
            if pick_number is None:
                raise NotImplementedError("need to figure out what time is best...")
            else:
                print(
                    "picking movie number %d (specify with --candidate)" % pick_number
                )
                if len(mean_fmf):
                    mean_fmf = [mean_fmf[pick_number]]
                    assert len(mean_ufmf) == 0
                else:
                    mean_ufmf = [mean_ufmf[pick_number]]
                    assert len(mean_fmf) == 0
        if len(mean_fmf):
            mode = "fmf"
        else:
            mode = "ufmf"

        mean_fname = mean_files[0]
        basename = os.path.split(mean_fname)[-1]
        struct_time = time.strptime(basename, "%Y%m%d_%H%M%S_mean." + mode)
        approx_start = time.mktime(struct_time)
        main_file = time.strftime("%Y%m%d_%H%M%S." + mode, struct_time)
        candidates.append(os.path.join(cam_dir, main_file))
    return candidates
