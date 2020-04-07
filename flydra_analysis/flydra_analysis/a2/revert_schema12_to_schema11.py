#!/usr/bin/env python
from __future__ import print_function
import h5py
import sys


def remove_field_name(a, name):
    # See http://stackoverflow.com/a/15577562/1633026
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


def revert_schema12_to_schema11(fname):
    readonly = False
    expected_first = '{"schema": "http://strawlab.org/schemas/flydra/1.2"}'
    new_first = '{"schema": "http://strawlab.org/schemas/flydra/1.1"}'
    lef = len(expected_first)
    assert len(new_first) == lef

    with open(fname, mode="rb") as fd:
        first = fd.read(lef)
        print("found %r" % (first,))
        if first != expected_first:
            print("schema not 1.2")
            return
        print("schema 1.2 found, reverting")

    # now we know we have a schema 1.2 file
    if readonly:
        mode = "r"
    else:
        mode = "r+"
    with h5py.File(fname, mode) as f:
        dset_orig = f["trajectories"]
        fps = dset_orig.attrs["frames_per_second"]
        smoothed_source = dset_orig.attrs.get("smoothed_source", None)
        print("fps", fps)
        print("smoothed_source", smoothed_source)
        if not readonly:
            trajs = dset_orig[:]
            print(trajs.dtype)

            trajs = remove_field_name(trajs, "covariance_x")
            trajs = remove_field_name(trajs, "covariance_y")
            trajs = remove_field_name(trajs, "covariance_z")
            print(trajs.dtype)

            del f["trajectories"]
            # f['trajectories'] = trajs
            dset = f.create_dataset(
                "trajectories", data=trajs, compression="gzip", compression_opts=9
            )
            dset.attrs["frames_per_second"] = fps
            assert dset.attrs["frames_per_second"] == fps
            if smoothed_source is not None:
                dset.attrs["smoothed_source"] = smoothed_source
                assert dset.attrs["smoothed_source"] == smoothed_source

    if not readonly:
        with open(fname, mode="r+b") as fd:
            fd.seek(0)
            fd.write(new_first)


if __name__ == "__main__":
    fname = sys.argv[1]
    revert_schema12_to_schema11(fname)
