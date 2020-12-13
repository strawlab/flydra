from __future__ import division
from __future__ import print_function
import numpy
import tables as PT
import scipy.io
import sys, math
import tables.flavor
from flydra_analysis.analysis.save_as_flydra_hdf5 import save_as_flydra_hdf5

tables.flavor.restrict_flavors(keep=["numpy"])


def main():
    filename = sys.argv[1]
    do_it(filename=filename)


def get_valid_userblock_size(min):
    result = 2 ** int(math.ceil(math.log(min, 2)))
    if result < 512:
        result = 512
    return result


def do_it(
    filename=None,
    rows=None,
    ignore_observations=False,
    min_num_observations=10,
    newfilename=None,
    extra_vars=None,
    orientation_quality=None,
    hdf5=False,
    tzname=None,
    fps=None,
    smoothed_source=None,
    smoothed_data_filename=None,
    raw_data_filename=None,
    dynamic_model_name=None,
    recording_flydra_version=None,
    smoothing_flydra_version=None,
):

    if hdf5:
        import h5py

        assert tzname is not None
        assert fps is not None

    if filename is None and rows is None:
        raise ValueError("either filename or rows must be set")

    if filename is not None and rows is not None:
        raise ValueError("either filename or rows must be set, but not both")

    if extra_vars is None:
        extra_vars = {}

    if filename is not None:
        kresults = PT.open_file(filename, mode="r")
        print("reading files...")
        table1 = kresults.root.kalman_estimates.read()
        if not ignore_observations:
            table2 = kresults.root.ML_estimates.read()
        print("done.")
        kresults.close()
        del kresults

    if rows is not None:
        table1 = rows
        if not ignore_observations:
            raise ValueError("no observations can be saved in rows mode")

    if not ignore_observations:
        obj_ids = table1["obj_id"]
        obj_ids = numpy.unique(obj_ids)

        obs_cond = None
        k_cond = None

        for obj_id in obj_ids:
            this_obs_cond = table2["obj_id"] == obj_id
            n_observations = numpy.sum(this_obs_cond)
            if n_observations > min_num_observations:
                if obs_cond is None:
                    obs_cond = this_obs_cond
                else:
                    obs_cond = obs_cond | this_obs_cond

                this_k_cond = table1["obj_id"] == obj_id
                if k_cond is None:
                    k_cond = this_k_cond
                else:
                    k_cond = k_cond | this_k_cond

        table1 = table1[k_cond]
        table2 = table2[obs_cond]

        if newfilename is None:
            if hdf5:
                newfilename = filename + "-short-only.h5"
            else:
                newfilename = filename + "-short-only.mat"
    else:
        if newfilename is None:
            if hdf5:
                newfilename = filename + ".h5"
            else:
                newfilename = filename + ".mat"

    data = dict(
        kalman_obj_id=table1["obj_id"],
        kalman_frame=table1["frame"],
        kalman_x=table1["x"],
        kalman_y=table1["y"],
        kalman_z=table1["z"],
        kalman_xvel=table1["xvel"],
        kalman_yvel=table1["yvel"],
        kalman_zvel=table1["zvel"],
        P00=table1["P00"],
        P11=table1["P11"],
        P22=table1["P22"],
    )

    if orientation_quality is not None:
        assert len(orientation_quality) == len(data["kalman_obj_id"])
        data["orientation_quality"] = orientation_quality

    # save (un-smoothed) orientation data if available
    if "rawdir_x" in table1.dtype.fields:
        for d in ("rawdir_x", "rawdir_y", "rawdir_z"):
            data[d] = table1[d]

    # save smoothed orientation data if available
    if "dir_x" in table1.dtype.fields:
        for d in ("dir_x", "dir_y", "dir_z"):
            data[d] = table1[d]

    if "xaccel" in table1:
        # acceleration state not in newer dynamic models
        dict2 = dict(
            kalman_xaccel=table1["xaccel"],
            kalman_yaccel=table1["yaccel"],
            kalman_zaccel=table1["zaccel"],
        )
        data.update(dict2)

    if not ignore_observations:
        extra = dict(
            observation_obj_id=table2["obj_id"],
            observation_frame=table2["frame"],
            observation_x=table2["x"],
            observation_y=table2["y"],
            observation_z=table2["z"],
        )
        data.update(extra)

    if 0:
        print("converting int32 to float64 to avoid scipy.io.savemat bug")
        for key in data:
            # print 'converting field',key, data[key].dtype, data[key].dtype.char
            if data[key].dtype.char == "l":
                data[key] = data[key].astype(numpy.float64)

    for key, value in extra_vars.items():
        if key in data:
            print(
                "WARNING: requested to save extra variable %s, but already in data, not overwriting"
                % key
            )
            continue
        data[key] = value

    if hdf5:
        table_info = {
            "trajectories": [
                ("kalman_obj_id", "obj_id"),
                ("kalman_frame", "framenumber"),
                ("kalman_x", "x"),
                ("kalman_y", "y"),
                ("kalman_z", "z"),
                ("P00", "covariance_x"),
                ("P11", "covariance_y"),
                ("P22", "covariance_z"),
            ],
            "trajectory_start_times": [
                ("obj_ids", "obj_id"),
                ("timestamps", "first_timestamp_secs"),
                ("timestamps", "first_timestamp_nsecs"),
            ],
            "experiment_info": [("experiment_uuid", "uuid")],
        }
        # gather data
        data_dict = {}

        for table_name in table_info:
            colnames = table_info[table_name]
            dtype_elements = []
            num_rows = None
            for orig_colname, new_colname in colnames:
                if new_colname.endswith("_secs") or new_colname.endswith("_nsecs"):
                    dtype_elements.append((new_colname, numpy.uint64))
                else:
                    if orig_colname not in data:
                        # do not do this column
                        continue
                    dtype_elements.append((new_colname, data[orig_colname].dtype))
                assert data[orig_colname].ndim == 1
                if num_rows is None:
                    num_rows = data[orig_colname].shape[0]
                else:
                    assert num_rows == data[orig_colname].shape[0]
            if len(dtype_elements) == 0:
                # do not save this table
                continue
            my_dtype = numpy.dtype(dtype_elements)
            arr = numpy.empty(num_rows, dtype=my_dtype)
            for orig_colname, new_colname in colnames:
                if new_colname.endswith("_secs"):
                    timestamps = data[orig_colname]
                    arr[new_colname] = numpy.floor(timestamps).astype(numpy.uint64)
                elif new_colname.endswith("_nsecs"):
                    timestamps = data[orig_colname]
                    arr[new_colname] = (numpy.mod(timestamps, 1.0) * 1e9).astype(
                        numpy.uint64
                    )
                else:
                    arr[new_colname] = data[orig_colname]

            if len(arr) > 0:  # don't save empty data (h5py doesn't like it)
                data_dict[table_name] = arr

        # save as h5 file
        save_as_flydra_hdf5(
            newfilename,
            data_dict,
            tzname,
            fps,
            smoothed_source=smoothed_source,
            smoothed_data_filename=smoothed_data_filename,
            raw_data_filename=raw_data_filename,
            dynamic_model_name=dynamic_model_name,
            recording_flydra_version=recording_flydra_version,
            smoothing_flydra_version=smoothing_flydra_version,
        )

    else:
        scipy.io.savemat(newfilename, data, appendmat=False)


if __name__ == "__main__":
    print("WARNING: are you sure you want to run this program and not 'data2smoothed'?")
    main()
