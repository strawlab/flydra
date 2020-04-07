from __future__ import print_function
import argparse
import tables
import math
import struct
import ctypes
import numpy as np

import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils
import flydra_core.reconstruct


def _get_struct_fmt(cloud, field_names=None):
    from sensor_msgs.msg import PointField

    _DATATYPES = {}
    _DATATYPES[PointField.INT8] = ("b", 1)
    _DATATYPES[PointField.UINT8] = ("B", 1)
    _DATATYPES[PointField.INT16] = ("h", 2)
    _DATATYPES[PointField.UINT16] = ("H", 2)
    _DATATYPES[PointField.INT32] = ("i", 4)
    _DATATYPES[PointField.UINT32] = ("I", 4)
    _DATATYPES[PointField.FLOAT32] = ("f", 4)
    _DATATYPES[PointField.FLOAT64] = ("d", 8)

    # originally from https://code.ros.org/trac/ros-pkg/attachment/ticket/4440/point_cloud.py
    fmt = ">" if cloud.is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(cloud.fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    assert cloud
    fmt = _get_struct_fmt(cloud, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in xrange(height):
                offset = row_step * v
                for u in xrange(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in xrange(height):
                offset = row_step * v
                for u in xrange(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def convert_to_flydrah5(
    bag_file, topic_name="pointcloud", out_h5=None, reconstructor=None
):
    import rosbag

    if out_h5 is None:
        out_h5 = bag_file + ".h5"

    bag = rosbag.Bag(bag_file, "r")

    h5file = tables.open_file(out_h5, mode="w", title="Flydra data file (from ROS bag)")
    ct = h5file.create_table  # shorthand
    root = h5file.root  # shorthand

    # save data as both "observations" (ML estimates) and "kalman estimates" (MAP estimates)

    FilteredObservations = flydra_kalman_utils.FilteredObservations
    h5data3d_ML_estimates = ct(
        root, "ML_estimates", FilteredObservations, "3d data (input to Kalman filter)"
    )
    h5_obs_names = tables.Description(FilteredObservations().columns)._v_names

    # we're not actually doing any kalman filtering, so just get a model with position
    kalman_saver_info_instance = flydra_kalman_utils.KalmanSaveInfo(
        name="mamarama, units: mm"
    )
    KalmanEstimatesDescription = kalman_saver_info_instance.get_description()
    h5data3d_kalman_estimates = ct(
        root, "kalman_estimates", KalmanEstimatesDescription, "3d data"
    )
    h5_xhat_names = tables.Description(KalmanEstimatesDescription().columns)._v_names

    obj_id = 0
    for topic, cloud, t in bag.read_messages(topics=[topic_name]):

        pts = []
        for p in read_points(cloud):
            pts.append((p[0], p[1], p[2], 0, 0, 0))  # velocity = 0
        obj_id += 1
        pts = np.array(pts)

        # save observations
        shape1d = (len(pts),)
        shape2d_6 = (len(pts), 6)
        this_idxs = np.zeros(shape1d, dtype=np.uint64)
        # this_idxs = numpy.array( this_idxs, dtype=numpy.uint64 ) # becomes obs_2d_idx (index into 'ML_estimates_2d_idxs')

        observations_frames = np.arange(len(pts), dtype=np.uint64)
        obj_id_array = np.empty(observations_frames.shape, dtype=np.uint32)
        obj_id_array.fill(obj_id)
        observations_data = np.array(pts[:, :3], dtype=np.float32)
        observations_Lcoords = np.zeros(shape2d_6, dtype=np.float32)
        list_of_obs = [
            observations_data[:, i] for i in range(observations_data.shape[1])
        ]
        list_of_lines = [
            observations_Lcoords[:, i] for i in range(observations_Lcoords.shape[1])
        ]
        array_list = (
            [obj_id_array, observations_frames]
            + list_of_obs
            + [this_idxs]
            + list_of_lines
        )
        obs_recarray = np.rec.fromarrays(array_list, names=h5_obs_names)

        h5data3d_ML_estimates.append(obs_recarray)
        h5data3d_ML_estimates.flush()

        # save xhat info (kalman estimates)
        shape3d_6x6 = (len(pts), 6, 6)
        frames = np.arange(len(pts), dtype=np.uint64)
        timestamps = np.zeros(shape1d, dtype=np.float64)
        xhat_data = np.array(pts, dtype=np.float32)
        P_data_full = np.zeros(shape3d_6x6, dtype=np.float32)
        obj_id_array = np.empty(frames.shape, dtype=np.uint32)
        obj_id_array.fill(obj_id)
        list_of_xhats = [xhat_data[:, i] for i in range(xhat_data.shape[1])]
        ksii = kalman_saver_info_instance
        list_of_Ps = ksii.covar_mats_to_covar_entries(P_data_full)
        xhats_recarray = np.rec.fromarrays(
            [obj_id_array, frames, timestamps] + list_of_xhats + list_of_Ps,
            names=h5_xhat_names,
        )

        h5data3d_kalman_estimates.append(xhats_recarray)
        h5data3d_kalman_estimates.flush()

    if reconstructor is not None:
        R = flydra_core.reconstruct.Reconstructor(reconstructor)
        R.save_to_h5file(h5file)
    h5file.close()
    bag.close()


def main():
    # defer import to allow this module to be imported without ROS installed (for nose tests)
    import roslib

    roslib.load_manifest("sensor_msgs")

    parser = argparse.ArgumentParser()
    parser.add_argument("bag_file")
    parser.add_argument("--topic", default="pointcloud")
    parser.add_argument("--out_h5")
    parser.add_argument("--reconstructor")
    args = parser.parse_args()
    convert_to_flydrah5(
        args.bag_file,
        topic_name=args.topic,
        out_h5=args.out_h5,
        reconstructor=args.reconstructor,
    )


if __name__ == "__main__":
    main()
