from distutils.version import LooseVersion
import tables as PT
import tables
import flydra_core.data_descriptions
from flydra_core.reconstruct import Reconstructor
import time
import flydra_analysis.version

Info2D = flydra_core.data_descriptions.Info2D
Info2DCol_description = tables.Description(Info2D().columns)._v_nested_descr
CamSyncInfo = flydra_core.data_descriptions.CamSyncInfo
TextLogDescription = flydra_core.data_descriptions.TextLogDescription


def startup_message(h5textlog, fps):
    textlog_row = h5textlog.row
    cam_id = "mainbrain"
    timestamp = time.time()

    # This line is important (including the formatting). It is
    # read by flydra_analysis.a2.check_atmel_clock.

    list_of_textlog_data = [
        (
            timestamp,
            cam_id,
            timestamp,
            (
                "MainBrain running at %s fps, "
                "(flydra_version %s, "
                "time_tzname0 %s)"
                % (str(fps), flydra_analysis.version.__version__, time.tzname[0],)
            ),
        ),
        (
            timestamp,
            cam_id,
            timestamp,
            "using flydra version %s" % (flydra_analysis.version.__version__,),
        ),
    ]
    for textlog_data in list_of_textlog_data:
        (mainbrain_timestamp, cam_id, host_timestamp, message) = textlog_data
        textlog_row["mainbrain_timestamp"] = mainbrain_timestamp
        textlog_row["cam_id"] = cam_id
        textlog_row["host_timestamp"] = host_timestamp
        textlog_row["message"] = message
        textlog_row.append()

    h5textlog.flush()


def save_data(fname, data2d, reconstructor, fps, eccentricity):
    assert isinstance(reconstructor, Reconstructor)
    with PT.open_file(fname, mode="w", title="Flydra data file") as h5file:
        reconstructor.save_to_h5file(h5file)

        ct = h5file.create_table  # shorthand
        root = h5file.root  # shorthand
        h5textlog = ct(root, "textlog", TextLogDescription, "text log")

        startup_message(h5textlog=h5textlog, fps=fps)

        t = data2d["t"]

        h5cam_info = ct(root, "cam_info", CamSyncInfo, "Cam Sync Info")
        cam_info_row = h5cam_info.row

        cam_id2camn = {}
        for camn, cam_id in enumerate(data2d["2d_pos_by_cam_ids"].keys()):
            cam_id2camn[cam_id] = camn
            cam_info_row["camn"] = camn
            cam_info_row["cam_id"] = cam_id
            cam_info_row.append()

        h5data2d = ct(root, "data2d_distorted", Info2D, "2d data")
        detection = h5data2d.row

        frame_pt_idx = 0
        for frame in range(len(t)):
            timestamp = t[frame]
            for cam_id in data2d["2d_pos_by_cam_ids"].keys():
                pt2d = data2d["2d_pos_by_cam_ids"][cam_id][frame]
                slope = data2d["2d_slope_by_cam_ids"][cam_id][frame]

                camn = cam_id2camn[cam_id]

                detection["camn"] = camn
                detection["frame"] = frame
                detection["timestamp"] = timestamp
                detection["cam_received_timestamp"] = timestamp
                detection["x"] = pt2d[0]
                detection["y"] = pt2d[1]
                detection["area"] = 1
                detection["slope"] = slope
                detection["eccentricity"] = eccentricity
                detection["frame_pt_idx"] = frame_pt_idx

                # fake values
                detection["cur_val"] = 123.45
                detection["mean_val"] = 1.2345
                detection["sumsqf_val"] = 1.2345

                frame_pt_idx += 1
                detection.append()

        h5data2d.flush()
