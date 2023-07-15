import tables as PT

# This describes indexes in the "pt_undistorted" tuple.  These are
# used in MainBrain.py, flydra_tracker.py, and kalmanize.py
PT_TUPLE_IDX_X = 0
PT_TUPLE_IDX_Y = 1
PT_TUPLE_IDX_AREA = 2
PT_TUPLE_IDX_SLOPE = 3
PT_TUPLE_IDX_ECCENTRICITY = 4
# 3D coordinates of plane formed by camera center and slope line
# centered on object.
PT_TUPLE_IDX_P1 = 5
PT_TUPLE_IDX_P2 = 6
PT_TUPLE_IDX_P3 = 7
PT_TUPLE_IDX_P4 = 8
PT_TUPLE_IDX_LINE_FOUND = 9
PT_TUPLE_IDX_FRAME_PT_IDX = 10
PT_TUPLE_IDX_CUR_VAL_IDX = 11
PT_TUPLE_IDX_MEAN_VAL_IDX = 12
PT_TUPLE_IDX_SUMSQF_VAL_IDX = 13

WIRE_ORDER_CUR_VAL_IDX = 6
WIRE_ORDER_MEAN_VAL_IDX = 7
WIRE_ORDER_SUMSQF_VAL_IDX = 8

# 2D data format for PyTables:
class Info2D(PT.IsDescription):
    camn = PT.UInt16Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.FloatCol(
        pos=2
    )  # when the image trigger happened (returned by timestamp modeler on MainBrain)
    cam_received_timestamp = PT.FloatCol(
        pos=3
    )  # when the image was acquired by flydra software (on camera computer)
    x = PT.Float32Col(pos=4)
    y = PT.Float32Col(pos=5)
    area = PT.Float32Col(pos=6)
    slope = PT.Float32Col(pos=7)
    eccentricity = PT.Float32Col(pos=8)
    frame_pt_idx = PT.UInt8Col(
        pos=9
    )  # index of point if there were > 1 points in frame
    cur_val = PT.UInt8Col(pos=10)
    mean_val = PT.Float32Col(pos=11)
    sumsqf_val = PT.Float32Col(pos=12)  # estimate of <x^2> (running_sumsqf)


class TextLogDescription(PT.IsDescription):
    mainbrain_timestamp = PT.FloatCol(pos=0)
    cam_id = PT.StringCol(255, pos=1)
    host_timestamp = PT.FloatCol(pos=2)
    message = PT.StringCol(4*1024*1024, pos=3) # 4 MB message max


class CamSyncInfo(PT.IsDescription):
    cam_id = PT.StringCol(256, pos=0)
    camn = PT.UInt16Col(pos=1)
    hostname = PT.StringCol(2048, pos=2)


class HostClockInfo(PT.IsDescription):
    remote_hostname = PT.StringCol(255, pos=0)
    start_timestamp = PT.FloatCol(pos=1)
    remote_timestamp = PT.FloatCol(pos=2)
    stop_timestamp = PT.FloatCol(pos=3)


class TriggerClockInfo(PT.IsDescription):
    start_timestamp = PT.FloatCol(pos=0)
    framecount = PT.Int64Col(pos=1)
    tcnt = PT.UInt16Col(pos=2)
    stop_timestamp = PT.FloatCol(pos=3)


class MovieInfo(PT.IsDescription):
    cam_id = PT.StringCol(16, pos=0)
    filename = PT.StringCol(255, pos=1)
    approx_start_frame = PT.Int64Col(pos=2)
    approx_stop_frame = PT.Int64Col(pos=3)


class ExperimentInfo(PT.IsDescription):
    uuid = PT.StringCol(32, pos=0)
