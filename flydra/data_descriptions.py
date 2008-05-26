import tables as PT

# This describes indexes in the "pt_undistorted" tuple.
# These are used in MainBrain.py and flydra_tracker.py
PT_TUPLE_IDX_X = 0
PT_TUPLE_IDX_Y = 1
PT_TUPLE_IDX_AREA = 2
PT_TUPLE_IDX_FRAME_PT_IDX = 10
PT_TUPLE_IDX_CUR_VAL_IDX = 11
PT_TUPLE_IDX_MEAN_VAL_IDX = 12
PT_TUPLE_IDX_NSTD_VAL_IDX = 13

# 2D data format for PyTables:
class Info2D(PT.IsDescription):
    camn         = PT.UInt16Col(pos=0)
    frame        = PT.UInt64Col(pos=1)
    timestamp    = PT.FloatCol(pos=2)
    cam_received_timestamp  = PT.FloatCol(pos=3)
    x            = PT.Float32Col(pos=4)
    y            = PT.Float32Col(pos=5)
    area         = PT.Float32Col(pos=6)
    slope        = PT.Float32Col(pos=7)
    eccentricity = PT.Float32Col(pos=8)
    frame_pt_idx = PT.UInt8Col(pos=9)
    cur_val      = PT.UInt8Col(pos=10)
    mean_val     = PT.UInt8Col(pos=11)
    nstd_val     = PT.UInt8Col(pos=12)
