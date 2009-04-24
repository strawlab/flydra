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

# 2D data format for PyTables:
class Info2D(PT.IsDescription):
    camn         = PT.UInt16Col(pos=0)
    frame        = PT.UInt64Col(pos=1)
    timestamp    = PT.FloatCol(pos=2) # when the image trigger happened (returned by low-level camera driver)
    cam_received_timestamp  = PT.FloatCol(pos=3) # when the image was acquired by flydra software (on camera computer)
    x            = PT.Float32Col(pos=4)
    y            = PT.Float32Col(pos=5)
    area         = PT.Float32Col(pos=6)
    slope        = PT.Float32Col(pos=7)
    eccentricity = PT.Float32Col(pos=8)
    frame_pt_idx = PT.UInt8Col(pos=9) # index of point if there were > 1 points in frame
    cur_val      = PT.UInt8Col(pos=10)
    mean_val     = PT.Float32Col(pos=11)
    sumsqf_val   = PT.Float32Col(pos=12) # estimate of <x^2> (running_sumsqf)
