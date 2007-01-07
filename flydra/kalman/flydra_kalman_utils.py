import tables as PT
import numpy
import flydra.data_descriptions

PT_TUPLE_IDX_FRAME_PT_IDX = flydra.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX

class KalmanEstimates(PT.IsDescription):
    obj_id     = PT.UInt32Col(pos=0,indexed=True)
    frame      = PT.UInt64Col(pos=1)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    xvel       = PT.Float32Col(pos=5)
    yvel       = PT.Float32Col(pos=6)
    zvel       = PT.Float32Col(pos=7)
    xaccel     = PT.Float32Col(pos=8)
    yaccel     = PT.Float32Col(pos=9)
    zaccel     = PT.Float32Col(pos=10)
    # save diagonal of P matrix
    P00        = PT.Float32Col(pos=11)
    P11        = PT.Float32Col(pos=11)
    P22        = PT.Float32Col(pos=11)
    P33        = PT.Float32Col(pos=11)
    P44        = PT.Float32Col(pos=11)
    P55        = PT.Float32Col(pos=11)
    P66        = PT.Float32Col(pos=11)
    P77        = PT.Float32Col(pos=11)
    P88        = PT.Float32Col(pos=11)

class FilteredObservations(PT.IsDescription):
    obj_id     = PT.UInt32Col(pos=0,indexed=True)
    frame      = PT.UInt64Col(pos=1,indexed=True)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    obs_2d_idx = PT.UInt64Col(pos=5) # index into VLArray 'kalman_observations_2d_idxs'
    
def convert_format(current_data,camn2cam_id):
    """convert data from format used for Kalman tracker to hypothesis tester"""
    found_data_dict = {}
    first_idx_by_camn = {}
    for camn, stuff_list in current_data.iteritems():
        if not len(stuff_list):
            # no data for this camera, continue
            continue
        this_point,projected_line = stuff_list[0] # algorithm only accepts 1 point per camera
        if not numpy.isnan(this_point[0]): # only use if point was found
            cam_id = camn2cam_id[camn]
            found_data_dict[cam_id] = this_point[:9]
            first_idx_by_camn[camn] = this_point[PT_TUPLE_IDX_FRAME_PT_IDX]
    return found_data_dict, first_idx_by_camn
