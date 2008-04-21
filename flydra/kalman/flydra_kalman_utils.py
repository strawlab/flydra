import tables as PT
import numpy
import flydra.data_descriptions

PT_TUPLE_IDX_FRAME_PT_IDX = flydra.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX
PT_TUPLE_IDX_AREA = flydra.data_descriptions.PT_TUPLE_IDX_AREA

class KalmanEstimates(PT.IsDescription):
    obj_id     = PT.UInt32Col(pos=0)
    frame      = PT.UInt64Col(pos=1)
    timestamp  = PT.Float64Col(pos=2) # time of reconstruction
    x          = PT.Float32Col(pos=3)
    y          = PT.Float32Col(pos=4)
    z          = PT.Float32Col(pos=5)
    xvel       = PT.Float32Col(pos=6)
    yvel       = PT.Float32Col(pos=7)
    zvel       = PT.Float32Col(pos=8)
    xaccel     = PT.Float32Col(pos=9)
    yaccel     = PT.Float32Col(pos=10)
    zaccel     = PT.Float32Col(pos=11)
    # save diagonal of P matrix
    P00        = PT.Float32Col(pos=12)
    P11        = PT.Float32Col(pos=13)
    P22        = PT.Float32Col(pos=14)
    P33        = PT.Float32Col(pos=15)
    P44        = PT.Float32Col(pos=16)
    P55        = PT.Float32Col(pos=17)
    P66        = PT.Float32Col(pos=18)
    P77        = PT.Float32Col(pos=19)
    P88        = PT.Float32Col(pos=20)

class KalmanEstimatesVelOnly(PT.IsDescription):
    obj_id     = PT.UInt32Col(pos=0)
    frame      = PT.UInt64Col(pos=1)
    timestamp  = PT.Float64Col(pos=2) # time of reconstruction
    x          = PT.Float32Col(pos=3)
    y          = PT.Float32Col(pos=4)
    z          = PT.Float32Col(pos=5)
    xvel       = PT.Float32Col(pos=6)
    yvel       = PT.Float32Col(pos=7)
    zvel       = PT.Float32Col(pos=8)
    # save diagonal of P matrix
    P00        = PT.Float32Col(pos=12)
    P11        = PT.Float32Col(pos=13)
    P22        = PT.Float32Col(pos=14)
    P33        = PT.Float32Col(pos=15)
    P44        = PT.Float32Col(pos=16)
    P55        = PT.Float32Col(pos=17)

class FilteredObservations(PT.IsDescription):
    obj_id     = PT.UInt32Col(pos=0)
    frame      = PT.UInt64Col(pos=1)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    obs_2d_idx = PT.UInt64Col(pos=5) # index into VLArray 'kalman_observations_2d_idxs'

def convert_format(current_data,camn2cam_id,area_threshold=0.0):
    """convert data from format used for Kalman tracker to hypothesis tester"""
    found_data_dict = {}
    first_idx_by_camn = {}
    for camn, stuff_list in current_data.iteritems():
        if not len(stuff_list):
            # no data for this camera, continue
            continue
        for (pt_undistorted,projected_line) in stuff_list:
            if not numpy.isnan(pt_undistorted[0]): # only use if point was found

                # perform area filtering
                area = pt_undistorted[PT_TUPLE_IDX_AREA]
                if area < area_threshold:
                    continue

                cam_id = camn2cam_id[camn]
                found_data_dict[cam_id] = pt_undistorted[:9]
                first_idx_by_camn[camn] = pt_undistorted[PT_TUPLE_IDX_FRAME_PT_IDX]
                break # algorithm only accepts 1 point per camera
    return found_data_dict, first_idx_by_camn
