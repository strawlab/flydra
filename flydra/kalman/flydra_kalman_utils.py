import tables as PT

class KalmanEstimates(PT.IsDescription):
    obj_id     = PT.Int32Col(pos=0,indexed=True)
    frame      = PT.Int32Col(pos=1)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    xvel       = PT.Float32Col(pos=5)
    yvel       = PT.Float32Col(pos=6)
    zvel       = PT.Float32Col(pos=7)
    xaccel     = PT.Float32Col(pos=8)
    yaccel     = PT.Float32Col(pos=9)
    zaccel     = PT.Float32Col(pos=10)

class FilteredObservations(PT.IsDescription):
    obj_id     = PT.Int32Col(pos=0,indexed=True)
    frame      = PT.Int32Col(pos=1,indexed=True)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    

def convert_format(current_data):
    found_data_dict = {}
    for cam_id, stuff_list in current_data.iteritems():
        if not len(stuff_list):
            # no data for this camera, continue
            continue
        this_point,projected_line = stuff_list[0] # algorithm only accepts 1 point per camera
        if this_point[9]: # only use if found_anything
            found_data_dict[cam_id] = this_point[:9]
    return found_data_dict
