import camera_feature_point_proto
import numpy as np

def to_proto_time(double_ts):
    sec = int(np.floor(double_ts))
    nsec = int(np.floor((double_ts - sec) * 1e9))
    return camera_feature_point_proto.Timestamp( sec=sec,
                                                 nsec=nsec )

def from_protobuf_time(msg):
    return msg.sec + float(msg.nsec)*1e-9
