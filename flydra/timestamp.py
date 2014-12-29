import numpy as np

def to_pb2_time(parent,double_ts):
    sec = int(np.floor(double_ts))
    nsec = int(np.floor((double_ts - sec) * 1e9))
    parent.sec = sec
    parent.nsec = nsec

def from_protobuf_time(msg):
    return msg.sec + float(msg.nsec)*1e-9
