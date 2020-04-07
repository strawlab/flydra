from __future__ import print_function
import struct
import numpy as nx


class SMDFile:
    def __init__(self, filename):
        self.filename = filename
        self.fd = open(filename, mode="r")
        self.fmt = "<dII"
        self.row_sz = struct.calcsize(self.fmt)
        ts = []
        print("loading SMD file", filename)
        smd = self.fd.read()
        print(" read buffer, parsing...")
        len_smd = len(smd)
        idx = 0
        last_ts = None
        while 1:
            stop_idx = idx + self.row_sz
            if stop_idx > len_smd:
                break
            cmp_ts, left, bottom = struct.unpack(self.fmt, smd[idx:stop_idx])
            ts.append(cmp_ts)
            idx = stop_idx
        print(" done parsing buffer")
        self.timestamps = nx.array(ts)

        check_monotonic = True
        if check_monotonic:
            tdiff = self.timestamps[1:] - self.timestamps[:-1]
            if nx.amin(tdiff) < 0:
                raise ValueError("timestamps are not monotonic")

    def get_all_timestamps(self):
        return self.timestamps

    def get_left_bottom(self, timestamp):
        idx = self.timestamps.searchsorted([timestamp])[0]
        self.fd.seek(idx * self.row_sz)
        row_buf = self.fd.read(self.row_sz)
        cmp_ts, left, bottom = struct.unpack(self.fmt, row_buf)
        assert cmp_ts == timestamp
        return left, bottom

    def close(self):
        self.fd.close()
