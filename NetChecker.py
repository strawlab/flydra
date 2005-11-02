#!/usr/bin/env python
from __future__ import division

import math, socket, struct, select, sys, time

__all__=['NoNewDataError','NetChecker']

class NoNewDataError(Exception):
    pass

class NetChecker:
    def __init__(self):
        self.data = ''
        self.x = None
        self.y = None
        self.z = None
        self.corrected_framenumber = None

        hostname = ''
        self.sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sockobj.setblocking(0)

        port = 28931

        self.sockobj.bind(( hostname, port))
        
    def get_last_xyz_fno(self):
        if self.x is None:
            raise NoNewDataError()
        result = (self.x, self.y, self.z), self.corrected_framenumber
        self.x = None
        self.y = None
        self.z = None
        self.corrected_framenumber = None
        return result
    
    def check_network(self):
        fmt = '<iBfffffffffdf' # little endian
        fmt_size = struct.calcsize(fmt)

        # return if data not ready
        while 1:
            in_ready, trash1, trash2 = select.select( [self.sockobj], [], [], 0.0 )
            if not len(in_ready):
                break

            newdata, addr = self.sockobj.recvfrom(4096)

            mytime = time.time()
            self.data = self.data + newdata
            while len(self.data) >= fmt_size:
                tmp = struct.unpack(fmt,self.data[:fmt_size])
                self.data = self.data[fmt_size:]
                repr_error = tmp[-1]
                if repr_error < 10.0:
                    self.corrected_framenumber,line3d_valid,self.x,self.y,self.z = tmp[:5]
                    find3d_time = tmp[-2]

if __name__=='__main__':
    nc = NetChecker()
    quit_now = False
    while not quit_now:

        nc.check_network()
        try:
            xyz, cf = nc.get_last_xyz_fno()
            print xyz, cf
        except NoNewDataError:
            pass

        time.sleep(0.1)
