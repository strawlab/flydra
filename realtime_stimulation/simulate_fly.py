#!/usr/bin/env python
from __future__ import division
from numarray.ieeespecial import nan
import socket, struct, time

sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

hostname = ''
port = 28931


fmt = 'ifffffffffd'
fmt_size = struct.calcsize(fmt)
trig = True
fno = 0
while 1:
    fno += 1
    if trig:
        xyz = 1250.0,152.5,150.0
    else:
        xyz = 0,0,0
    trig = not trig
    line3d = nan, nan, nan, nan, nan, nan
    find3d_time = time.time()
    data = (fno,) + xyz + line3d + (find3d_time,)

    print 'press return to send data',data[:4]
    packet = struct.pack(fmt,*data)
    raw_input()
    sockobj.sendto( packet, (hostname,port))
