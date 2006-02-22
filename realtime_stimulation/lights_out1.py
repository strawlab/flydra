#!/usr/bin/env python
# $Id: $

# echos back the trigger time for roundtrip latency estimation

from __future__ import division

import numpy as nx
import math, socket, struct, select, sys, time
import serial

lights_off_all = [0.0, 1.0]
lights_off_idx = 0

ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
ser.write('!0S\x00') # lights on

trigger_x = 730
trigger_radius = 25 # mm

outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#################################
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
        return (self.x, self.y, self.z), self.corrected_framenumber
    
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
    ##            approx_latency = (mytime-find3d_time)*1000.0
    ##            print 'approx_latency',(approx_latency-min_latency) # display only varying part of latency
    ##            if approx_latency < min_latency:
    ##                min_latency = approx_latency
    ##            x = (xo - 1250) * 4 + 320
    ##            y = (yo - 150) * 4 + 240


#################################

def query_fly_in_trigger_volume(xyz):
    if xyz[0] is None:
        return False
    dist = abs(xyz[0]-trigger_x) # only consider x component
    if dist <= trigger_radius:
        return True
    else:
        return False

stim_dur_sec = 5.0
pause_dur_sec = 1.0

log_file = open(time.strftime( 'lights_out1_%Y%m%d_%H%M%S.log' ), mode='wb')
#log_file = sys.stdout
log_file.write( '#trigger_x = %s\n'%str(trigger_x))
log_file.write( '#trigger_radius = %s\n'%str(trigger_radius))
log_file.write( '#stim_dur_sec = %s\n'%str(stim_dur_sec) )
log_file.write( '#pause_dur_sec = %s\n'%str(pause_dur_sec) )
log_file.write( '## trig_frame trig_time lights_off\n')

nc = NetChecker()
quit_now = False
status = 'armed'
mode_start_time = time.time()
trigger_corrected_framenumber = None
run_hz = 100.0
cycle_wait = 1/run_hz
while not quit_now:
    time.sleep(cycle_wait)
    ######################

    nc.check_network()
    xyz, cf = nc.get_last_xyz_fno()

    if cf is not None:
        # do something useful
        1+1

    now = time.time()

    if (status == 'armed' and trigger_corrected_framenumber!=cf and
        query_fly_in_trigger_volume(xyz)):
        status = 'triggered'
        lights_off_idx += 1
        lights_off_idx = lights_off_idx % len(lights_off_all)
        lights_off = lights_off_all[lights_off_idx]
        if lights_off:
            ser.write('!0S\x01') # lights off
        else:
            ser.write('!0S\x00') # lights on (control)
        timetime = time.time()
        log_file.write('%d %s %f\n'%(cf,repr(timetime),lights_off))
        log_file.flush()
        mode_start_time = now
        trigger_corrected_framenumber = cf
        outgoing_UDP_socket.sendto(repr(cf),0,('192.168.1.151',28932))
    elif status == 'triggered':
        if (now-mode_start_time) >= stim_dur_sec:
            status = 'waiting'
            mode_start_time = now
            ser.write('!0S\x00')
        else:
            IFI = last_frame-now
    elif status == 'waiting':
        if (now-mode_start_time) >= pause_dur_sec:
            status = 'armed'
    last_frame = now

    ######################
