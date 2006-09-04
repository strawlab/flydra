#!/usr/bin/env python
from __future__ import division

import serial
import struct

class TriggerDevice:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0',
                                 baudrate=57600,
                                 bytesize=8,
                                 parity='N',
                                 stopbits=1,
                                 #timeout=0, # don't wait
                                 timeout=None, # wait forever
                                 xonxoff=0,
                                 rtscts=0)
        self.mcu_freq = self._get_freq()
        self.mcu_tick = 1/self.mcu_freq

    def _get_freq(self):
        self.ser.write('f')
        header = self.ser.read(2)
        assert header[0] == 'f'
        nbytes = ord(header[1])
        assert nbytes == 4

        buf = self.ser.read(nbytes)
        freq, = struct.unpack('>i',buf)
        return freq

trigger_freq = 100.0 # Hz
device = TriggerDevice()
print device.mcu_freq

trigger_dt = 1/trigger_freq
#cpu_dt = device.mcu_tick
cpu_dt = 1/32768

print 'cpu_dt',cpu_dt

prescalar = 1

print 'prescalar',prescalar
clocks_per_trig = (trigger_dt/cpu_dt)/prescalar
print 'clocks_per_trig',clocks_per_trig
print 'real trigger freq',1.0/(int(round(clocks_per_trig))*cpu_dt*prescalar)
