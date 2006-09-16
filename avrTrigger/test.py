#!/usr/bin/env python
from __future__ import division

import serial
import struct

# packet types
PT_STATE = 'r'
PT_FREQ = 'f'
PT_TEMP = 't'
PT_TRIG_TS = 'T'
PT_QUERY_TS = 'Q'

class MalformedPacketError(RuntimeError):
    pass

class TriggerDevice:
    def __init__(self):
        port = '/dev/ttyUSB0'
        kws = dict(
            baudrate=115200,
            bytesize=8,
            parity='N',
            stopbits=1,
            #timeout=0, # don't wait, ever
            timeout=None, # wait forever
            xonxoff=0,
            rtscts=0)

        if 0:
            # clear input
            drain_kws = {}
            drain_kws.update(kws)
            drain_kws['timeout']=0
            
            drain = serial.Serial(port,**drain_kws)
            while 1:
                buf = drain.read(1)
                if len(buf)==0:
                    break
                print 'draining:',repr(buf),hex(ord(buf))
            drain.close()
            
        self.ser = serial.Serial(port,**kws)

        self.mcu_freq = self._get_freq()
        self.mcu_tick = 1/self.mcu_freq

    def _get_packet(self):
        header = self.ser.read(2)
        packet_type = header[0]
        nbytes = ord(header[1])

        buf = self.ser.read(nbytes)

        def hexord(val):
            return hex(ord(val))
        
        print 'packet_type, len(buf), buf',packet_type, len(buf), ' '.join(map(hexord,buf)), repr(buf)
        
        X = self.ser.read(1)
        if X!='X':
            print 'X',hex(ord(X)),repr(X)
            raise MalformedPacketError('')
        return packet_type, buf

    def _get_freq(self):
        self.ser.write('f')
        while 1:
            packet_type, buf = self._get_packet()
            if packet_type==PT_FREQ:
                assert len(buf)==4
                freq, = struct.unpack('>i',buf)
                return freq
    
    def go(self):
        self.ser.write('g')

    def endless_display(self):
        while 1:
            packet_type, buf = self._get_packet()
            if packet_type == PT_TRIG_TS:
                print 'buf',repr(buf)
                timestamp, = struct.unpack('>Q',buf)
                print 'timestamp',hex(timestamp)

trigger_freq = 100.0 # Hz
device = TriggerDevice()
print device.mcu_freq

device.go()
device.endless_display()
