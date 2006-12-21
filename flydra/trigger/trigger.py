import pylibusb as usb
import ctypes
import sys, time
from optparse import OptionParser

__all__ = ['Device',
           'enter_dfu_mode',
           'check_device',
           'set_frequency',
           'trigger_once',
           ]

CS_dict = { 0:0, # off
            1:1,
            8:2,
            64:3,
            256:4,
            1024:5}

TASK_FLAGS_ENTER_DFU = 0x01
TASK_FLAGS_NEW_TIMER3_DATA = 0x02
TASK_FLAGS_DO_TRIG_ONCE = 0x04

def debug(*args):
    if 1:
        print >> sys.stderr, ' '.join([str(arg) for arg in args])

class Device:
    def __init__(self):
        usb.init()
        
        if not usb.get_busses():
            usb.find_busses()
            usb.find_devices()

        busses = usb.get_busses()

        found = False
        for bus in busses:
            for dev in bus.devices:
                debug('idVendor: 0x%04x idProduct: 0x%04x'%(dev.descriptor.idVendor,
                                                            dev.descriptor.idProduct))
                if (dev.descriptor.idVendor == 0x1781 and
                    dev.descriptor.idProduct == 0x0BAF):
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError("Cannot find device.")

        debug('found device',dev)
        self.libusb_handle = usb.open(dev)

        interface_nr = 0
        if hasattr(usb,'get_driver_np'):
            # non-portable libusb extension
            name = usb.get_driver_np(self.libusb_handle,interface_nr)
            if name != '':
                debug("attached to kernel driver '%s', detaching."%name )
                usb.detach_kernel_driver_np(self.libusb_handle,interface_nr)

        if dev.descriptor.bNumConfigurations > 1:
            debug("WARNING: more than one configuration, choosing first")

        debug('setting configuration')
        debug('dev.config[0]',dev.config[0])
        config = dev.config[0]
        debug('config.bConfigurationValue',config.bConfigurationValue)
        usb.set_configuration(self.libusb_handle, config.bConfigurationValue)
        debug('claiming interface')
        debug('config.bNumInterfaces',config.bNumInterfaces)
        print 'config.interface',config.interface
        for i in range(config.bNumInterfaces):
            iface = config.interface[i]
            print iface
            print iface.altsetting
            
        usb.claim_interface(self.libusb_handle, interface_nr)

        self.OUTPUT_BUFFER = ctypes.create_string_buffer(16)

        self.FOSC = 1000000 # 1 MHz # hey, i thought it was at 8 ?!?
        trigger_carrier_freq = 0.0 # stopped

        self.timer3_CS = 1
        self._set_timer3_metadata(trigger_carrier_freq)
        
    def set_carrier_frequency( self, freq=None ):
        if freq is None:
            print 'setting frequency to default (200 Hz)'
            freq = 200.0
        print 'setting freq to',freq
        if freq != 0:
            if self.timer3_CS == 0:
                self.timer3_CS = 1
        self._set_timer3_metadata(freq)
        
    def _set_timer3_metadata(self, carrier_freq):
        if carrier_freq == 0:
            self.timer3_CS = 0
        else:
            if self.timer3_CS == 0:
                raise ValueError('cannot set non-zero freq because clock select is zero')
            
        if self.timer3_CS == 0:
            buf = self.OUTPUT_BUFFER # shorthand
            buf[8] = chr(TASK_FLAGS_NEW_TIMER3_DATA)
            buf[9] = chr(CS_dict[self.timer3_CS])
            self.send_buf()
            return
            
        F_CLK = self.FOSC/float(self.timer3_CS) # clock frequency, Hz

        clock_tick_duration = 1.0/F_CLK
        carrier_duration = 1.0/carrier_freq
        n_ticks_for_carrier = int(round(carrier_duration/clock_tick_duration))
        if n_ticks_for_carrier > 0xFFFF:
            raise ValueError('n_ticks_for_carrier too large for 16 bit counter, try increasing self.timer3_CS')
        print 'F_CPU',self.FOSC
        print 'F_CLK',F_CLK
        print 'clock_tick_duration',clock_tick_duration
        print 'carrier_freq',carrier_freq
        print 'carrier_duration',carrier_duration
        print 'n_ticks_for_carrier',n_ticks_for_carrier
        
        self.timer3_TOP = n_ticks_for_carrier
        print 'self.timer3_TOP',self.timer3_TOP
        self.timer3_clock_tick_duration = clock_tick_duration
        self.set_output_durations(A_sec=100e-6, # 100 usec
                                  B_sec=0,
                                  C_sec=0,
                                  ) # output compare A duration 10 usec

    def trigger_once(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_DO_TRIG_ONCE)
        self.send_buf()
        
    def set_output_durations(self, A_sec=None, B_sec=None, C_sec=None):
        dur_A = A_sec
        dur_B = B_sec
        dur_C = C_sec
        
        ##########
        
        ticks_pwm_A = int(round(dur_A/self.timer3_clock_tick_duration))
        if ticks_pwm_A > self.timer3_TOP:
            raise ValueError('ticks_pwm_A larger than timer3_TOP')
        ticks_pwm_B = int(round(dur_B/self.timer3_clock_tick_duration))
        if ticks_pwm_B > self.timer3_TOP:
            raise ValueError('ticks_pwm_B larger than timer3_TOP')
        ticks_pwm_C = int(round(dur_C/self.timer3_clock_tick_duration))
        if ticks_pwm_C > self.timer3_TOP:
            raise ValueError('ticks_pwm_C larger than timer3_TOP')
        ##########
        
        ocr3a = ticks_pwm_A
        ocr3b = ticks_pwm_B
        ocr3c = ticks_pwm_C
        top = self.timer3_TOP
        
        buf = self.OUTPUT_BUFFER # shorthand
        buf[0] = chr(ocr3a//0x100)
        buf[1] = chr(ocr3a%0x100)
        buf[2] = chr(ocr3b//0x100)
        buf[3] = chr(ocr3b%0x100)

        buf[4] = chr(ocr3c//0x100)
        buf[5] = chr(ocr3c%0x100)
        buf[6] = chr(top//0x100)
        buf[7] = chr(top%0x100)
        
        buf[8] = chr(TASK_FLAGS_NEW_TIMER3_DATA)
        buf[9] = chr(CS_dict[self.timer3_CS])

        self.send_buf()

    def send_buf(self):
        buf = self.OUTPUT_BUFFER # shorthand
        #buf[9] = chr(1)
        print 'ord(buf[8])',ord(buf[8])
        print 'ord(buf[9])',ord(buf[9])

        if 1:
            val = usb.bulk_write(self.libusb_handle, 0x06, buf, 9999)
            debug('set_output_durations result: %d'%(val,))

        if 1:
            INPUT_BUFFER = ctypes.create_string_buffer(16)
            
            try:
                val = usb.bulk_read(self.libusb_handle, 0x82, INPUT_BUFFER, 1000)
                if 1:
                    print 'read',val
            except usb.USBNoDataAvailableError:
                if 0:
                    sys.stdout.write('?')
                    sys.stdout.flush()

    def enter_dfu_mode(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_ENTER_DFU)
        val = usb.bulk_write(self.libusb_handle, 0x06, buf, 9999)
        
def enter_dfu_mode():
    dev = Device()
    dev.enter_dfu_mode()
    
def check_device():
    dev = Device()

def set_frequency():
    usage = '%prog FILE [options]'
    parser = OptionParser(usage)
    
    parser.add_option("--freq", type="float",
                      metavar="FREQ")
    (options, args) = parser.parse_args()
    dev = Device()
    dev.set_carrier_frequency( options.freq )

def get_time():
    if sys.platform.startswith('win'):
        return time.clock()
    else:
        return time.time()

def trigger_once():
    dev = Device()
    dev.set_carrier_frequency( 0.0 )
    time.sleep(0.5)
    start = get_time()
    dev.trigger_once()
    stop = get_time()
    dur = stop-start
    print 'max trigger roundtrip latency %.2f ms'%(dur*1000.0,)
