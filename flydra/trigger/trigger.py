import pylibusb as usb
import ctypes
import sys, time,os
from optparse import OptionParser

REQUIRE_FLYDRA_TRIGGER=int(os.environ.get('REQUIRE_FLYDRA_TRIGGER',1))

__all__ = ['Device',
           'enter_dfu_mode',
           'check_device',
           'set_frequency',
           'trigger_once',
           'get_trigger_device',
           ]

CS_dict = { 0:0, # off
            1:1,
            8:2,
            64:3,
            256:4,
            1024:5}

# keep in sync with defines in trigger_task.c
TASK_FLAGS_ENTER_DFU = 0x01
TASK_FLAGS_NEW_TIMER3_DATA = 0x02
TASK_FLAGS_DO_TRIG_ONCE = 0x04
TASK_FLAGS_DOUT_HIGH = 0x08
TASK_FLAGS_GET_DATA = 0x10
TASK_FLAGS_RESET_FRAMECOUNT_A = 0x20
TASK_FLAGS_SET_EXT_TRIG1 = 0x40

def debug(*args):
    if 0:
        print >> sys.stderr, ' '.join([str(arg) for arg in args])

def get_trigger_device():
    try:
        device = Device()
    except RuntimeError,err:
        if REQUIRE_FLYDRA_TRIGGER:
            raise
        else:
            device = FakeDevice()
    return device

class FakeDevice:
    def set_carrier_frequency(self,*args,**kw):
        return
    def get_carrier_frequency(self,*args,**kw):
        return 123.4
    def get_timer_max(self,*args,**kw):
        return 0xFFFF
    def get_framecount_stamp(self,*args,**kw):
        return 0, 0x0001

class Device:
    def __init__(self,ignore_version_mismatch=False):
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
            raise RuntimeError("Cannot find device. (Perhaps run with environment variable REQUIRE_FLYDRA_TRIGGER=0.)")

        debug('found device',dev)
        self.libusb_handle = usb.open(dev)

        manufacturer = usb.get_string_simple(self.libusb_handle,dev.descriptor.iManufacturer)
        product = usb.get_string_simple(self.libusb_handle,dev.descriptor.iProduct)
        serial = usb.get_string_simple(self.libusb_handle,dev.descriptor.iSerialNumber)

        assert manufacturer == 'Strawman'
        valid_product = 'Flydra Trigger Control 1.0'
        if product != valid_product:
            errmsg = 'Expected product "%s", but you have "%s"'%(
                valid_product,product)
            if ignore_version_mismatch:
                print 'WARNING:',errmsg
            else:
                raise ValueError(errmsg)

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
        #print 'config.interface',config.interface
##        for i in range(config.bNumInterfaces):
##            iface = config.interface[i]
##            print iface
##            print iface.altsetting
            
        usb.claim_interface(self.libusb_handle, interface_nr)

        self.OUTPUT_BUFFER = ctypes.create_string_buffer(16)

        self.FOSC = 8000000 # 8 MHz
        trigger_carrier_freq = 0.0 # stopped

        self.timer3_CS = 8
        #print 'set A self.timer3_CS to',self.timer3_CS        
        self._set_timer3_metadata(trigger_carrier_freq)
        
    def set_carrier_frequency( self, freq=None ):
        if freq is None:
            #print 'setting frequency to default (200 Hz)'
            freq = 200.0
        #print 'setting freq to',freq
        if freq != 0:
            if self.timer3_CS == 0:
                success = False
                timer_vals = CS_dict.keys()
                timer_vals.sort()
                for timer3_CS in timer_vals:
                    self.timer3_CS = timer3_CS
                    try:
                        self._set_timer3_metadata(freq)
                    except ValueError, err:
                        continue # try again
                    else:
                        success = True
                        break
                if not success:
                    raise RuntimeError('cound not set timer3 metadata')
            else:
                self._set_timer3_metadata(freq)
        else:
            self._set_timer3_metadata(freq)
            
    def get_carrier_frequency( self ):
        IFI = self.timer3_TOP * self.timer3_clock_tick_duration
        return 1.0/IFI
        
    def get_timer_max( self ):
        return self.timer3_TOP
        
    def _set_timer3_metadata(self, carrier_freq):
        if carrier_freq <= 0:
            self.timer3_CS = 0
        else:
            if self.timer3_CS == 0:
                raise ValueError('cannot set non-zero freq because clock select is zero')
            
        if self.timer3_CS == 0:
            buf = self.OUTPUT_BUFFER # shorthand
            buf[9] = chr(CS_dict[self.timer3_CS])
            if carrier_freq >= 0:
                buf[8] = chr(TASK_FLAGS_NEW_TIMER3_DATA)
            else:
                # if negative, raise value high
                buf[8] = chr(TASK_FLAGS_NEW_TIMER3_DATA|TASK_FLAGS_DOUT_HIGH)
                
            self.send_buf()
            return
            
        F_CLK = self.FOSC/float(self.timer3_CS) # clock frequency, Hz

        #print 'F_CPU',self.FOSC
        #print 'F_CLK',F_CLK

        clock_tick_duration = 1.0/F_CLK
        carrier_duration = 1.0/carrier_freq
        n_ticks_for_carrier = int(round(carrier_duration/clock_tick_duration))
        if n_ticks_for_carrier > 0xFFFF:
            raise ValueError('n_ticks_for_carrier too large for 16 bit counter, try increasing self.timer3_CS')
        
##        print 'clock_tick_duration',clock_tick_duration
##        print 'carrier_freq',carrier_freq
##        print 'carrier_duration',carrier_duration
##        print 'n_ticks_for_carrier',n_ticks_for_carrier

        #actual_freq = 1.0/(n_ticks_for_carrier*clock_tick_duration)
        #print 'actual_freq',actual_freq

        
        self.timer3_TOP = n_ticks_for_carrier
        self.timer3_clock_tick_duration = clock_tick_duration
        self.set_output_durations(A_sec=1e-3, # 1 msec
                                  B_sec=0,
                                  C_sec=0,
                                  ) # output compare A duration 10 usec

    def trigger_once(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_DO_TRIG_ONCE)
        self.send_buf()

    def ext_trig1(self):
        print 'triggering!'
        TASK_FLAGS_SET_EXT_TRIG1
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_SET_EXT_TRIG1)
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

    def send_buf(self,return_input=False):
        buf = self.OUTPUT_BUFFER # shorthand
        #buf[9] = chr(1)
        #print 'ord(buf[8])',ord(buf[8])
        #print 'ord(buf[9])',ord(buf[9])

        if 1:
            val = usb.bulk_write(self.libusb_handle, 0x06, buf, 9999)
            debug('set_output_durations result: %d'%(val,))

        if 1:
            INPUT_BUFFER = ctypes.create_string_buffer(16)
            
            try:
                val = usb.bulk_read(self.libusb_handle, 0x82, INPUT_BUFFER, 1000)
                if 0:
                    print 'read',val
            except usb.USBNoDataAvailableError:
                if 0:
                    sys.stdout.write('?')
                    sys.stdout.flush()
                val = None
            if return_input:
                return INPUT_BUFFER

    def enter_dfu_mode(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_ENTER_DFU)
        val = usb.bulk_write(self.libusb_handle, 0x06, buf, 9999)

    def _read_data_from_device(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_GET_DATA)
        returned_data = self.send_buf(return_input=True)
        #for i in range(16):
        #    print '%02d %d'%(i,ord(returned_data[i]))
        return returned_data

    def get_framecount_stamp(self):
        data = self._read_data_from_device()
        framecount = 0
        for i in range(8):
            framecount += ord(data[i]) << (i*8)
        tcnt3 = ord(data[8]) + (ord(data[9]) << 8)
        return framecount, tcnt3

    def reset_framecount_A(self):
        buf = self.OUTPUT_BUFFER # shorthand
        buf[8] = chr(TASK_FLAGS_RESET_FRAMECOUNT_A)
        self.send_buf()

def enter_dfu_mode():
    import sys
    from optparse import OptionParser
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--ignore-version-mismatch", action='store_true',
                      dest='ignore_version_mismatch',
                      default=False)
    (options, args) = parser.parse_args()
    dev = Device(ignore_version_mismatch=options.ignore_version_mismatch)
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
    
    dev.set_carrier_frequency( 0.0 )
    dev.reset_framecount_A()
    dev.set_carrier_frequency( options.freq )
    t_start = time.time()
    n_secs = 5.0
    t_stop = t_start+n_secs
    while time.time() < t_stop:
        # busy wait for accurate timing
        pass
    framecount, tcnt3 = dev.get_framecount_stamp()
    fps = framecount/n_secs
    #print 'framecount, tct3,fps',framecount, tcnt3,fps
    theory = dev.get_carrier_frequency()
    measured = fps
    print 'theoretical fps',theory
    print 'measured fps',measured

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

if __name__=='__main__':
    device = Device()
    
