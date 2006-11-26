import pylibusb as usb
import ctypes
import sys

__all__ = ['Device','enter_dfu_mode']

CS_dict = { 0:0, # off
            1:1,
            8:2,
            64:3,
            256:4,
            1024:5}

FLAG_ENTER_DFU = 0x01

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
        debug('dev.config[0].bConfigurationValue',dev.config[0].bConfigurationValue)
        usb.set_configuration(self.libusb_handle, dev.config[0].bConfigurationValue)
        debug('claiming interface')
        usb.claim_interface(self.libusb_handle, interface_nr)

        self.OUTPUT_BUFFER = ctypes.create_string_buffer(16)
        if 0:
            for i in range(len(self.OUTPUT_BUFFER)):
                self.OUTPUT_BUFFER[i] = chr(0x88)
        
def enter_dfu_mode():
    print 'hi'
    dev = Device()
    print 'got device',dev