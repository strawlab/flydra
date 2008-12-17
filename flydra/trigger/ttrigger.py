from trigger import Device
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import pickle

# TODO: use trait handler to validate FPS before setting trigger
# device. This will allow GUI views to go red when invalid value is
# entered. Probably this will require subclassing TraitsHandler.
#
# The above results in a known bug: the GUI can get out-of-sync with
# the actual device.

def fps_fget(obj,name,value):
    return obj.real_fps

def fps_fset(obj,name,value):
    float_value = float(value)

    obj.ensure_device()

    try:
        obj.private_device.set_carrier_frequency(float_value)
    finally:
        obj.real_fps = obj.private_device.get_carrier_frequency()

class TriggerDevice(traits.HasTraits):
    frames_per_second = traits.Property(fget=fps_fget,
                                        fset=fps_fset,
                                        )

    # private - don't modify
    real_fps = traits.Float(transient=True)
    private_device = traits.Instance(Device, transient=True)

    traits_view = View( Group( ( Item('frames_per_second',
                                      #style='custom',
                                      ),
                                 ),
                               orientation = 'horizontal',
                               show_border = False,
                               ),
                        title = 'Trigger device',
                        )

    def __init__(self,*args,**kwargs):
        super(TriggerDevice,self).__init__(*args,**kwargs)
        self.ensure_device()

    def ensure_device(self):
        if self.private_device is None:
            self.private_device = Device()
            self.private_device.set_carrier_frequency(10.0) # safe value(?)
            self.real_fps = self.private_device.get_carrier_frequency()

    def get_framecount_stamp(self):
        return self.private_device.get_framecount_stamp()

    def get_timer_max(self):
        return self.private_device.get_timer_max()

    def reset_framecount_A(self):
        return self.private_device.reset_framecount_A()

# This will only work if the device is attached.

## def test_pickle():
##     td=TriggerDevice()
##     buf = pickle.dumps(td)
##     del td
##     td2 = pickle.loads(buf)
