from trigger import Device
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import pickle

class UnintializedTriggerDevice(Exception): pass

class RemoteFpsHandler(traits.BaseFloat):
    info_text = 'a float'

    def validate ( self, obj, name, value ):
        value = super(RemoteFpsHandler, self).validate(obj, name, value)
        try:
            if obj.private_device is None:
                return value
            obj.private_device.set_carrier_frequency(value)

            actual_value = obj.private_device.get_carrier_frequency()
            return actual_value
        except Exception,err:
            print 'error in validate: %s'%err
        self.error( obj, name, value )


class TriggerDevice(traits.HasTraits):
    frames_per_second = RemoteFpsHandler()

    # private - don't modify
    private_device = traits.Instance(Device, transient=True)

    traits_view = View( Group( ( Group( Item('frames_per_second',
                                             label='frame rate',
                                             ),
                                        Item('frames_per_second',
                                             show_label=False,
                                             style='readonly',
                                             ),
                                        orientation = 'horizontal',
                                        )
                                 ),
                               orientation = 'vertical',
                               ),
                        title = 'Trigger device',
                        )

    def ensure_device(self):
        if self.private_device is None:
            self.private_device = Device()
            if self.frames_per_second is not None:
                self.private_device.set_carrier_frequency(
                    self.frames_per_second)

    def get_framecount_stamp(self):
        self.ensure_device()
        return self.private_device.get_framecount_stamp()

    def get_timer_max(self):
        self.ensure_device()
        value = self.private_device.get_timer_max()
        if value is None:
             raise UnintializedTriggerDevice('')
        return value

    def reset_framecount_A(self):
        self.ensure_device()
        return self.private_device.reset_framecount_A()

# This will only work if the device is attached.

## def test_pickle():
##     td=TriggerDevice()
##     buf = pickle.dumps(td)
##     del td
##     td2 = pickle.loads(buf)

## if __name__=='__main__':
##     test_pickle()
