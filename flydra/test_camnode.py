import camnode
import sys

# get default options
class OptionsDict:
    def __init__(self,d):
        self.__dict__.update(d)
    def __getattr__(self,attr):
        print 'OptionsDict returning None for attr "%s"'%attr
        return None

defaults = camnode.get_app_defaults()
options = OptionsDict(defaults)

options.emulation_image_sources = '/media/disk/mamarama/20080331/full_20080331_144110_mama01_0.fmf'

app_state = camnode.AppState(options = options,
                             use_dummy_mainbrain = True,
                             )
app_state.set_quit_function( sys.exit )
if 1:
    app_state.main_thread_task()
    app_state.main_thread_task()
    app_state.main_thread_task()
