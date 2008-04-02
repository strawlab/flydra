import camnode
import sys, time

# get default options
class OptionsDict:
    def __init__(self,d):
        self.__dict__.update(d)
    def __getattr__(self,attr):
        print 'OptionsDict returning None for attr "%s"'%attr
        return None

if 1:
    defaults = camnode.get_app_defaults()
    options = OptionsDict(defaults)

    options.emulation_image_sources = '/media/disk/mamarama/20080331/full_20080331_144110_mama01_0.fmf'

    app_state = camnode.AppState(options = options,
                                 use_dummy_mainbrain = True,
                                 )
    app_state.set_quit_function( sys.exit )

    while 1:

        # This first step is probably not necessary - this checks for
        # pyro calls, quits if cameras done, etc
        app_state.main_thread_task()

        for (m,c) in zip(app_state.image_sources,
                         app_state.image_controllers):

            # Now, get image.
            c.trigger_single_frame_start()

            # wait until processing chain is done...
            pool = m.get_buffer_pool()
            while pool.get_num_outstanding_buffers() > 0:
                time.sleep(0.001) # polling is easier to implement than blocking. HACK.
