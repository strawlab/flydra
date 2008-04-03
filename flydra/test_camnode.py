#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import camnode, camnode_utils
import sys, time

# get default options
class OptionsDict:
    def __init__(self,d):
        self.__dict__.update(d)
    def __getattr__(self,attr):
        print 'OptionsDict returning None for attr "%s"'%attr
        return None

class GatherResults(object):
    def __init__(self,
                 cam_id=None,
                 quit_event=None,
                 ):
        self._chain = camnode_utils.ChainLink()
        self._cam_id = cam_id
        self._quit_event = quit_event
        self._results = []

    def get_chain(self):
        return self._chain

    def mainloop(self):
        while not self._quit_event.isSet():
            with camnode_utils.use_buffer_from_chain(self._chain) as buf:
                sys.stdout.write('.')
                sys.stdout.flush()
                # post images and processed points to wx
                if hasattr(buf,'processed_points'):
                    pts = buf.processed_points
                    fno = buf.framenumber
                    self._results.append( (fno, pts) )
                else:
                    pts = None

def analyze_file(filename):
    defaults = camnode.get_app_defaults()
    options = OptionsDict(defaults)

    options.emulation_image_sources = filename

    app_state = camnode.AppState(options = options,
                                 use_dummy_mainbrain = True,
                                 )
    app_state.set_quit_function( sys.exit )
    app_state.append_chain( klass = GatherResults,
                            basename = 'GatherResults' )
    results = {}
    while 1:
        # This first step is probably not necessary - this checks for
        # pyro calls, quits if cameras done, etc
        app_state.main_thread_task()

        any_not_finished = False

        for (m,c) in zip(app_state.get_image_sources(),
                         app_state.get_image_controllers()):
            # We do all image sources and controllers in case we want to
            # process N camera views simultaneously.

            if c.is_finished():
                #print 'c is finished'
                continue
            else:
                any_not_finished = True


            c.trigger_single_frame_start()

            # wait until processing chain is done...
            pool = m.get_buffer_pool()
            while pool.get_num_outstanding_buffers() > 0:
                time.sleep(0.001) # polling is easier to implement than blocking. HACK.

        if not any_not_finished:
            break

    for thread in app_state.critical_threads:
        print 'waiting for thread to end'
        print thread.getName()
        thread.join()
    return results

def main():
    filename = '/media/disk/mamarama/20080402/full_20080402_163045_mama04_0.fmf'
    results = analyze_file(filename)

if __name__=='__main__':
    main()
