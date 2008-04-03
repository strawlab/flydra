#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import camnode, camnode_utils
import sys, time
import numpy
import pylab

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
                 total_frames = None,
                 verbose=False,
                 ):
        self._chain = camnode_utils.ChainLink()
        self._cam_id = cam_id
        self._results_fnos = []
        self._results_data = []
        self._done = False
        self._total_frames = total_frames
        self._verbose = verbose

    def get_chain(self):
        return self._chain

    def mainloop(self):
        fcount = 0
        while 1:
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf:
                if chainbuf.quit_now:
                    break

                if self._verbose:
                    if fcount == 0:
                        tstart = time.time()
                    else:
                        if fcount % 100==0:
                            now = time.time()
                            dur = now-tstart
                            fps = fcount/dur
                            print '%s: frame %d of %d (%.1f fps)'%(
                                self._cam_id, fcount, self._total_frames,fps)

                fcount += 1
                if 0 and self._verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                # post images and processed points to wx
                if hasattr(chainbuf,'processed_points'):
                    pts = chainbuf.processed_points
                    fno = chainbuf.framenumber
                    for pt in pts:
                        self._results_data.append( pt )
                        self._results_fnos.append( fno )
        self._done = True
    def get_results(self):
        if not self._done:
            raise RuntimeError('asking for results when not done!')
        return numpy.array(self._results_fnos), numpy.array(self._results_data)

def analyze_file(filenames):
    defaults = camnode.get_app_defaults()
    options = OptionsDict(defaults)

    options.emulation_image_sources = filenames

    app_state = camnode.AppState(options = options,
                                 use_dummy_mainbrain = True,
                                 )
    app_state.set_quit_function( sys.exit )

    ftotal = [c.get_n_frames() for c in app_state.get_image_controllers()]
    kwargs_per_instance = [ {'total_frames':ftot} for ftot in ftotal ]
    if 0:
        kwargs = None
    else:
        kwargs = dict(verbose=True)
    result_gatherers = app_state.append_chain( klass = GatherResults,
                                               kwargs_per_instance=kwargs_per_instance,
                                               kwargs = kwargs,
                                               basename = 'GatherResults' )


    while 1:
        # This first step is probably not necessary - this checks for
        # pyro calls, quits if cameras done, etc
        app_state.main_thread_task()

        any_not_finished = False # any camera still going sets this true
        for camno,(m,c) in enumerate(zip(app_state.get_image_sources(),
                                         app_state.get_image_controllers())):
            # We do all image sources and controllers in case we want to
            # process N camera views simultaneously.
            if c.is_finished():
                c.quit_now()
                continue # do next camera
            else:
                any_not_finished = True


            # On last frame, this will result in c.is_finished() returning true.
            c.trigger_single_frame_start()

            # Hack to wait until processing chain is done before
            # starting next frame.
            pool = m.get_buffer_pool()
            if 1:
                while 1:
                    pool.wait_for_0_outstanding_buffers(0.01) # timeout
                    if c.is_finished():
                        c.quit_now()
                        break # do next camera
                    if pool.get_num_outstanding_buffers() == 0:
                        break # done processing
            else:
                while pool.get_num_outstanding_buffers() > 0:
                    if c.is_finished():
                        c.quit_now()
                        break # do next camera
                    time.sleep(0.001) # polling is easier to implement than blocking. HACK.

        if not any_not_finished:
            break

    for thread in app_state.critical_threads:
        thread.join()

    results = {}
    for cam_id, gather_result_instance in result_gatherers.iteritems():
        fnos, data = gather_result_instance.get_results()
        results[cam_id] = (fnos, data)
    return results

def main():
    filenames = '/media/disk/mamarama/20080402/full_20080402_163045_mama04_0.fmf'
    results = analyze_file(filenames)#, use_wx=True)
    for cam_id,(fnos, data) in results.iteritems():
        pylab.figure()
        pylab.plot( fnos, data[:,0], 'r.')
        pylab.plot( fnos, data[:,1], 'g.')
        pylab.title(cam_id)
    pylab.show()

if __name__=='__main__':
    main()
