from __future__ import division
from __future__ import with_statement
import pkg_resources
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, time

import numpy
import tables as PT
from optparse import OptionParser
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.xml_stimulus_osg as xml_stimulus_osg
import flydra.a2.core_analysis as core_analysis
import flydra.a2.analysis_options as analysis_options
import flydra.analysis.result_utils as result_utils
import flydra.a2.flypos
import fsee
import fsee.Observer
import fsee.plot_utils
import pylab

class PathMaker:
    def __init__(self,P=None,Q=None,data_dt=None,sample_dt=None,frame=None):
        self.P = P
        self.Q = Q
        self.data_dt = data_dt
        if sample_dt != data_dt:
            raise NotImplementedError('no interpolation implemented')
        self.cur_idx = 0
        self.frame = frame

    def get_frame(self):
        """get frame number of just-returned position and orientation"""
        if self.cur_idx <= 0:
            raise RuntimeError('only query get_frame() after calling step()')
        return self.frame[self.cur_idx-1]

    def step(self):
        cur_pos = self.P[self.cur_idx]
        cur_ori = self.Q[self.cur_idx]
        self.cur_idx += 1
        return cur_pos, cur_ori

def doit(options=None):
    assert options is not None
    assert options.stim_xml is not None

    ca = core_analysis.get_global_CachingAnalyzer()

    (obj_ids, use_obj_ids, is_mat_file, data_file, extra
     ) = ca.initial_file_load(options.kalman_filename)

    fps = result_utils.get_fps( data_file )

    if 1:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    if 1:
        data_file_path, data_file_base = os.path.split(data_file.filename)
        file_timestamp = data_file_base[4:19]
        stim_xml = xml_stimulus.xml_stimulus_from_filename(
            options.stim_xml,
            timestamp_string=file_timestamp,
            )
        try:
            fanout = xml_stimulus.xml_fanout_from_filename( options.stim_xml )
        except xml_stimulus.WrongXMLTypeError:
            walking_start_stops = []
            include_obj_ids = exclude_obj_ids = None
        else:
            include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp(
                timestamp_string=file_timestamp )
            walking_start_stops = fanout.get_walking_start_stops_for_timestamp(
                timestamp_string=file_timestamp )
            if include_obj_ids is not None:
                use_obj_ids = include_obj_ids
            if exclude_obj_ids is not None:
                use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )
            stim_xml = fanout.get_stimulus_for_timestamp(timestamp_string=file_timestamp)

        if include_obj_ids is not None:
            use_obj_ids = include_obj_ids
        if exclude_obj_ids is not None:
            use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )
        stim_xml_osg = xml_stimulus_osg.StimulusWithOSG( stim_xml.get_root() )

    if len(options.obj_only) != 0:
        # XXX Should maybe check that obj_only is not excluded and is included
        use_obj_ids = options.obj_only

    kalman_rows = flydra.a2.flypos.fuse_obj_ids(
        use_obj_ids, data_file,
        dynamic_model_name = dynamic_model,
        frames_per_second=fps)
    frame = kalman_rows['frame']
    if (options.start is not None) or (options.stop is not None):
        valid_cond = numpy.ones( frame.shape, dtype=numpy.bool )
        if options.start is not None:
            valid_cond &= (frame >= options.start)
        if options.stop is not None:
            valid_cond &= (frame <= options.stop)

        kalman_rows = kalman_rows[valid_cond]
    frame = kalman_rows['frame']
    X = numpy.array( [kalman_rows['x'],
                      kalman_rows['y'],
                      kalman_rows['z']]).T

    Q = flydra.a2.flypos.pos2ori(X,force_pitch_0=True)

    hz = fps
    dt = 1.0/hz
    path_maker = PathMaker(P=X,Q=Q,data_dt=dt,sample_dt=dt,frame=frame)


    with stim_xml_osg.OSG_model_path() as osg_model_path:
        vision = fsee.Observer.Observer(model_path=osg_model_path,
                                        #scale=1000.0, # from meters to mm
                                        hz=hz,
                                        skybox_basename=None,
                                        full_spectrum=True,
                                        optics='buchner71',
                                        do_luminance_adaptation=False,
                                        )
        count = 0
        tstart = time.time()
        while count < len(X):
            tnow = time.time()
            if count > 0 and (count%100)==0:
                dur = tnow-tstart
                print '%.1f fps (%d frames in %.1f seconds)'%(
                    count/dur,count,dur)
            count += 1
            cur_pos, cur_ori = path_maker.step()
            frame = path_maker.get_frame()
            vision.step(cur_pos,cur_ori)

            if options.save_envmap:
                vision.save_last_environment_map('envmap%07d.png'%frame)

            if options.plot_receptors:
                if frame%50==0:
                    print 'saving',frame
                R = vision.get_last_retinal_imageR()
                G = vision.get_last_retinal_imageG()
                B = vision.get_last_retinal_imageB()
                if options.plot_emds:
                    emds = vision.get_last_emd_outputs()
                else:
                    emds = None

                fname = 'receptors%07d'%frame
                fig = fsee.plot_utils.plot_receptor_and_emd_fig(
                    R=R,G=G,B=B,
                    emds=emds,
                    scale=5e-4,
                    figsize=(6.40,4.80),
                    dpi=100,
                    save_fname=fname+'.png',
                    optics=vision.get_optics(),
                    proj='stere',
                    subplot_titles_enabled=False,
                    basemap_lw=0.5,
                    )
                pylab.close(fig)

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--save-envmap", action='store_true',
                      default=False)

    parser.add_option("--plot-receptors", action='store_true',
                      default=False)

    parser.add_option("--plot-emds", action='store_true',
                      default=False)

    analysis_options.add_common_options( parser )
    (options, args) = parser.parse_args()

    if options.plot_emds:
        assert options.plot_receptors

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args):
        parser.print_help()
        return

    doit( options=options,
         )

if __name__=='__main__':
    main()

