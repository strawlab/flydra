from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import sets, os, sys, math
sys.path.insert(0,os.curdir)
from enthought.tvtk.api import tvtk
import numpy
import tables as PT
from optparse import OptionParser
import core_analysis
import stimulus_positions
import scipy.io
import conditions

import pkg_resources
from flydra.a2.pos_ori2fu import pos_ori2fu
try:
    import cgtypes # cgtypes 1.2.x
except ImportError, err:
    import cgkit.cgtypes as cgtypes # cgkit 2.0

class AnimationPath(object):
    def __init__(self,fname):
        fd = open(fname,mode='r')
        data = []
        for line in fd.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            if not len(line):
                continue
            split_line = line.split()
            fl = map(float,split_line)
            data.append(fl)
        self.data = numpy.array(data)
        print 'self.data',self.data
    def get_pos_ori(self,t):
        file_ts = self.data[:,0]
        tdiff = file_ts[1:]-file_ts[:-1]
        if tdiff.min() < 0.0:
            raise ValueError("animation path times go backwards!")
        t = t%file_ts[-1] # wrap around
        lower_idx = numpy.nonzero((file_ts <= t))[0][-1]
        upper_idx = lower_idx + 1
        lower_t = file_ts[lower_idx]
        file_dt = file_ts[upper_idx]-lower_t
        frac = (t-lower_t)/file_dt
        
        pos_lower = self.data[lower_idx,1:4]
        ori_lower = cgtypes.quat(self.data[lower_idx,4:8])

        pos_upper = self.data[upper_idx,1:4]
        ori_upper = cgtypes.quat(self.data[upper_idx,4:8])

        pos = frac*(pos_upper-pos_lower) + pos_lower
        ori = cgtypes.slerp(frac,ori_lower,ori_upper)

        return pos, ori

def doit(filename,
         obj_only=None,
         radius=0.002, # in meters
         min_length=10,
         use_kalman_smoothing=True,
         data_fps=100.0,
         save_fps = 25,
         vertical_scale=False,
         max_vel='auto',
         draw_stim_func_str=None,
         floor = True,
         animation_path_fname = None,
         output_dir='.',
         cam_only_move_duration=5.0,
         ):
    
    if animation_path_fname is None:
        animation_path_fname = pkg_resources.resource_filename(__name__,"kdmovie_saver_default_path.kmp")
    camera_animation_path = AnimationPath(animation_path_fname)

    mat_data = None
    try:
        try:
            data_path, data_filename = os.path.split(filename)
            data_path = os.path.expanduser(data_path)
            sys.path.insert(0,data_path)
            mat_data = scipy.io.mio.loadmat(data_filename)
        finally:
            del sys.path[0]
    except IOError, err:
        print 'not a .mat file at %s, treating as .hdf5 file'%(os.path.join(data_path, data_filename))
    
    if mat_data is not None:
        obj_ids = mat_data['kalman_obj_id']
        obj_ids = obj_ids.astype( numpy.uint32 )
        obs_obj_ids = obj_ids # use as observation length, even though these aren't observations
        is_mat_file = True
    else:
        kresults = PT.openFile(filename,mode="r")
        obs_obj_ids = kresults.root.kalman_observations.read(field='obj_id')
        is_mat_file = False

    filename_trimmed = os.path.split(os.path.splitext(filename)[0])[-1]
    
    assert obj_only is not None

    #################
    rw = tvtk.RenderWindow(size=(1024, 768))
    
    ren = tvtk.Renderer(background=(1.0,1.0,1.0))
    camera = ren.active_camera
        
    rw.add_renderer(ren)
    
    lut = tvtk.LookupTable(hue_range = (0.667, 0.0))
    #################
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    ca = core_analysis.CachingAnalyzer()
    
    if len(obj_only)==1:
        animate_path = True
        # allow path to grow during trajectory
    else:
        animate_path = False
        obj_verts = []
        speeds = []
 
    for obj_id in obj_only:

        if not is_mat_file:
            n_observations = numpy.sum(obs_obj_ids == obj_id)
            if int(n_observations) < int(min_length):
                print 'WARNING: n_observerations < min_length'
            data_file = kresults
        else:
            data_file = mat_data

        print 'loading %d'%obj_id
        results = ca.calculate_trajectory_metrics(obj_id,
                                                  data_file,
                                                  use_kalman_smoothing=use_kalman_smoothing,
                                                  frames_per_second=data_fps,
                                                  method='position based',
                                                  method_params={'downsample':1,
                                                                 })
    
        if len(obj_only)==1:
            obj_verts = results['X_kalmanized']
            speeds = results['speed_kalmanized']
        else:
            obj_verts.append( results['X_kalmanized'] )
            speeds.append( results['speed_kalmanized'] )
            
    if not len(obj_only)==1:
        obj_verts = numpy.concatenate(obj_verts,axis=0)
        speeds = numpy.concatenate(speeds,axis=0)
        
    ####################### start draw permanently on stuff ############################
    
    if draw_stim_func_str:
        def my_import(name):
            mod = __import__(name)
            components = name.split('.')
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod
        draw_stim_module_name, draw_stim_func_name = draw_stim_func_str.strip().split(':')
        draw_stim_module = my_import(draw_stim_module_name)
        draw_stim_func = getattr(draw_stim_module, draw_stim_func_name)
        stim_actors = draw_stim_func(filename=filename)
        print '*'*80,'drew with custom code'
        for stim_actor in stim_actors:
            ren.add_actor( stim_actor )
            
    ####################### 

    if max_vel == 'auto':
        max_vel = speeds.max()
    else:
        max_vel = float(max_vel)
    vel_mapper = tvtk.PolyDataMapper()
    vel_mapper.lookup_table = lut
    vel_mapper.scalar_range = 0.0, max_vel

    if 1:
        # Create a scalar bar
        if vertical_scale:
            scalar_bar = tvtk.ScalarBarActor(orientation='vertical',
                                             width=0.08, height=0.4)
        else: 
            scalar_bar = tvtk.ScalarBarActor(orientation='horizontal',
                                             width=0.4, height=0.08)
        scalar_bar.title = "Speed (m/s)"
        scalar_bar.lookup_table = vel_mapper.lookup_table
        
        scalar_bar.property.color = 0.0, 0.0, 0.0 # black

        scalar_bar.title_text_property.color = 0.0, 0.0, 0.0
        scalar_bar.title_text_property.shadow = False
        
        scalar_bar.label_text_property.color = 0.0, 0.0, 0.0
        scalar_bar.label_text_property.shadow = False
        
 	scalar_bar.position_coordinate.coordinate_system = 'normalized_viewport'
        if vertical_scale:
            scalar_bar.position_coordinate.value = 0.01, 0.01, 0.0
        else:
            scalar_bar.position_coordinate.value = 0.1, 0.01, 0.0
        
        ren.add_actor( scalar_bar )

    imf = tvtk.WindowToImageFilter(input=rw)
    writer = tvtk.PNGWriter()

    ####################### end draw permanently on stuff ############################

    save_dt = 1.0/save_fps

    if animate_path:
        data_dt = 1.0/data_fps
        n_frames = len(obj_verts)
        dur = n_frames*data_dt
    else:
        data_dt = 0.0
        dur = 0.0
        
    t_now = 0.0
    frame_number = 0
    while t_now <= dur:
        frame_number += 1
        t_now += save_dt
        print 't_now',t_now

        pos, ori = camera_animation_path.get_pos_ori(t_now)
        focal_point, view_up = pos_ori2fu(pos,ori)
        #print 'focal_point, view_up',focal_point, view_up

        camera.position = tuple(pos)
        #camera.focal_point = (focal_point[0], focal_point[1], focal_point[2])
        #camera.view_up = (view_up[0], view_up[1], view_up[2])
        camera.focal_point = tuple(focal_point)
        camera.view_up = tuple(view_up)

        if data_dt != 0.0:
            draw_n_frames = int(math.ceil(t_now / data_dt))
        else:
            draw_n_frames = len(obj_verts)
        print 'frame_number, draw_n_frames', frame_number, draw_n_frames

        #################

        pd = tvtk.PolyData()
        pd.points = obj_verts[:draw_n_frames]
        pd.point_data.scalars = speeds
        if numpy.any(speeds>max_vel):
            print 'WARNING: maximum speed (%.3f m/s) exceeds color map max'%(speeds.max(),)

        g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                         vector_mode = 'use_vector',
                         input=pd)
        vel_mapper.input = g.output
        ss = tvtk.SphereSource(radius = radius)
        g.source = ss.output
        a = tvtk.Actor(mapper=vel_mapper)

        ##################

        ren.add_actor(a)

        if 1:
            imf.update()
            imf.modified()
            writer.input = imf.output
            fname = 'movie_%s_%03d_frame%05d.png'%(filename_trimmed,obj_id,frame_number)
            full_fname = os.path.join(output_dir, fname)
            writer.file_name = full_fname
            writer.write()

        ren.remove_actor(a)
        
    ren.add_actor(a) # restore actors removed
    dur = dur+cam_only_move_duration

    while t_now < dur:
        frame_number += 1
        t_now += save_dt
        print 't_now',t_now

        pos, ori = camera_animation_path.get_pos_ori(t_now)
        focal_point, view_up = pos_ori2fu(pos,ori)
        camera.position = tuple(pos)
        camera.focal_point = tuple(focal_point)
        camera.view_up = tuple(view_up)
        if 1:
            imf.update()
            imf.modified()
            writer.input = imf.output
            if len(obj_only)==1:
                fname = 'movie_%s_%03d_frame%05d.png'%(filename_trimmed,obj_id,frame_number)
            else:
                fname = 'movie_%s_many_frame%05d.png'%(filename_trimmed,frame_number)
            full_fname = os.path.join(output_dir, fname)
            writer.file_name = full_fname
            writer.write()

    if not is_mat_file:
        kresults.close()
        
def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")
    
    parser.add_option("--draw-stim",
                      type="string",
                      dest="draw_stim_func_str",
                      default="flydra.a2.conditions_draw:draw_default_stim",
                      )
    
    parser.add_option("--cam-only-move-duration", type="float", # formerly called hover
                      dest="cam_only_move_duration",
                      default=5.0)

    parser.add_option("--output-dir", type="string",
                      dest="output_dir")

    parser.add_option("--animation_path_fname",type="string",
                      dest="animation_path_fname")
    
    parser.add_option("--min-length", dest="min_length", type="int",
                      help="minimum number of tracked points (not observations!) required to plot",
                      default=10,)
    
    parser.add_option("--radius", type="float",
                      help="radius of line (in meters)",
                      default=0.002,
                      metavar="RADIUS")
    
    parser.add_option("--max-vel", type="string",
                      help="maximum velocity of colormap",
                      dest='max_vel',
                      default='auto')
    
    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--vertical-scale", action='store_true',dest='vertical_scale',
                      help="scale bar has vertical orientation")

    (options, args) = parser.parse_args()

    if options.filename is not None:
        args.append(options.filename)
        
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return
        
    h5_filename=args[0]

    if options.obj_only is not None:
        options.obj_only = options.obj_only.replace(',',' ')
        seq = map(int,options.obj_only.split())
        options.obj_only = seq

    if options.output_dir is None:
        options.output_dir = os.curdir

    doit(filename=h5_filename,
         obj_only=options.obj_only,
         cam_only_move_duration=options.cam_only_move_duration,
         use_kalman_smoothing=options.use_kalman_smoothing,
         radius = options.radius,
         min_length = options.min_length,
         vertical_scale = options.vertical_scale,
         draw_stim_func_str = options.draw_stim_func_str,
         max_vel = options.max_vel,
         floor=True,
         animation_path_fname = options.animation_path_fname,
         output_dir = options.output_dir,
         )
    
if __name__=='__main__':
    main()

