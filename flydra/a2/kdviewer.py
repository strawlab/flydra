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
import scipy.io
import pytz, datetime
pacific = pytz.timezone('US/Pacific')

try:
    import cgkit.cgtypes as cgtypes # cgkit 2
except ImportError, err:
    import cgtypes # cgkit 1
import flydra.a2.pos_ori2fu

def print_cam_props(camera):
    print 'camera.parallel_projection = ',camera.parallel_projection
    print 'camera.focal_point = ',camera.focal_point
    print 'camera.position = ',camera.position
    print 'camera.view_angle = ',camera.view_angle
    print 'camera.view_up = ',camera.view_up
    print 'camera.clipping_range = ',camera.clipping_range
    print 'camera.parallel_scale = ',camera.parallel_scale
    
def doit(filename,
         show_obj_ids=False,
         obj_start=None,
         obj_end=None,
         obj_only=None,
         show_n_longest=None,
         radius=0.002, # in meters
         min_length=10,
         show_saccades = True,
         show_observations = False,
         show_saccade_times = False,
         draw_stim_func_str = None,
         use_kalman_smoothing=True,
         fps=100.0,
         vertical_scale=False,
         max_vel=0.25,
         show_only_track_ends = False,
         save_still = False,
         exclude_vel_mps = None,
         ):

    try:
        sys.path.insert(0,os.curdir)
        mat_data = scipy.io.mio.loadmat(filename)
    except IOError, err:
        mat_data = None
        
    if mat_data is not None:
        obj_ids = mat_data['kalman_obj_id']
        obj_ids = obj_ids.astype( numpy.uint32 )
        print 'max obj_id',numpy.amax(obj_ids)
    
        obs_obj_ids = obj_ids # use as observation length, even though these aren't observations
        use_obj_ids = numpy.unique(obj_ids)
        is_mat_file = True
    else:
        kresults = PT.openFile(filename,mode="r")
        obs_obj_ids = kresults.root.kalman_observations.read(field='obj_id')
        my_obj_ids = kresults.root.kalman_estimates.read(field='obj_id')
        use_obj_ids = numpy.unique(obs_obj_ids)
        print 'max obj_id',numpy.amax(use_obj_ids)
        is_mat_file = False
    
    if show_n_longest is not None:
        if ((obj_start is not None) or
            (obj_end is not None) or
            (obj_only is not None)):
            raise ValueError("show_n_longest incompatible with other limiters")
        
        if is_mat_file:
            frames = mat_data['kalman_frame']
            if exclude_vel_mps is not None:
                x = mat_data['x']            
                y = mat_data['y']            
                z = mat_data['z']            
        else:
            frames = kresults.root.kalman_observations.read(field='frame')
            if exclude_vel_mps is not None:
                x = kresults.root.kalman_observations.read(field='x')
                y = kresults.root.kalman_observations.read(field='y')
                z = kresults.root.kalman_observations.read(field='z')
        obj_ids_by_n_frames = {}
        for i,obj_id in enumerate(use_obj_ids):
            if i%100==0:
                print 'doing %d of %d'%(i,len(use_obj_ids))
            obs_cond = obs_obj_ids==obj_id
            obj_frames = frames[obs_cond]
            n_frames = obj_frames[-1]-obj_frames[0]
            if exclude_vel_mps is not None:
                x_frames = x[obs_cond]
                y_frames = y[obs_cond]
                z_frames = z[obs_cond]

                dx = x_frames[1:]-x_frames[:-1]
                dy = y_frames[1:]-y_frames[:-1]
                dz = z_frames[1:]-z_frames[:-1]

                dist = numpy.sqrt(dx**2 + dy**2 + dz**2)
                total_dist = dist.sum()
                total_time = n_frames/fps
                mean_vel = total_dist / total_time
                if mean_vel < exclude_vel_mps:
                    continue
            obj_ids_by_n_frames.setdefault( n_frames, [] ).append( obj_id )
        n_frames_list = obj_ids_by_n_frames.keys()
        n_frames_list.sort()

        obj_only = []
        while len(n_frames_list):
            n_frames = n_frames_list.pop()
            obj_ids = obj_ids_by_n_frames[n_frames]
            obj_only.extend( obj_ids)
            if len(obj_only) > show_n_longest:
                break

        print 'longest traces = ',obj_only
        use_obj_ids = numpy.array(obj_only)

    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_end is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]
    if obj_only is not None:
        use_obj_ids = numpy.array(obj_only)

    #################
    rw = tvtk.RenderWindow(size=(1024, 768))
    
    if show_obj_ids:
        # Because I can't get black text right now (despite trying),
        # make background blue to see white text. - ADS
        ren = tvtk.Renderer(background=(0.6,0.6,1.0)) # blue
    else:
        ren = tvtk.Renderer(background=(1.0,1.0,1.0)) # white
        
    camera = ren.active_camera

    if 0:
        camera.parallel_projection =  0
        camera.focal_point =  (0.52719625417063776, 0.15695605837665305, 0.10876143712478874)
        camera.position =  (0.39743071773877131, -0.4114652255728779, 0.097431169175252269)
        camera.view_angle =  30.0
        camera.view_up =  (-0.072067516965519787, -0.0034285481144054573, 0.99739386305323308)
        camera.clipping_range =  (0.25210456649736646, 1.0012868084455435)
        camera.parallel_scale =  0.294595461395
    if 0:
        camera.parallel_projection =  0
        camera.focal_point =  (0.49827304637942593, 0.20476671221773424, 0.090222461715116345)
        camera.position =  (0.41982519417302594, -0.55501151899867784, 0.40089956585064912)
        camera.view_angle =  30.0
        camera.view_up =  (0.025460553314687551, 0.37610935779812088, 0.92622541057865326)
        camera.clipping_range =  (0.38425211041324286, 1.3299558503823485)
        camera.parallel_scale =  0.294595461395
        
    rw.add_renderer(ren)
    rwi = tvtk.RenderWindowInteractor(render_window=rw,
                                      interactor_style = tvtk.InteractorStyleTrackballCamera())
    
    lut = tvtk.LookupTable(hue_range = (0.667, 0.0))
    actors = []
    actor2obj_id = {}
    #################
    
    if show_only_track_ends:
        track_end_verts = []
    
    ca = core_analysis.CachingAnalyzer()
    #last_time = None
    if not len(use_obj_ids):
        raise ValueError('no trajectories to plot')
        
    for obj_id_enum,obj_id in enumerate(use_obj_ids):
        if (obj_id_enum%100)==0 and len(use_obj_ids) > 5:
            print 'obj_id %d of %d'%(obj_id_enum,len(use_obj_ids))
            if 0:
                import time
                now = time.time()
                if last_time is not None:
                    dur = now-last_time
                    print dur,'seconds'
                last_time = now

        if 1:
            my_idx = numpy.nonzero(my_obj_ids==obj_id)[0]
            my_rows = kresults.root.kalman_estimates.readCoordinates(my_idx)
            my_timestamp = my_rows['timestamp'][0]
            dur = my_rows['timestamp'][-1] - my_timestamp
            print '%d 3D triangulation started at %s (took %.2f seconds)'%(obj_id,datetime.datetime.fromtimestamp(my_timestamp,pacific),dur)
            print '  estimate frames: %d - %d (%d frames, %.1f sec at %.1f fps)'%(
                my_rows['frame'][0],
                my_rows['frame'][-1],
                my_rows['frame'][-1]-my_rows['frame'][0],
                (my_rows['frame'][-1]-my_rows['frame'][0])/fps,
                fps)
            
        if show_observations:
            obs_idx = numpy.nonzero(obs_obj_ids==obj_id)[0]
            obs_rows = kresults.root.kalman_observations.readCoordinates(obs_idx)
            obs_x = obs_rows['x']
            obs_y = obs_rows['y']
            obs_z = obs_rows['z']
            obs_frames = obs_rows['frame']
            print '  observation frames: %d - %d'%(obs_frames[0], obs_frames[-1])
            obs_X = numpy.vstack((obs_x,obs_y,obs_z)).T

            pd = tvtk.PolyData()
            pd.points = obs_X

            g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                             vector_mode = 'use_vector',
                             input=pd)
            ss = tvtk.SphereSource(radius = radius/3)
            g.source = ss.output
            vel_mapper = tvtk.PolyDataMapper(input=g.output)
            a = tvtk.Actor(mapper=vel_mapper)
            a.property.color = 1.0, 0.0, 0.0
            actors.append(a)
            actor2obj_id[a] = obj_id

        if not is_mat_file:
            n_observations = numpy.sum(obs_obj_ids == obj_id)
            if int(n_observations) < int(min_length):
                continue
            data_file = kresults
        else:
            data_file = mat_data

        if not show_only_track_ends:
            try:
                results = ca.calculate_trajectory_metrics(obj_id,
                                                          data_file,
                                                          use_kalman_smoothing=use_kalman_smoothing,
                                                          frames_per_second=fps,
                                                          method='position based',
                                                          method_params={'downsample':1,
                                                                         })
            except Exception, err:
                if 1:
                    raise
                else:
                    print 'ERROR: while processing obj_id %d, skipping this obj_id'%obj_id
                    print err
                    continue
            verts = results['X_kalmanized']
            speeds = results['speed_kalmanized']

        else:
            rows=ca.load_data(obj_id,
                              data_file)
            
            x0 = rows.field('x')[0]
            x1 = rows.field('x')[-1]
            
            y0 = rows.field('y')[0]
            y1 = rows.field('y')[-1]
            
            z0 = rows.field('z')[0]
            z1 = rows.field('z')[-1]
            
            track_end_verts.append( (x0,y0,z0) )
            track_end_verts.append( (x1,y1,z1) )

        if 0:
            print 'WARNING: limiting data'
            slicer = slice(1350,1600)
            verts = verts[slicer]
            speeds = speeds[slicer]
        
        
        if show_saccades:
            saccades = ca.detect_saccades(obj_id,
                                          data_file,
                                          use_kalman_smoothing=use_kalman_smoothing,
                                          frames_per_second=fps,
                                          method='position based',
                                          method_params={'downsample':1,
                                                         'horizontal only':False,
                                                         #'horizontal only':True,
                                                         })
            saccade_verts = saccades['X']
            saccade_times = saccades['times']

        
        #################

        if not show_only_track_ends:
            pd = tvtk.PolyData()
            pd.points = verts
            pd.point_data.scalars = speeds
#            if numpy.any(speeds>max_vel):
#                print 'WARNING: maximum speed (%.3f m/s) exceeds color map max'%(speeds.max(),)

            g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                             vector_mode = 'use_vector',
                             input=pd)
            ss = tvtk.SphereSource(radius = radius)
            g.source = ss.output
            vel_mapper = tvtk.PolyDataMapper(input=g.output)
            vel_mapper.lookup_table = lut
            vel_mapper.scalar_range = 0.0, float(max_vel)
            a = tvtk.Actor(mapper=vel_mapper)
            if show_observations:
                a.property.opacity = 0.3
            actors.append(a)
            actor2obj_id[a] = obj_id

        if 0:
            # show time of each saccade
            for X,showtime in zip(verts,results['time_kalmanized']):
                ta = tvtk.TextActor(input=str( showtime ))
                ta.property.color = 0.0, 0.0, 0.0 # black
                ta.position_coordinate.coordinate_system = 'world'
                ta.position_coordinate.value = tuple(X)
                actors.append(ta)
                actor2obj_id[a] = obj_id

        if show_obj_ids:
            print 'showing ob_jd %d at %s'%(obj_id,str(verts[0]))
            obj_id_ta = tvtk.TextActor(input=str( obj_id )+' start')
            obj_id_ta.property = tvtk.Property2D(color = (0.0, 0.0, 0.0), # black
                                                 )
#            obj_id_ta.property = tvtk.TextProperty(color = (0.0, 0.0, 0.0), # black
#                                                   )
            #obj_id_ta.property.color = 0.0, 0.0, 0.0 # black
            #obj_id_ta.set(color = (0.0, 0.0, 0.0)) # black
            obj_id_ta.position_coordinate.coordinate_system = 'world'
            obj_id_ta.position_coordinate.value = tuple(verts[0])
            actors.append(obj_id_ta)
            actor2obj_id[a] = obj_id
            

        ##################
    
        if show_saccades:
            pd = tvtk.PolyData()
            pd.points = saccade_verts

            g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                             vector_mode = 'use_vector',
                             input=pd)
            ss = tvtk.SphereSource(radius = 0.005,
                                   theta_resolution=3,
                                   phi_resolution=3,
                                   )
            g.source = ss.output
            mapper = tvtk.PolyDataMapper(input=g.output)
            a = tvtk.Actor(mapper=mapper)
            #a.property.color = (0,1,0) # green
            a.property.color = (0,0,0) # black
            a.property.opacity = 0.3
            actors.append(a)
            actor2obj_id[a] = obj_id

        if show_saccade_times:
            # show time of each saccade
            for X,showtime in zip(saccade_verts,saccade_times):
                ta = tvtk.TextActor(input=str( showtime ))
                ta.property.color = 0.0, 0.0, 0.0 # black
                ta.position_coordinate.coordinate_system = 'world'
                ta.position_coordinate.value = tuple(X)
                actors.append(ta)
                actor2obj_id[a] = obj_id

    if not is_mat_file:
        kresults.close()
    
    ################################

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
        actors.extend( stim_actors )
        print '*'*80,'drew with custom code'
    
    ################################
    
    if 0:
        a=tvtk.AxesActor(normalized_tip_length=(0.4, 0.4, 0.4),
                         normalized_shaft_length=(0.6, 0.6, 0.6),
                         shaft_type='cylinder')
        actors.append(a)

    if show_only_track_ends:
        pd = tvtk.PolyData()

        verts = numpy.array( track_end_verts )

        if 1:
            print 'limiting ends shown to approximate arena boundaries'
            cond = (verts[:,2] < 0.25) & (verts[:,2] > -0.05)
            #cond = cond & (verts[:,1] < 0.29) & (verts[:,1] > 0.0)
            showverts = verts[cond]
        else:
            showverts = verts
            
        pd.points = showverts

        g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                         vector_mode = 'use_vector',
                         input=pd)
        ss = tvtk.SphereSource(radius = 0.005,
                               theta_resolution=3,
                               phi_resolution=3,
                               )
        g.source = ss.output
        mapper = tvtk.PolyDataMapper(input=g.output)
        a = tvtk.Actor(mapper=mapper)
        #a.property.color = (0,1,0) # green
        a.property.color = (1,0,0) # red
        a.property.opacity = 0.3
        actors.append(a)
            
    for a in actors:
        ren.add_actor(a)
        
    if 0:
        # this isn't working yet
        axes2 = tvtk.CubeAxesActor2D()
        axes2.camera = ren.active_camera
        #axes2.input = verts
        ren.add_actor(axes2)
        
    if not show_only_track_ends:
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
        
        if 1:
            # Use the ScalarBarWidget so we can drag the scalar bar around.
            sc_bar_widget = tvtk.ScalarBarWidget(interactor=rwi,
                                                 scalar_bar_actor=scalar_bar)


            rwi.initialize()
            sc_bar_widget.enabled = True

    if 1:
        picker = tvtk.CellPicker(tolerance=1e-9)
        #print 'dir(picker)',dir(picker)
        def annotatePick(object, event):
            # XXX keep all this math for reference with pos_ori2fu.py
            vtm = numpy.array([ [ren.active_camera.view_transform_matrix.get_element(i,j) for j in range(4)] for i in range(4)])
            #print 'camera.view_transform_matrix = ',vtm
            vtmcg = cgtypes.mat4(list(vtm.T))
            view_translation,view_rotation_mat4,view_scaling = vtmcg.decompose()
            q = aa=cgtypes.quat().fromMat(view_rotation_mat4)
            #print 'orientation quaternion',q
            aa=q.toAngleAxis()
            #print
            #print 'camera.position = ',ren.active_camera.position
            cpos = view_rotation_mat4.inverse()*-view_translation # same as camera.position
            #print 'view_rotation_mat4.inverse()*-view_translation',cpos
            #print 'view_scaling',view_scaling

            #print 'camera.orientation_wxyz = ',ren.active_camera.orientation_wxyz
            #print 'aa',aa
            #print 'q',q
            #print 'camera.focal_point = ',ren.active_camera.focal_point
            #print 'camera.view_up = ',ren.active_camera.view_up

            upq = view_rotation_mat4.inverse()*cgtypes.vec3(0,1,0) # get view up
            vd = view_rotation_mat4.inverse()*cgtypes.vec3(0,0,-1) # get view forward
            #print 'upq',upq
            #print 'viewdir1',(vd).normalize()
            #print 'viewdir2',(cgtypes.vec3(ren.active_camera.focal_point)-cpos).normalize()
            #print
            p = cgtypes.vec3(camera.position)
            print 'animation path variable (t=time) (t,x,y,z,qw,qx,qy,qz):'
            print 't', p[0], p[1], p[2], q.w, q.x, q.y, q.z
            print

            if not picker.cell_id < 0:
                found = sets.Set([])
                for actor in picker.actors:
                    objid = actor2obj_id[actor]
                    found.add(objid)
                found = list(found)
                found.sort()
                print ' '.join(map(str,found))

        picker.add_observer('EndPickEvent', annotatePick)
        rwi.picker = picker

    if not save_still:
        rwi.start()
        print_cam_props( ren.active_camera )
    else:
        imf = tvtk.WindowToImageFilter(input=rw)
        writer = tvtk.PNGWriter()

        imf.update()
        imf.modified()
        writer.input = imf.output
        fname = 'kdviewer_output.png'
        writer.file_name = fname
        writer.write()
        
def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

##    parser.add_option("--debug", type="int",
##                      help="debug level",
##                      metavar="DEBUG")
        
    parser.add_option("--start", type="int",
                      help="first object ID to plot",
                      metavar="START")
        
    parser.add_option("--stop", type="int",
                      help="last object ID to plot",
                      metavar="STOP")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")
    
    parser.add_option("--draw-stim",
                      type="string",
                      dest="draw_stim_func_str",
                      default="flydra.a2.conditions_draw:draw_default_stim",
                      )
    
    parser.add_option("--n-top-traces", type="int",
                      help="show N longest traces")
    
    parser.add_option("--min-length", dest="min_length", type="int",
                      help="minimum number of tracked points (not observations!) required to plot",
                      default=10,)
    
    parser.add_option("--radius", type="float",
                      help="radius of line (in meters)",
                      default=0.002,
                      metavar="RADIUS")
    
    parser.add_option("--max-vel", type="float",
                      help="maximum velocity of colormap",
                      dest='max_vel',
                      default=0.25)
    
    parser.add_option("--exclude-vel", type="float",
                      help="exclude traces with mean velocity less than this value",
                      dest='exclude_vel_mps',
                      default=None)
    
    parser.add_option("--show-obj-ids", action='store_true',dest='show_obj_ids',
                      help="show object ID numbers at start of trajectory")

    parser.add_option("--show-saccades", action='store_true',dest='show_saccades',
                      help="show saccades")

    parser.add_option("--show-only-track-ends", action='store_true',dest='show_only_track_ends')

    parser.add_option("--show-observations", action='store_true',dest='show_observations',
                      help="show observations")

    parser.add_option("--show-saccade-times", action='store_true',dest='show_saccade_times',
                      help="show saccade times")

    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--vertical-scale", action='store_true',dest='vertical_scale',
                      help="scale bar has vertical orientation")

    parser.add_option("--save-still", action='store_true',dest='save_still',
                      help="save still image as kdviewer_output.png")

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

        if options.start is not None or options.stop is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    doit(filename=h5_filename,
         obj_start=options.start,
         obj_end=options.stop,
         obj_only=options.obj_only,
         use_kalman_smoothing=options.use_kalman_smoothing,
         show_n_longest=options.n_top_traces,
         show_obj_ids = options.show_obj_ids,
         radius = options.radius,
         min_length = options.min_length,
         show_saccades = options.show_saccades,
         show_observations = options.show_observations,
         show_saccade_times = options.show_saccade_times,
         draw_stim_func_str = options.draw_stim_func_str,
         fps = 100.0,
         vertical_scale = options.vertical_scale,
         max_vel = options.max_vel,
         show_only_track_ends = options.show_only_track_ends,
         save_still = options.save_still,
         exclude_vel_mps = options.exclude_vel_mps,
         )
    
if __name__=='__main__':
    main()

