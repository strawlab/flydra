from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import sets, os, sys, math

import pkg_resources
from enthought.tvtk.api import tvtk
import numpy
import tables as PT
from optparse import OptionParser
import core_analysis
import scipy.io
import pytz, datetime
import pkg_resources
import flydra.reconstruct as reconstruct

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

def do_show_cameras(results, renderers, frustums=True, labels=True, centers=True):
    if isinstance(results,reconstruct.Reconstructor):
        R = results
    else:
        R = reconstruct.Reconstructor(results)

    R = R.get_scaled( R.get_scale_factor() )

    actors = []
    if centers:
        cam_centers = tvtk.Points()

        for cam_id, pmat in R.Pmat.iteritems():
            X = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            X = numpy.array(X.flat)
            cam_centers.insert_next_point(*X)

        points_poly_data = tvtk.PolyData( points=cam_centers )
        ball = tvtk.SphereSource(radius=0.020,
                                 theta_resolution=25,
                                 phi_resolution=25)
        balls = tvtk.Glyph3D(scale_mode='data_scaling_off',
                             vector_mode = 'use_vector',
                             input=points_poly_data,
                             source = ball.output)
        mapBalls = tvtk.PolyDataMapper(input=balls.output)
        ballActor = tvtk.Actor(mapper=mapBalls)
        ballActor.property.diffuse_color = (1,0,0)
        ballActor.property.specular = .3
        ballActor.property.specular_power = 30
        for ren in renderers:
            ren.add_actor(ballActor)

    if frustums:
        line_points = tvtk.Points()
        polys = tvtk.CellArray()
        point_num = 0

        for cam_id in R.Pmat.keys():
            pmat = R.get_pmat( cam_id )
            width,height = R.get_resolution( cam_id )

            # cam center
            C = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            C = numpy.array(C.flat)

            z = 1
            first_vert = None

            for x,y in ((0,0),(0,height-1),(width-1,height-1),(width-1,0)):
                    x2d = x,y,z
                    X = R.find3d_single_cam(cam_id,x2d) # returns column matrix
                    X = X.flat
                    X = X[:3]/X[3]

                    line_points.insert_next_point(*C)
                    point_num += 1

                    U = X-C # direction
                    # rescale to unit length
                    U=U/math.sqrt(U[0]**2 + U[1]**2 + U[2]**2)
                    X = C+500.0*U

                    line_points.insert_next_point(*X)
                    point_num += 1

                    if first_vert is None:
                        first_vert = point_num-2
                    else:
                        polys.insert_next_cell(4)
                        polys.insert_cell_point(point_num-4)
                        polys.insert_cell_point(point_num-3)
                        polys.insert_cell_point(point_num-1)
                        polys.insert_cell_point(point_num-2)

            polys.insert_next_cell(4)
            polys.insert_cell_point(point_num-2)
            polys.insert_cell_point(point_num-1)
            polys.insert_cell_point(first_vert+1)
            polys.insert_cell_point(first_vert)

        profileData = tvtk.PolyData(points=line_points, polys=polys)
        profileMapper = tvtk.PolyDataMapper(input=profileData)
        profile = tvtk.Actor(mapper=profileMapper)
        p = profile.property
        p.opacity = 0.1
        p.diffuse_color = 1,0,0
        p.specular = .3
        p.specular_power = 30
        for ren in renderers:
            ren.add_actor(profile)

    if labels:
        for cam_id, pmat in R.Pmat.iteritems():
            X = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            X = numpy.array(X.flat)

            ta = tvtk.TextActor(input=cam_id)
            #ta.set(scaled_text=True, height=0.05)
            pc = ta.position_coordinate
            pc.coordinate_system = 'world'
            pc.value = X

            for ren in renderers:
                ren.add_actor(ta)

class PluginLoader:
    def __init__(self):
        self.all_names = []
        pkg_env = pkg_resources.Environment()
        self.EP = 'flydra.kdviewer.plugins'
        for name in pkg_env:
            egg = pkg_env[name][0]
            for name in egg.get_entry_map(self.EP):
                egg.activate()
                self.all_names.append(name)
    def __call__(self,draw_stim_func_str):
        pkg_env = pkg_resources.Environment()
        for name in pkg_env:
            egg = pkg_env[name][0]
            for name in egg.get_entry_map(self.EP):
                if name != draw_stim_func_str:
                    continue
                entry_point = egg.get_entry_info(self.EP,draw_stim_func_str)
                PluginFunc = entry_point.load()
                return PluginFunc
        raise ValueError('Could not find stimulus drawing plugin. Available: %s'%( str(self.all_names), ))
plugin_loader = PluginLoader()

def doit(filename,
         show_obj_ids=False,
         start=None,
         stop=None,
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
         fps=None,
         vertical_scale=False,
         max_vel=0.25,
         show_only_track_ends = False,
         save_still = False,
         exclude_vel_mps = None,
         exclude_vel_data = 'kalman',
         stereo = False,
         dynamic_model = None,
         show_cameras=False,
         obj_color=False,
         link_all_simultaneous_objs=True,
         ):

    assert exclude_vel_data in ['kalman','observations'] # kalman means smoothed or filtered, depending on use_kalman_smoothing

    if link_all_simultaneous_objs:
        allsave = []

    if not use_kalman_smoothing:
        if (fps is not None) or (dynamic_model is not None):
            print >> sys.stderr, 'ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting fps and dynamic model options (--fps and --dynamic-model)'
            sys.exit(1)
    else:
        if fps is None:
            fps = 100.0
            import warnings
            warnings.warn('Setting fps to default value of %f'%fps)

    ca = core_analysis.CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(filename)

    if not is_mat_file:
        mat_data = None

    if show_n_longest is not None:
        if ((obj_start is not None) or
            (obj_end is not None) or
            (obj_only is not None)):
            raise ValueError("show_n_longest incompatible with other limiters")

        obj_ids_by_n_frames = {}
        for i,obj_id in enumerate(use_obj_ids):
            if i%100==0:
                print 'doing %d of %d'%(i,len(use_obj_ids))

            if not ca.has_obj_id(obj_id, data_file):
                continue
            try:
                rows = ca.load_data( obj_id, data_file,
                                     use_kalman_smoothing=use_kalman_smoothing,
                                     kalman_dynamic_model = dynamic_model,
                                     frames_per_second=fps)
            except core_analysis.ObjectIDDataError,err:
                continue

            frames = rows['frame']
            n_frames = rows['frame'][-1]-rows['frame'][0]+1

            if exclude_vel_mps and exclude_vel_data != 'kalman':
                raise NotImplementedError('')

            if exclude_vel_mps and exclude_vel_data == 'kalman':
                vel = numpy.sqrt(rows['xvel']**2 + rows['yvel']**2 + rows['zvel']**2)
                mean_vel = numpy.mean(vel)
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
    rw = tvtk.RenderWindow(size=(1024, 768),
                           stereo_capable_window=stereo,
                           alpha_bit_planes=True,
                           )
    if stereo:
    ##     rw.stereo_render_on()
    ##     rw.set_stereo_type_to_red_blue()
        rw.set(stereo_type='red_blue',
               stereo_render=stereo)

    if show_obj_ids:
        # Because I can't get black text right now (despite trying),
        # make background blue to see white text. - ADS
        ren = tvtk.Renderer(background=(0.6,0.6,1.0)) # blue
    else:
        ren = tvtk.Renderer(background=(1.0,1.0,1.0)) # white

    camera = ren.active_camera

    if show_cameras:
        if is_mat_file:
            raise RuntimeError('.mat file does not contain camera information')
        do_show_cameras(extra['kresults'], [ren])

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
                                      interactor_style = tvtk.InteractorStyleTrackballCamera(),
                                      #stereo = stereo,
                                      )

    lut = tvtk.LookupTable(hue_range = (0.667, 0.0))
    actors = []
    actor2obj_id = {}
    #################

    if show_only_track_ends:
        track_end_verts = []

    if not len(use_obj_ids):
        raise ValueError('no trajectories to plot')

    had_any_obj_id_data = False
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

        if not is_mat_file:
            # h5 file has timestamps for each frame
            #my_rows = ca.get_recarray(data_file,obj_id,which_data='kalman')
            try:
                my_rows = ca.load_data( obj_id, data_file,
                                        use_kalman_smoothing=use_kalman_smoothing,
                                        kalman_dynamic_model = dynamic_model,
                                        frames_per_second=fps)
            except core_analysis.ObjectIDDataError, err:
                continue

            my_timestamp = my_rows['timestamp'][0]
            dur = my_rows['timestamp'][-1] - my_timestamp
            print '%d 3D triangulation started at %s (took %.2f seconds)'%(obj_id,datetime.datetime.fromtimestamp(my_timestamp,pacific),dur)
            print '  estimate frames: %d - %d (%d frames)'%(
                my_rows['frame'][0],
                my_rows['frame'][-1],
                my_rows['frame'][-1]-my_rows['frame'][0]),
            if fps is None:
                fpses = [60.0, 100.0, 200.0]
            else:
                fpses = [fps]
            for my_fps in fpses:
                    print '(%.1f sec at %.1f fps)'%(
                        (my_rows['frame'][-1]-my_rows['frame'][0])/my_fps,
                        my_fps),
            print

        if show_observations:
            obs_rows = ca.load_observations( obj_id, data_file)

            if start is not None or stop is not None:
                obs_frames = obs_rows['frame']
                ok1 = obs_frames >= start
                ok2 = obs_frames <= stop
                ok = ok1 & ok2
                obs_rows = obs_rows[ok]

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

        rows = ca.load_data( obj_id, data_file,
                             use_kalman_smoothing=use_kalman_smoothing,
                             kalman_dynamic_model = dynamic_model,
                             frames_per_second=fps)

        if len(rows):
            frames = rows['frame']
            n_frames = rows['frame'][-1]-rows['frame'][0]+1
        else:
            n_frames = 0

        if n_frames < int(min_length):
            continue
        had_any_obj_id_data = True

        if not show_only_track_ends:
            try:
                rows = ca.load_data(obj_id,
                                    data_file,
                                    use_kalman_smoothing=use_kalman_smoothing,
                                    frames_per_second=fps,
                                    kalman_dynamic_model = dynamic_model)
            except Exception, err:
                if 1:
                    raise
                else:
                    print 'ERROR: while processing obj_id %d, skipping this obj_id'%obj_id
                    print err
                    continue

            if start is not None or stop is not None:
                frames = rows['frame']
                ok1 = frames >= start
                ok2 = frames <= stop
                ok = ok1 & ok2
                rows = rows[ok]

            verts = numpy.array( [rows['x'], rows['y'], rows['z']] ).T
            if link_all_simultaneous_objs:
                allsave.append( rows )
            if not obj_color:
                vels = numpy.array( [rows['xvel'], rows['yvel'], rows['zvel']] )#. T
                speeds = numpy.sqrt(numpy.sum(vels**2,axis=0))
        else:
            x0 = rows.field('x')[0]
            x1 = rows.field('x')[-1]

            y0 = rows.field('y')[0]
            y1 = rows.field('y')[-1]

            z0 = rows.field('z')[0]
            z1 = rows.field('z')[-1]

            track_end_verts.append( (x0,y0,z0) )
            track_end_verts.append( (x1,y1,z1) )

        if show_observations:
            if 1:
                # draw lines connecting observation with Kalman point

                line_verts = numpy.concatenate([verts, obs_X])

                line_edges = []
                for i, obs_frame in enumerate(obs_frames):
                    kidx = numpy.nonzero( rows['frame']==obs_frame )[0]
                    assert len(kidx)==1
                    kidx = kidx[0]
                    line_edges.append( [kidx, i+len(verts)] )

                pd = tvtk.PolyData()
                pd.points = line_verts
                pd.lines = line_edges

                pt = tvtk.TubeFilter(radius=0.001,input=pd,
                                     number_of_sides=4,
                                     vary_radius='vary_radius_off',
                                     )
                m = tvtk.PolyDataMapper(input=pt.output)
                a = tvtk.Actor(mapper=m)
                a.property.color = (1,0,0) # red
                a.property.specular = 0.3
                actors.append(a)


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
            if not obj_color:
                pd.point_data.scalars = speeds
#            if numpy.any(speeds>max_vel):
#                print 'WARNING: maximum speed (%.3f m/s) exceeds color map max'%(speeds.max(),)

            g = tvtk.Glyph3D(scale_mode='data_scaling_off',
                             vector_mode = 'use_vector',
                             input=pd)
            ss = tvtk.SphereSource(radius = radius)
            g.source = ss.output
            vel_mapper = tvtk.PolyDataMapper(input=g.output)
            if not obj_color:
                vel_mapper.lookup_table = lut
                vel_mapper.scalar_range = 0.0, float(max_vel)
            a = tvtk.Actor(mapper=vel_mapper)
            if show_observations:
                a.property.opacity = 0.3 # sets transparency/alpha
            if obj_color:
                if obj_id == 158:
                    a.property.color = .9, .8, 0
                elif obj_id == 160:
                    a.property.color = 0, .45, .70
                else:
                    a.property.color = .9, .9, .9
            actors.append(a)
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

    if link_all_simultaneous_objs:
        allsave = numpy.concatenate(allsave)
        allframes = allsave['frame']
        allframes_unique = numpy.unique(allframes)
        link_verts = []
        link_edges = []
        vert_count = 0
        for frameno in allframes_unique:
            this_frame_data = allsave[ allframes == frameno ]
            if len(this_frame_data)<2:
                continue

            start_vert = vert_count
            for this_row in this_frame_data:
                link_verts.append( [this_row['x'], this_row['y'], this_row['z']] )
                vert_count += 1
            end_vert = vert_count
            link_edges.append( list(range(start_vert,end_vert)) )

        pd = tvtk.PolyData()
        pd.points = link_verts
        pd.lines = link_edges

        pt = tvtk.TubeFilter(radius=0.001,input=pd,
                             number_of_sides=4,
                             vary_radius='vary_radius_off',
                             )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = (1,1,1)
        a.property.specular = 0.3
        actors.append(a)

    if not is_mat_file:
        extra['kresults'].close()

    ################################

    if draw_stim_func_str:
        PluginFunc = plugin_loader(draw_stim_func_str)
        stim_actors = PluginFunc(filename=filename)
        actors.extend( stim_actors )

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

    if 1:
        # Inspired by pyface.tvtk.decorated_scene
        marker = tvtk.OrientationMarkerWidget()

        axes = tvtk.AxesActor()
        axes.set(
            normalized_tip_length=(0.4, 0.4, 0.4),
            normalized_shaft_length=(0.6, 0.6, 0.6),
            shaft_type='cylinder'
            )

        p = axes.x_axis_caption_actor2d.caption_text_property
        axes.y_axis_caption_actor2d.caption_text_property = p
        axes.z_axis_caption_actor2d.caption_text_property = p

        p.color = 0.0, 0.0, 0.0 # black

        marker.orientation_marker = axes
        marker.interactor = rwi
        marker.enabled = True

    if not show_only_track_ends and not obj_color and had_any_obj_id_data:

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

            if 1:
                imf = tvtk.WindowToImageFilter(input=rw, input_buffer_type='rgba' )
                writer = tvtk.PNGWriter()

                imf.update()
                imf.modified()
                writer.input = imf.output
                fname = 'kdviewer_output.png'
                writer.file_name = fname
                writer.write()

        picker.add_observer('EndPickEvent', annotatePick)
        rwi.picker = picker

    if not save_still:
        rwi.start()
        print_cam_props( ren.active_camera )
    else:
        imf = tvtk.WindowToImageFilter(input=rw, input_buffer_type='rgba' )
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

    parser.add_option("--start", dest='start',
                      type="int",
                      help="first frame to plot",
                      metavar="START")

    parser.add_option("--stop", dest='stop',
                      type="int",
                      help="last frame to plot",
                      metavar="STOP")

    parser.add_option("--obj-start", dest='obj_start',
                      type="int",
                      help="first object ID to plot",
                      metavar="OBJSTART")

    parser.add_option("--obj-stop", dest='obj_stop',
                      type="int",
                      help="last object ID to plot",
                      metavar="OBJSTOP")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")

    parser.add_option("--draw-stim",
                      type="string",
                      dest="draw_stim_func_str",
                      default="default",
                      help="possible values: %s"%str(plugin_loader.all_names),
                      )

    parser.add_option("--dynamic-model",
                      type="string",
                      dest="dynamic_model",
                      default=None,
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

    parser.add_option("--stereo", action='store_true',dest='stereo',
                      help="display in stereo (red-blue analglyphic)",
                      default=False)

    parser.add_option("--show-cameras", action='store_true',dest='show_cameras',
                      help="show camera locations/frustums",
                      default=False)

    parser.add_option("--fps", dest='fps', type='float',
                      default = None,
                      help="frames per second (used for Kalman filtering/smoothing)")

    parser.add_option("--show-saccade-times", action='store_true',dest='show_saccade_times',
                      help="show saccade times")

    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--vertical-scale", action='store_true',dest='vertical_scale',
                      help="scale bar has vertical orientation")

    parser.add_option("--save-still", action='store_true',dest='save_still',
                      help="save still image as kdviewer_output.png")

    parser.add_option("--obj-color", action='store_true',default=False,dest='obj_color')

    parser.add_option("--link-all-simultaneous-objs", action='store_true', default=False,
                      dest='link_all_simultaneous_objs')

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
        options.obj_only = core_analysis.parse_seq(options.obj_only)
        ## options.obj_only = options.obj_only.replace(',',' ')
        ## seq = map(int,options.obj_only.split())
        ## options.obj_only = seq

        if options.obj_start is not None or options.obj_stop is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    doit(filename=h5_filename,
         start=options.start,
         stop=options.stop,
         obj_start=options.obj_start,
         obj_end=options.obj_stop,
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
         fps = options.fps,
         vertical_scale = options.vertical_scale,
         max_vel = options.max_vel,
         show_only_track_ends = options.show_only_track_ends,
         save_still = options.save_still,
         exclude_vel_mps = options.exclude_vel_mps,
         stereo = options.stereo,
         dynamic_model = options.dynamic_model,
         show_cameras = options.show_cameras,
         obj_color = options.obj_color,
         link_all_simultaneous_objs=options.link_all_simultaneous_objs,
         )

if __name__=='__main__':
    main()

