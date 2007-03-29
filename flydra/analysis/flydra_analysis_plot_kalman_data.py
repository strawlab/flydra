from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import sets
import flydra.reconstruct
import reconstruct_orientation

import vtk_results
import vtk.util.colors as colors
try:
    from vtkpython import vtk
except:
    print 'WARNING: hack to update vtk to version in Ubuntu Edgy'
    import vtk
import sys
from optparse import OptionParser

plot_scale = 1000.0 # plot in mm

def get_top_sequences(objid_by_n_observations,n_top_traces = 20):
    n_observations_list = objid_by_n_observations.keys()
    n_observations_list.sort()
    long_obs = []
    long_obs_n = []
    while len(long_obs) < n_top_traces:
        if not len(n_observations_list):
            break
        n_observations = n_observations_list.pop()
        long_obj_ids = objid_by_n_observations[n_observations]
        
        long_obs.extend( long_obj_ids )
        long_obs_n.extend( [n_observations]*len(long_obj_ids) )
    long_obs = long_obs[:n_top_traces]
    long_obs_n = long_obs_n[:n_top_traces]
    return long_obs, long_obs_n
    print 'longest obj_ids:',long_obs
    print 'longest obj_ids (number):',long_obs_n

def show_vtk(filename,
             show_obj_ids=False,
             obj_start=None,
             obj_end=None,
             obj_only=None,
             show_cameras=False,
             show_orientation=False,
             show_observations=False,
             min_length=None,
             stereo=False,
             debug=0,
             show_n_longest=None,
             radius=0.001, # in meters
             ):
    actor2objid = {}

    kresults = PT.openFile(filename,mode="r")
    if show_orientation:
        reconstructor = flydra.reconstruct.Reconstructor(kresults)
        recon2 = reconstructor.get_scaled( reconstructor.scale_factor)
        
        body_line_points = vtk.vtkPoints()
        body_lines = vtk.vtkCellArray()
        body_point_num = 0
        
    color_cycle = ['tomato', 'banana', 'azure', 'blue',
                   'black', 'red', 'green', 'white', 'yellow', 'lime_green', 'cerulean',
                   'light_grey', 'dark_orange', 'brown', 'light_beige']
    
    renWin, renderers = vtk_results.init_vtk(stereo=stereo)
    if 1:
        camera = renderers[0].GetActiveCamera()

        if 0:
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (435.71871180094649, 122.4090122591231, -278.64208334323968)
            camera.SetPosition (562.75100798895119, 3693.498202839558, 1113.7885756800238)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (0.062595103756296219, -0.36449830309814407, 0.92909786353446755)
            camera.SetClippingRange (2689.7556000877539, 5464.887997721311)
            camera.SetParallelScale(319.400653668)

        if 1:

            camera.SetParallelProjection(0)
            camera.SetFocalPoint (1711.1726091248922, 703.81692844931104, 1375.5848044671152)
            camera.SetPosition (275.55235723413432, -4773.4823848768538, 2985.0885689508045)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (0.10761558045997606, 0.25422425708086804, 0.96113938320825409)
            camera.SetClippingRange (3271.2791970812791, 9713.790822523013)
            camera.SetParallelScale(319.400653668)

    obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
    use_obj_ids = obj_ids
    
    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_end is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]

    if obj_only is not None:
        use_obj_ids = numpy.array(obj_only)
        
    # find unique obj_ids:
    use_obj_ids = numpy.array(list(sets.Set([int(obj_id) for obj_id in use_obj_ids])))

    if debug >=1:
        print 'DEBUG: obj_ids.min()',obj_ids.min()
        print 'DEBUG: obj_ids.max()',obj_ids.max()

    ####### iterate once to find longest observations #########
    if show_n_longest is not None:
        objid_by_n_observations = {}
        for obj_id_enum,obj_id in enumerate(use_obj_ids):
            if obj_id_enum%100==0:
                print 'reading %d of %d'%(obj_id_enum,len(use_obj_ids))
            if PT.__version__ <= '1.3.3':
                obj_id_find=int(obj_id)
            else:
                obj_id_find=obj_id

            observation_frame_idxs = kresults.root.kalman_observations.getWhereList(
                kresults.root.kalman_observations.cols.obj_id==obj_id_find,
                flavor='numpy')
            observation_frames = kresults.root.kalman_observations.readCoordinates(
                observation_frame_idxs,
                field='frame',
                flavor='numpy')

            n_observations = len( observation_frames )

            if debug >=3:
                print 'DEBUG: obj_id %d: %d observations'%(obj_id, n_observations)
            objid_by_n_observations.setdefault(n_observations,[]).append(obj_id)
        long_obs, long_obs_n = get_top_sequences(objid_by_n_observations,
                                                 n_top_traces=show_n_longest)
        for obj_id, n_observations in zip( long_obs, long_obs_n ):
            print 'showing obj_id %d (%d observations)'%(obj_id, n_observations)
        print 'longest obj_ids:',long_obs

        use_obj_ids = long_obs
        
    ####### iterate again for plotting ########
    objid_by_n_observations = {}
    for obj_id_enum,obj_id in enumerate(use_obj_ids):
        print obj_id_enum,'obj_id',obj_id
        if obj_id_enum%100==0:
            print 'reading %d of %d'%(obj_id_enum,len(use_obj_ids))
        
        if PT.__version__ <= '1.3.3':
            obj_id_find=int(obj_id)
        else:
            obj_id_find=obj_id

        observation_frame_idxs = kresults.root.kalman_observations.getWhereList(
            kresults.root.kalman_observations.cols.obj_id==obj_id_find,
            flavor='numpy')
        observation_frames = kresults.root.kalman_observations.readCoordinates(
            observation_frame_idxs,
            field='frame',
            flavor='numpy')
        max_observation_frame=observation_frames.max()

        row_idxs = numpy.nonzero( obj_ids == obj_id )[0]
        estimate_frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='frame',flavor='numpy')
        valid_condition = estimate_frames <= max_observation_frame
        row_idxs = row_idxs[valid_condition]
        n_observations = len( observation_frames )
        
        if debug >=3:
            print 'DEBUG: obj_id %d: %d/%d observations'%(obj_id, n_observations,len(estimate_frames))
        objid_by_n_observations.setdefault(n_observations,[]).append(obj_id)
            
        
#        this_len = len(row_idxs)
#        if this_len < min_length:
        if n_observations < min_length:
#            print 'obj_id %d: %d frames, skipping'%(obj_id,this_len,)
            print 'obj_id %d: %d observation frames, skipping'%(obj_id,n_observations,)
            continue

        if show_observations:
            obs_xs = kresults.root.kalman_observations.readCoordinates(
                observation_frame_idxs,field='x',flavor='numpy')
            obs_ys = kresults.root.kalman_observations.readCoordinates(
                observation_frame_idxs,field='y',flavor='numpy')
            obs_zs = kresults.root.kalman_observations.readCoordinates(
                observation_frame_idxs,field='z',flavor='numpy')
            obs_verts = numpy.vstack((obs_xs,obs_ys,obs_zs)).T * plot_scale
            actors = vtk_results.show_spheres(renderers,obs_verts,
                                              #radius=0.001*plot_scale,
                                              #opacity=0.2,
                                              #color=colors.blue,
                                              )
            for actor in actors:
                actor2objid[actor] = obj_id
                
                    
        xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
        ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
        zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')
        
        verts = numpy.vstack((xs,ys,zs)).T

        if len(verts):
            if show_obj_ids:
                start_label = '%d (%d-%d)'%(obj_id,
                                            observation_frames[0],
                                            observation_frames[-1],
                                            )
            else:
                start_label = None


            color_idx = obj_id_enum%len(color_cycle)
            color_name = color_cycle[color_idx]
            color = getattr(colors,color_name)
            actors = vtk_results.show_longline(renderers,verts*plot_scale,
                                               start_label=start_label,
                                               radius=radius*plot_scale,
                                               nsides=3,opacity=0.5,
                                               color=color)#s.blue)
            for actor in actors:
                actor2objid[actor] = obj_id

            if show_orientation:
                frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='frame',flavor='numpy')
                #print 'CALLING for ',obj_id_find
                by_frame = reconstruct_orientation.reconstruct_line_3ds( kresults, recon2, obj_id_find)
                for frame,X in zip(frames,verts):
                    frame = int(frame)
                    if frame not in by_frame:
                        continue
                    line3d = numpy.array( by_frame[frame], dtype=numpy.float)
                    
                    #print 'frame, X, line3d',frame, X, line3d
                    if numpy.any(numpy.isnan(line3d)):
                        #print 'nan -> skip'
                        continue
                    #print line3d.shape
                    line3d.shape = (1,6)

                    
                    L = line3d
                    #L = line3d[numpy.newaxis,:] # Plucker coordinates
                    #print frame,'L',L
                    U = flydra.reconstruct.line_direction(line3d)
                    
                    tube_length = 0.004 # meters ( 4 mm)
                    #orientation_corrected = False
                    orientation_corrected = True
                    
                    if orientation_corrected:
                        pt1 = X-tube_length*U
                        pt2 = X
                    else:
                        pt1 = X-tube_length*.5*U
                        pt2 = X+tube_length*.5*U

                    pt1 = pt1*plot_scale
                    pt2 = pt2*plot_scale
                    #print 'pt1 pt2',pt1, pt2

                    body_line_points.InsertNextPoint(*pt1)
                    body_point_num += 1
                    body_line_points.InsertNextPoint(*pt2)
                    body_point_num += 1

                    body_lines.InsertNextCell(2)
                    body_lines.InsertCellPoint(body_point_num-2)
                    body_lines.InsertCellPoint(body_point_num-1)

    if show_orientation:        
        profileData = vtk.vtkPolyData()
        profileData.SetPoints(body_line_points)
        profileData.SetLines(body_lines)
        
        if 1:
            if 1:
                # Add thickness to the resulting line.
                profileTubes = vtk.vtkTubeFilter()
                profileTubes.SetNumberOfSides(8)
                profileTubes.SetInput(profileData)
                profileTubes.SetRadius(.2)
                #profileTubes.SetRadius(.8)

                profileMapper = vtk.vtkPolyDataMapper()
                profileMapper.SetInput(profileTubes.GetOutput())

                profile = vtk.vtkActor()
                profile.SetMapper(profileMapper)
                profile.GetProperty().SetDiffuseColor( colors.black ) #0xd6/255.0, 0xec/255.0, 0x1c/255.0)
                #profile.GetProperty().SetDiffuseColor(cerulean)
                #profile.GetProperty().SetDiffuseColor(banana)
                profile.GetProperty().SetSpecular(.3)
                profile.GetProperty().SetSpecularPower(30)

                for renderer in renderers:
                    renderer.AddActor( profile )

    if 1:
        if 0:
            yplus = [( 458.4, 257.5, 203.8), # 3d location
                     ( 461.2, 269.0,-28.3)]
            yminus = [( 460.1, 154.9, 193.7),
                      ( 461.4, 166.0,-35.8)]
            all_verts = [yplus, yminus]
        else:
            post = [( 456.7, 202.9, 195.8),
                    
                    ( 458.1, 216.6,-32.9)]
            all_verts = [post]
        print 'warning: post location hardcoded in file'
        
        for verts in all_verts:
            verts = numpy.asarray(verts)
            #verts = verts*plot_scale
            actors = vtk_results.show_longline(renderers,verts,
                                               #start_label=start_label,
                                               radius=0.008*plot_scale,
                                               nsides=8,
                                               color=colors.black)
                
    if show_n_longest is None:
        long_obs, long_obs_n = get_top_sequences(objid_by_n_observations)
        print 'longest obj_ids:',long_obs
        print 'longest obj_ids (number):',long_obs_n
    
    if show_cameras:
        vtk_results.show_cameras(kresults,renderers)

##    if show_orientation:
##        reconstructor = flydra.reconstruct.Reconstructor(kresults)
##        recon2 = reconstructor.get_scaled( reconstructor.scale_factor)
        
            

    kresults.close()
    
    if 1:
        bbox_points = vtk.vtkPoints()
        X_zero_frame = None
        if X_zero_frame is None:
            bbox_points.InsertNextPoint(0,0,0)
            bbox_points.InsertNextPoint(1000,300,300)
        else:
            bbox_points.InsertNextPoint(-150,-150,-150)
            bbox_points.InsertNextPoint(150,150,150)
        bbox_poly_data = vtk.vtkPolyData()
        bbox_poly_data.SetPoints(bbox_points)
        bbox_mapper = vtk.vtkPolyDataMapper()
        bbox_mapper.SetInput(bbox_poly_data)
        bbox=vtk.vtkActor()
        bbox.SetMapper(bbox_mapper)
        
        for renderer in renderers:
            axes2 = vtk.vtkCubeAxesActor2D()
            axes2.SetCamera(renderer.GetActiveCamera())
            axes2.SetProp(bbox)
            axes2.GetProperty().SetColor(0,0,0)
            renderer.AddActor(axes2)

    use_picker = True
    if use_picker:
        # inspired by Annotation/Python/annotatePick.py
        picker = vtk.vtkCellPicker()
        #print 'picker.GetTolerance()',picker.GetTolerance()
        picker.SetTolerance(1e-9)
        
        def annotatePick(object, event):
            if not picker.GetCellId() < 0:
                found = sets.Set([])
                actors = picker.GetActors()
                actors.InitTraversal()
                actor = actors.GetNextItem()
                while actor:
                    objid = actor2objid[actor]
                    found.add(objid)
                    actor = actors.GetNextItem()
                found = list(found)
                found.sort()
                for f in found:
                    print f
                    
        picker.AddObserver("EndPickEvent", annotatePick)
    else:
        picker=None

    renWin.SetMultiSamples(32)
    renWin.SetPolygonSmoothing(1)

    vtk_results.interact_with_renWin(renWin,picker=picker)
    
    camera = renderers[0].GetActiveCamera()
    vtk_results.print_cam_props(camera)

def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

    parser.add_option("--debug", type="int",
                      help="debug level",
                      metavar="DEBUG")
        
    parser.add_option("--start", type="int",
                      help="first object ID to plot",
                      metavar="START")
        
    parser.add_option("--stop", type="int",
                      help="last object ID to plot",
                      metavar="STOP")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")
    
    parser.add_option("--n-top-traces", type="int",
                      help="show N longest traces")
    
    parser.add_option("--min-length", type="int",
                      help="minimum number of observations required to plot",
                      dest="min_length",
                      default=10,
                      metavar="MIN_LENGTH")
    
    parser.add_option("--radius", type="float",
                      help="radius of line (in meters)",
                      default=0.001,
                      metavar="RADIUS")
    
    parser.add_option("--stereo", action='store_true',dest='stereo',
                      help="display in anaglyphic stereo")

    parser.add_option("--show-orientation", action='store_true',dest='show_orientation',
                      help="show orientation of points")

    parser.add_option("--show-obj-ids", action='store_true',dest='show_obj_ids',
                      help="show object ID numbers at start of trajectory")

    parser.add_option("--show-observations", action='store_true',dest='show_observations',
                      help="show observations as spheres")

    parser.add_option("--show-cameras", action='store_true',dest='show_cameras',
                      help="show cameras")

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
        seq = map(int,options.obj_only.split())
        options.obj_only = seq

        if options.start is not None or options.stop is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    show_vtk(filename=h5_filename,
             obj_start=options.start,
             obj_end=options.stop,
             obj_only=options.obj_only,
             show_n_longest=options.n_top_traces,
             show_orientation=options.show_orientation,
             stereo=options.stereo,
             show_obj_ids = options.show_obj_ids,
             show_observations = options.show_observations,
             show_cameras = options.show_cameras,
             min_length = options.min_length,
             debug = options.debug,
             radius = options.radius,
             )
    
if __name__=='__main__':
    main()
