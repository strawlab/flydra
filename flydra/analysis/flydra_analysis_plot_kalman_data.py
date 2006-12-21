from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sets

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

def show_vtk(filename,
             show_obj_ids=False,
             obj_start=None,
             obj_end=None,
             show_cameras=False,
             show_observations=False,
             min_length=None,
             stereo=False,
             debug=0):
    kresults = PT.openFile(filename,mode="r")

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

    DEBUG1=False
    #DEBUG1=True
    if DEBUG1:
        neg_obj_ids = []
            
    obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
    use_obj_ids = obj_ids
    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_end is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]
    # find unique obj_ids:
    use_obj_ids = numpy.array(list(sets.Set([int(obj_id) for obj_id in use_obj_ids])))

    if debug >=5:
        print 'DEBUG: obj_ids.min()',obj_ids.min()
        print 'DEBUG: obj_ids.max()',obj_ids.max()
    for obj_id_enum,obj_id in enumerate(use_obj_ids):
        
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
        estimate_frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
        valid_condition = estimate_frames <= max_observation_frame
        row_idxs = row_idxs[valid_condition]
        n_observations = len( observation_frames )
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
            vtk_results.show_spheres(renderers,obs_verts,
                                     #radius=0.001*plot_scale,
                                     #opacity=0.2,
                                     #color=colors.blue,
                                     )
        
        xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
        ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
        zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')
        
        verts = numpy.vstack((xs,ys,zs)).T * plot_scale

        

##        if DEBUG1:
##            if obj_id>320:
##                break
##            cond = ((xs < 0) & (zs < 0))
##            nz = numpy.nonzero(cond)[0]
##            if len(nz)>1:
##                frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,
##                                                                        field='frame',
##                                                                        flavor='numpy')
##                neg_obj_ids.append(obj_id)
##                vd = verts[1:]-verts[:-1]
##                vd = numpy.sum((vd**2),axis=1)
##                min_vd_idx = numpy.argmin(vd)
##                if vd[min_vd_idx] == 0.0:
                    
##                    print 'obj_id',obj_id
##                    for row in kresults.root.kalman_observations.where(
##                        kresults.root.kalman_observations.cols.obj_id==obj_id):
##                        print 'observations row:',row
##                    print
##                    for row in kresults.root.kalman_estimates.where(
##                        kresults.root.kalman_estimates.cols.obj_id==obj_id):
##                        print 'estimates row:',row['frame'],row['x'],row['y'],row['z']
                        
##                    print 'frames',frames[0],frames[-1]
##                    frame0 = frames[min_vd_idx]
##                    frame1 = frames[min_vd_idx+1]
##                    print '  index',min_vd_idx
##                    print 'vert, frame0 (scaled)',verts[min_vd_idx], frame0
##                    print 'vert, frame1 (scaled)',verts[min_vd_idx+1], frame1
                    
##                    for frame in [frame0,frame1]:
##                        frame = int(frame)
##                        for row in kresults.root.kalman_estimates.where(
##                            kresults.root.kalman_estimates.cols.frame==frame):
##                            if row['obj_id']==obj_id:
##                                print '  kalman_estimates row:',row
##                        for row in kresults.root.kalman_observations.where(
##                            kresults.root.kalman_observations.cols.frame==frame):
##                            if row['obj_id']==obj_id:
##                                print '  kalman_observations row:',row
##                        if 1:
##                            for row in kresults.root.data2d_distorted.where(
##                                kresults.root.data2d_distorted.cols.frame==frame):
##                                        print '  ',row['camn'],row['frame'],row['timestamp'],row['x'],row['y']
##                        print
                    
##                    #print
##                #print verts
##                #print frames
##                print


##                vtk_results.show_longline(renderers,verts,
##                                          radius=0.001*plot_scale,
##                                          nsides=4,opacity=0.2,
##                                          color=colors.blue)

##        if not DEBUG1:
        if 1:
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
                vtk_results.show_longline(renderers,verts,
                                          start_label=start_label,
                                          radius=0.003*plot_scale,
                                          nsides=3,opacity=0.5,
                                          color=color)#s.blue)

    if show_cameras:
        vtk_results.show_cameras(kresults,renderers)

    kresults.close()
    
    if DEBUG1:
        print 'neg_obj_ids = ',repr(neg_obj_ids)

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

    vtk_results.interact_with_renWin(renWin)
    
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
    
    parser.add_option("--min-length", type="int",
                      help="minimum number of observations required to plot",
                      dest="min_length",
                      default=10,
                      metavar="MIN_LENGTH")
    
    parser.add_option("--stereo", action='store_true',dest='stereo',
                      help="display in anaglyphic stereo")

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

    show_vtk(filename=h5_filename,
             obj_start=options.start,
             obj_end=options.stop,
             stereo=options.stereo,
             show_obj_ids = options.show_obj_ids,
             show_observations = options.show_observations,
             show_cameras = options.show_cameras,
             min_length = options.min_length,
             debug = options.debug,
             )
    
if __name__=='__main__':
    main()
