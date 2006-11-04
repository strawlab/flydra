from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sets
from vtk.util.colors import tomato, banana, azure, blue, \
     black, red, green, white, yellow, lime_green, cerulean, \
     light_grey, dark_orange, brown, light_beige

import vtk_results
import vtk.util.colors as colors
from vtkpython import vtk
import sys

obj_id2color = {140:blue,
                141:red,
                142:green,
                143:brown}

plot_scale = 1000.0 # plot in mm

def show_vtk(filename,max_err=10.0,fstart=None,fend=None):
    kresults = PT.openFile(filename,mode="r")

    n_viewports=1
    renWin, renderers = vtk_results.init_vtk(n_viewports=n_viewports)#stereo=True)
    save_to_file = False
    
    fname_base = 'frame%06d.png'
    #fname_base = 'frame%06d.jpg'
    
    if save_to_file:
        saved_frame_no = 0
        imf = vtk.vtkWindowToImageFilter()
        imf.SetInput(renWin)
        imf.Update()
        
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

    if 1:
        if 1:
            camera = renderers[0].GetActiveCamera()
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (346.5232502590427, 144.88887584454753, 179.14079082195457)
            camera.SetPosition (668.61852513538372, 194.70306493106423, 1129.5319308266744)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (-0.0019776273589632333, 0.99866204591539953, -0.051674046078254918)
            camera.SetClippingRange (356.86986913737519, 1773.1639083658979)
            camera.SetParallelScale(319.400653668)
        if n_viewports>1:
            camera = renderers[1].GetActiveCamera()

            camera.SetParallelProjection(0)
            camera.SetFocalPoint (347.01683877349728, -9.1367671986396708, 146.82282357254343)
            camera.SetPosition (298.86131996724845, -682.62990796070983, 82.970308684030073)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (0.022894611662144959, -0.095982266512180531, 0.99511971203068039)
            camera.SetClippingRange (443.77118691015818, 1357.5583186758349)
            camera.SetParallelScale(319.400653668)
            
    frames = kresults.root.kalman_estimates.read(field='frame',flavor='numpy')
    unique_frames = numpy.array( list(sets.Set(frames)) ) # find unique
    if 0:
        for frame in unique_frames:
            if frame < 14700:
                continue
            row_idxs = numpy.nonzero( frames == int(frame) )[0]
            this_len = len(row_idxs)

            obj_ids = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='obj_id',flavor='numpy')
            xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
            ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
            zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')

            verts = numpy.vstack((xs,ys,zs)).T * plot_scale

            for obj_id,vert in zip(obj_ids,verts):
                vtk_results.show_spheres(renderers,
                                         sphere_coords=[vert],
                                         sphere_radius=2.0,
                                         sphere_color=obj_id2color[obj_id],
                                         theta_resolution=8,
                                         phi_resolution=8,
                                         )
            renWin.Render()

            if save_to_file:

                if fname_base.endswith('.png'):
                    writer = vtk.vtkPNGWriter()
                elif fname_base.endswith('.jpg'):
                    writer = vtk.vtkJPEGWriter()
                    writer.SetQuality(100) # max
                else:
                    raise ValueError("don't know format")
                saved_frame_no += 1
                fname = fname_base%saved_frame_no

                #writer.SetInput(imf.GetOutput())

                imf.Modified()
                writer.SetInput(imf.GetOutput())
                print 'saving',fname
                writer.SetFileName(fname)
                writer.Write()
    else:
        frames = kresults.root.kalman_estimates.read(field='frame',flavor='numpy')
        obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
        for obj_id in range( obj_ids.max()+1 ):
            row_idxs = numpy.nonzero( obj_ids == obj_id )[0]
            this_len = len(row_idxs)
##            if this_len < 10:
##                print 'obj_id %d: %d frames, skipping'%(obj_id,this_len,)
##                continue
            frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='frame',flavor='numpy')
            xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
            ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
            zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')

            verts = numpy.vstack((xs,ys,zs)).T * plot_scale
            for idx in range(len(frames)):
                frame = frames[idx]
                cond = frames==frame
                condi = numpy.nonzero(cond)[0]
                if len(condi)>1:
                    #print 'hmm'
                    #print frames[condi]
                    vtk_results.show_spheres(renderers,
                                             sphere_coords=verts[condi],
                                             sphere_radius=12.0,
                                             sphere_color=obj_id2color.get(obj_id,lime_green),
                                             theta_resolution=8,
                                             phi_resolution=8,
                                             )
##            vtk_results.show_longline(renderers,verts,
##                                      radius=0.001*plot_scale,
##                                      nsides=3,opacity=0.2,
##                                      color=obj_id2color.get(obj_id,lime_green))
            vtk_results.show_spheres(renderers,
                                     sphere_coords=verts,
                                     sphere_radius=2.0,
                                     sphere_color=obj_id2color.get(obj_id,lime_green),
                                     theta_resolution=8,
                                     phi_resolution=8,
                                     )
    kresults.close()

    vtk_results.interact_with_renWin(renWin)

    print 'cam 0'
    camera = renderers[0].GetActiveCamera()
    vtk_results.print_cam_props(camera)
    if n_viewports>1:
        print 'cam 1'
        camera = renderers[1].GetActiveCamera()
        vtk_results.print_cam_props(camera)
    
if __name__=='__main__':
    if 1:
        fstart, fend = None, None
        filename = sys.argv[1]
    show_vtk(filename=filename,fstart=fstart,fend=fend)
