from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime

import vtk_results
import vtk.util.colors as colors
from vtkpython import vtk

plot_scale = 1000.0 # plot in mm

def main(max_err=10.0):
    if 1:
        #filename = 'DATA20060914_181401.tracked_fixed_accel.h5'
        filename = 'DATA20060915_173304.tracked_fixed_accel.h5'
        fstart, fend = None, None
            
    kresults = PT.openFile(filename,mode="r")
    
    renWin, renderers = vtk_results.init_vtk()#stereo=True)
    if 1:
        camera = renderers[0].GetActiveCamera()

        if 1:
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (435.71871180094649, 122.4090122591231, -278.64208334323968)
            camera.SetPosition (562.75100798895119, 3693.498202839558, 1113.7885756800238)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (0.062595103756296219, -0.36449830309814407, 0.92909786353446755)
            camera.SetClippingRange (2689.7556000877539, 5464.887997721311)
            camera.SetParallelScale(319.400653668)

            
    obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
    for obj_id in range( obj_ids.max()+1 ):
##        if not (0 <= obj_id < 400):
##            continue
##        if not (400 <= obj_id < 850):
##            continue
##        if not (850 <= obj_id < 1600):
##            continue
        if not (1600 <= obj_id < 3200):
            continue
            
        row_idxs = numpy.nonzero( obj_ids == obj_id )[0]
        this_len = len(row_idxs)
        if this_len < 10:
            print 'obj_id %d: %d frames, skipping'%(obj_id,this_len,)
            continue
        
        xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
        ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
        zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')

        verts = numpy.vstack((xs,ys,zs)).T * plot_scale

        vtk_results.show_longline(renderers,verts,
                                  radius=0.001*plot_scale,
                                  nsides=3,opacity=0.2,
                                  color=colors.blue)

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
            
    vtk_results.interact_with_renWin(renWin)
    
    camera = renderers[0].GetActiveCamera()
    vtk_results.print_cam_props(camera)
    
if __name__=='__main__':
    main()
