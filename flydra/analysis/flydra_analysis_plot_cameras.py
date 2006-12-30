from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sets

import flydra.reconstruct

import vtk_results
import vtk.util.colors as colors
try:
    from vtkpython import vtk
except:
    import vtk
import sys
from optparse import OptionParser

plot_scale = 1000.0 # plot in mm

def show_vtk(path,
             stereo=False):
    reconstructor = flydra.reconstruct.Reconstructor(path)

    cam_ids = reconstructor.get_cam_ids()
    cam_ids.sort()
    cam_centers = numpy.asarray([reconstructor.get_camera_center(cam_id)[:,0]
                                 for cam_id in cam_ids])
    print 'cam_centers'
    print cam_centers
    
    renWin, renderers = vtk_results.init_vtk(stereo=stereo)
    
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

    reconstructor = flydra.reconstruct.Reconstructor(path)
    vtk_results.show_cameras(reconstructor,renderers)

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
    usage = '%prog PATH [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-p", "--path", dest="path", type='string',
                      help="hdf5 file or calibration directory",
                      metavar="PATH")
    parser.add_option("--stereo", action='store_true',dest='stereo',
                      help="display in anaglyphic stereo")
    
    (options, args) = parser.parse_args()

    if options.path is not None:
        args.append(options.path)
        
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as PATH supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return
        
    path=args[0]

    show_vtk(path=path,
             stereo=options.stereo)
    
if __name__=='__main__':
    main()
