#!/usr/bin/env python

import os
os.environ['__GL_FSAA_MODE']='5' # 4x gaussian multisampling on geForce3 linux
opj=os.path.join
from vtkpython import *
from vtk.util.colors import tomato, banana

import numarray as nx

def init_vtk():

    renWin = vtkRenderWindow()
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow( renWin )
    
    camera = vtkCamera()
    camera.SetFocalPoint(119, 16, -59)
    camera.SetPosition(-800, -478, 241)
    camera.SetViewAngle( 30.0 )
    camera.SetViewUp(0,0,1)
    camera.SetClippingRange( 445, 1852)

    ren1 = vtkRenderer()
    lk = vtkLightKit()
    ren1.SetViewport(0.0,0.0,1.0,1.0)
    ren1.SetBackground( 1,1,1)

    ren1.SetActiveCamera( camera )

    renWin.AddRenderer( ren1 )
    renWin.SetSize( 640, 480 )

    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.Initialize ()
    
    return renWin, iren, ren1

def show_frames_vtk(results,renderer,f1,f2=None,fstep=None,typ=None):
    if typ is None:
        typ = 'best'
        
    if typ == 'fast':
        data3d = results.data3d_fast
    elif typ == 'best':
        data3d = results.data3d_best

    # Initialize VTK data structures
    
    cog_points = vtk.vtkPoints() # 'center of gravity'
    line_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Get data from results

    seq_args = [f1]
    if f2 is not None:
        seq_args.append(f2)
        if fstep is not None:
            seq_args.append(fstep)
    elif fstep is not None:
        print 'WARNING: fstep given, but not f2'

    Xs=[]
    line3ds=[]
    for frame_no in range(*seq_args):
        try:
            Xi = nx.nonzero( nx.equal( data3d[:,0], frame_no ))[0][0]
            X = data3d[Xi,1:4]
            line3d = data3d[Xi,4:10]
            if X[2] > 332:
                print 'frame_no',frame_no
            
        except IndexError:
            print 'WARNING: frame %d not found'%frame_no
            X = None
            line3d = None
        Xs.append(X)
        line3ds.append(line3d)

    point_num = 0
    for X,line3d in zip(Xs,line3ds):
        if X is not None:
            cog_points.InsertNextPoint(*X)
    
        if line3d is not None:
            L = line3d # Plucker coordinates
            U = L[:3] # direction of line
            # rescale to unit length
            U=U/math.sqrt(U[0]**2 + U[1]**2 + U[2]**2)
        
            line_points.InsertNextPoint(*(X+20*U))
            point_num += 1
            
            line_points.InsertNextPoint(*(X-20*U))
            point_num += 1

            lines.InsertNextCell(2)
            lines.InsertCellPoint(point_num-2)
            lines.InsertCellPoint(point_num-1)

    # point rendering
    
    points_poly_data = vtkPolyData()
    points_poly_data.SetPoints(cog_points)

    ball = vtk.vtkSphereSource()
    ball.SetRadius(3.0)
    ball.SetThetaResolution(15)
    ball.SetPhiResolution(15)
    balls = vtk.vtkGlyph3D()
    balls.SetInput(points_poly_data)
    balls.SetSource(ball.GetOutput())
    mapBalls = vtkPolyDataMapper()
    mapBalls.SetInput( balls.GetOutput())
    ballActor = vtk.vtkActor()
    ballActor.SetMapper(mapBalls)

    renderer.AddActor( ballActor )

    # line rendering 
    # ( see VTK demo Rendering/Python/CSpline.py )
    
    profileData = vtk.vtkPolyData()
    
    profileData.SetPoints(line_points)
    profileData.SetLines(lines)
    
    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(8)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(1.0)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInput(profileTubes.GetOutput())
    
    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(banana)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)

    renderer.AddActor( profile )
    return profile
    
def interact_with_renWin(renWin, iren, ren1=None, actor=None):
    if (ren1 is not None) and (actor is not None):
        # from Annotation/Python/cubeAxes.py
        tprop = vtk.vtkTextProperty()
        tprop.SetColor(0,0,0)
        #tprop.ShadowOn()

        axes2 = vtk.vtkCubeAxesActor2D()
        axes2.SetProp(actor)
        axes2.SetCamera(ren1.GetActiveCamera())
        axes2.SetLabelFormat("%6.4g")
        axes2.SetFlyModeToOuterEdges()
        #axes2.SetFlyModeToClosestTriad()
        axes2.SetFontFactor(0.8)
        axes2.ScalingOff()
        axes2.SetAxisTitleTextProperty(tprop)
        axes2.SetAxisLabelTextProperty(tprop)
        axes2.GetProperty().SetColor(0,0,0)
        ren1.AddProp(axes2)
    
    renWin.Render()
    
    iren.Start()

