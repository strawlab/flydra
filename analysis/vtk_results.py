#!/usr/bin/env python

import os
os.environ['__GL_FSAA_MODE']='5' # 4x gaussian multisampling on geForce3 linux
opj=os.path.join
from vtkpython import *
from vtk.util.colors import tomato, banana, azure, blue, black
import math

import flydra.reconstruct as reconstruct
import flydra.reconstruct as reconstruct

import numarray as nx
from numarray.ieeespecial import inf

def init_vtk():

    renWin = vtkRenderWindow()

    renderers = []
    for side_view in [True]:
        camera = vtkCamera()
        camera.SetParallelProjection(1)
        if side_view:

            camera.SetParallelProjection(1)
            camera.SetFocalPoint (180.46645292639732, 224.52830505371094, 176.89396047592163)
            camera.SetPosition (44.00298175247319, 530.72541129307763, 395.11644594845245)
            camera.SetViewAngle(30.0)
            camera.SetClippingRange (18.925321490682652, 913.86296121238297)
            camera.SetParallelScale(24.3987052689)

##            camera.SetFocalPoint (148.8128350675106, 243.92594909667969, 173.37159872055054)
##            camera.SetPosition (-196.05324871445703, 54.975538082313001, 246.618622102937)
##            camera.SetViewAngle(30.0)
##            camera.SetClippingRange (9.3450693570905621, 934.50693570905617)
##            camera.SetParallelScale(15.9151865864)
            camera.SetViewUp (0,0,1)


        else:
            camera.SetFocalPoint (52.963163375854492, 117.89408111572266, 37.192019939422607)
            camera.SetPosition (52.963163375854492, 117.89408111572266, 437.19201993942261)
            camera.SetViewUp (0.0, 1.0, 0.0)
            camera.SetParallelScale(230.112510026)
        #camera.SetViewAngle(30.0)
        camera.SetClippingRange (1e-3, 1e6)

        ren1 = vtkRenderer()
        lk = vtkLightKit()
        if side_view:
            ren1.SetViewport(0.0,0,1.0,1.0)
        else:
            ren1.SetViewport(0.9,0.0,1.0,1)
        ren1.SetBackground( 1,1,1)

        ren1.SetActiveCamera( camera )

        renWin.AddRenderer( ren1 )
        renderers.append( ren1 )
        
    renWin.SetSize( 800, 300 )

    return renWin, renderers

##    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
##    iren.Initialize ()
    
##    return renWin, iren, ren1

def show_cameras(results,renderers,frustums=True,labels=True,centers=True):
    import flydra.reconstruct
    R = flydra.reconstruct.Reconstructor(results)
    actors = []
    if centers:
        cam_centers = vtk.vtkPoints()

        for cam_id, pmat in R.Pmat.iteritems():
            X = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            X = X.flat
            cam_centers.InsertNextPoint(*X)

        points_poly_data = vtkPolyData()
        points_poly_data.SetPoints(cam_centers)

        ball = vtk.vtkSphereSource()
        ball.SetRadius(20.0)
        ball.SetThetaResolution(25)
        ball.SetPhiResolution(25)
        balls = vtk.vtkGlyph3D()
        balls.SetInput(points_poly_data)
        balls.SetSource(ball.GetOutput())
        mapBalls = vtkPolyDataMapper()
        mapBalls.SetInput( balls.GetOutput())
        ballActor = vtk.vtkActor()
        ballActor.GetProperty().SetDiffuseColor(azure)
        ballActor.GetProperty().SetSpecular(.3)
        ballActor.GetProperty().SetSpecularPower(30)
        ballActor.SetMapper(mapBalls)

        for renderer in renderers:
            renderer.AddActor( ballActor )
        actors.append( ballActor )

    if frustums:
        line_points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        point_num = 0
        

        for cam_id in R.Pmat.keys():
            pmat = R.get_pmat( cam_id )
            width,height = R.get_resolution( cam_id )

            # cam center
            C = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            C = C.flat

            z = 1
            first_vert = None

            for x,y in ((0,0),(0,height-1),(width-1,height-1),(width-1,0)):
                    x2d = x,y,z
                    X = R.find3d_single_cam(cam_id,x2d) # returns column matrix
                    X = X.flat
                    X = X[:3]/X[3]

                    line_points.InsertNextPoint(*C)
                    point_num += 1

                    U = X-C # direction
                    # rescale to unit length
                    U=U/math.sqrt(U[0]**2 + U[1]**2 + U[2]**2)
                    X = C+500.0*U
                    
                    line_points.InsertNextPoint(*X)
                    point_num += 1

                    if first_vert is None:
                        first_vert = point_num-2
                    else:
                        polys.InsertNextCell(4)
                        polys.InsertCellPoint(point_num-4)
                        polys.InsertCellPoint(point_num-3)
                        polys.InsertCellPoint(point_num-1)
                        polys.InsertCellPoint(point_num-2)
                        
            polys.InsertNextCell(4)
            polys.InsertCellPoint(point_num-2)
            polys.InsertCellPoint(point_num-1)
            polys.InsertCellPoint(first_vert+1)
            polys.InsertCellPoint(first_vert)

        profileData = vtk.vtkPolyData()

        profileData.SetPoints(line_points)
        profileData.SetPolys(polys)

        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInput(profileData)

        profile = vtk.vtkActor()
        profile.SetMapper(profileMapper)
#        profile.GetProperty().SetColor(azure)
        profile.GetProperty().SetOpacity(0.1)
        profile.GetProperty().SetDiffuseColor(tomato)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)

        for renderer in renderers:
            renderer.AddActor( profile )
        actors.append( profile )
    
        
    if labels:
        # labels
        for cam_id, pmat in R.Pmat.iteritems():
            X = reconstruct.pmat2cam_center(pmat) # X is column vector (matrix)
            X = X.flat

            # labels
            textlabel = vtkTextActor()
            textlabel.SetInput( cam_id )
            textlabel.GetPositionCoordinate().SetCoordinateSystemToWorld()
            textlabel.GetPositionCoordinate().SetValue(*X)
            textlabel.SetAlignmentPoint(0)
            textlabel.GetTextProperty().SetColor(0,0,0)
            #textlabel.GetTextProperty().SetJustificationToCentered() # does nothing?
            #print 'textlabel.GetScaledText()',textlabel.GetScaledText()
            for renderer in renderers:
                renderer.AddActor( textlabel )
            actors.append( textlabel )
    return actors

def show_line(renderers,v1,v2,color,radius,nsides=20):
    actors = []
    
##    top3 = [ 139.36847345,  238.72722076,  251.94798316]
##    bottom3 = [ 121.02785563,  237.63751778,  302.77628737]
    
    line_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    line_points.InsertNextPoint(*v1)
    line_points.InsertNextPoint(*v2)
    lines.InsertNextCell(2)
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)


    profileData = vtk.vtkPolyData()
    
    profileData.SetPoints(line_points)
    profileData.SetLines(lines)
    
    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(nsides)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(radius)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInput(profileTubes.GetOutput())
    
    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(color)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)
    
    for renderer in renderers:
        renderer.AddActor( profile )
    actors.append( profile )
    return actors

def show_frames_vtk(results,renderers,
                    f1,f2=None,fstep=None,
                    typ=None,labels=True,
                    show_bounds=False,
                    use_timestamps=False,
                    orientation_corrected=True,
                    max_err=None):
    if typ is None:
        typ = 'best'
        
    if typ == 'fastest':
        data3d = results.root.data3d_fastest
    elif typ == 'best':
        data3d = results.root.data3d_best

    actors = []

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

    frame_nos = range(*seq_args)
    Xs=[]
    line3ds=[]
    for frame_no in frame_nos:
        X = None
        line3d = None
        for row in data3d:
            if row['frame'] != frame_no:
                continue
            X = row['x'],row['y'],row['z']
            if row['p0'] is nan:
                line3d = None
            else:
                line3d = row['p0'],row['p1'],row['p2'],row['p3'],row['p4'],row['p5']
            err = row['mean_dist']
            break
        if X is None:
            print 'WARNING: frame %d not found'%frame_no
        else:
            if max_err is not None:
                if err > max_err:
                    print 'WARNING: frame %d err too large'%frame_no
                    X = None
                    line3d = None
        Xs.append(X)
        line3ds.append(line3d)

    if show_bounds:
        tmp = nx.array(Xs)
        print 'x range:',min( tmp[:,0] ),max( tmp[:,0] )
        print 'y range:',min( tmp[:,1] ),max( tmp[:,1] )
        print 'z range:',min( tmp[:,2] ),max( tmp[:,2] )
    point_num = 0
    xlim = [inf,-inf]
    ylim = [inf,-inf]
    zlim = [inf,-inf]
    for X,line3d in zip(Xs,line3ds):
        if X is not None:
            cog_points.InsertNextPoint(*X)
            # workaround ipython -pylab mode:
            max = sys.modules['__builtin__'].max
            min = sys.modules['__builtin__'].min
            xlim[0] = min( xlim[0], X[0] )
            xlim[1] = max( xlim[1], X[0] )
            ylim[0] = min( ylim[0], X[1] )
            ylim[1] = max( ylim[1], X[1] )
            zlim[0] = min( zlim[0], X[2] )
            zlim[1] = max( zlim[1], X[2] )
    
        if line3d is not None:
            
            L = line3d # Plucker coordinates
            U = reconstruct.line_direction(line3d)
            tube_length = 4
            if orientation_corrected:
                line_points.InsertNextPoint(*(X-tube_length*U))
            else:
                line_points.InsertNextPoint(*(X-tube_length*.5*U))
            point_num += 1
            
            if orientation_corrected:
                line_points.InsertNextPoint(*X)
            else:
                line_points.InsertNextPoint(*(X+tube_length*.5*U))
            #line_points.InsertNextPoint(*(X-tube_length*U))
            point_num += 1

            lines.InsertNextCell(2)
            lines.InsertCellPoint(point_num-2)
            lines.InsertCellPoint(point_num-1)

    # point rendering
    
    points_poly_data = vtkPolyData()
    points_poly_data.SetPoints(cog_points)

    ball = vtk.vtkSphereSource()
    ball.SetRadius(0.5)
    ball.SetThetaResolution(15)
    ball.SetPhiResolution(15)
    balls = vtk.vtkGlyph3D()
    balls.SetInput(points_poly_data)
    balls.SetSource(ball.GetOutput())
    mapBalls = vtkPolyDataMapper()
    mapBalls.SetInput( balls.GetOutput())
    ballActor = vtk.vtkActor()
    ballActor.SetMapper(mapBalls)

    for renderer in renderers:
        renderer.AddActor( ballActor )
    actors.append( ballActor )

    # line rendering 
    # ( see VTK demo Rendering/Python/CSpline.py )
    
    profileData = vtk.vtkPolyData()
    
    profileData.SetPoints(line_points)
    profileData.SetLines(lines)
    
    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(8)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(0.15)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInput(profileTubes.GetOutput())
    
    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(banana)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)

    for renderer in renderers:
        renderer.AddActor( profile )
    actors.append( profile )

    # bounding box
    bbox_points = vtkPoints()
    if 1:
        bbox_points.InsertNextPoint( xlim[0], ylim[0], zlim[0] )
        bbox_points.InsertNextPoint( xlim[1], ylim[1], zlim[1] )
    else:
        bbox_points.InsertNextPoint(-100,-75,-150)
        bbox_points.InsertNextPoint(250,350,100)
    bbox_poly_data = vtkPolyData()
    bbox_poly_data.SetPoints(bbox_points)
    bbox_mapper = vtk.vtkPolyDataMapper()
    bbox_mapper.SetInput(bbox_poly_data)
    bbox=vtk.vtkActor()
    bbox.SetMapper(bbox_mapper)
    # (don't render)

    if labels:
        for frame_no, X in zip(frame_nos,Xs):
            if X is None:
                continue
            if use_timestamps:
                if (frame_no-f1)%10 != 0:
                    continue
                label = str((frame_no-f1)/100.0)
            else:
                if frame_no%50 != 0:
                    continue
                label = str(frame_no)
##            X = X.flat
            # labels
            tl = vtkTextActor()
            tl.SetInput( label )
            tl.GetPositionCoordinate().SetCoordinateSystemToWorld()
            tl.GetPositionCoordinate().SetValue(*X)
            tl.SetAlignmentPoint(0)
            tl.GetTextProperty().SetColor(0,0,0)
            tl.GetTextProperty().SetJustificationToCentered() # does nothing?
            for renderer in renderers:
                renderer.AddActor( tl )
            actors.append( tl )


    if 1:
    #if (ren1 is not None) and (actor is not None):
        # from Annotation/Python/cubeAxes.py
        tprop = vtk.vtkTextProperty()
        tprop.SetColor(0,0,0)
        #tprop.ShadowOn()

        for renderer in renderers:
            axes2 = vtk.vtkCubeAxesActor2D()
            axes2.SetProp(bbox)
            axes2.SetCamera(renderer.GetActiveCamera())
            axes2.SetLabelFormat("%6.4g")
            axes2.SetFlyModeToOuterEdges()
            #axes2.SetFlyModeToClosestTriad()
            axes2.SetFontFactor(0.8)
            axes2.ScalingOff()
            axes2.SetAxisTitleTextProperty(tprop)
            axes2.SetAxisLabelTextProperty(tprop)
            axes2.GetProperty().SetColor(0,0,0)
            renderer.AddActor(axes2)
            actors.append( axes2 )
        #renderer.AddProp(axes2)
        
##    return bbox
    return actors
    
def interact_with_renWin(renWin, ren1=None, actor=None):
##def interact_with_renWin(renWin, iren, ren1=None, actor=None):

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow( renWin )

    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.Initialize ()
    
    renWin.Render()
    
    iren.Start()
    
def print_cam_props(camera):
    print 'camera.SetParallelProjection(%s)'%str(camera.GetParallelProjection())
    print 'camera.SetFocalPoint',camera.GetFocalPoint()
    print 'camera.SetPosition',camera.GetPosition()        
    print 'camera.SetViewAngle(%s)'%str(camera.GetViewAngle())
    print 'camera.SetViewUp',camera.GetViewUp()
    print 'camera.SetClippingRange',camera.GetClippingRange()
    print 'camera.SetParallelScale(%s)'%str(camera.GetParallelScale())
    print
    
if __name__=='__main__':

    results
    
##    import result_browser
##    try:
##        results
##    except NameError:
##        results = result_browser.Results()
    if 1:
        start_frame = 49048
        stop_frame = 49378
##        start_frame = 48989 +400
##        stop_frame = start_frame + 500
        
        renWin, renderers = init_vtk()
        #show_cameras(results,renderers)

        if 1:
            CT=array([ 181.88106377,  221.06126383,  168.28886479])
            CB=array([ 188.25655514,  218.76102605,   30.89531996])
            show_line(renderers,CT,CB,black,4)
        if 0:
            NZ = array([   9.11331261,  117.08933803,   53.84209957])
            NY = array([  10.98416978,  392.19324712,   70.9049832 ])
            show_line(renderers,NZ,NY,blue,1)
            
        show_frames_vtk(results,renderers,start_frame,stop_frame,1,
                        orientation_corrected=True,
                        show_bounds=False,
                        use_timestamps=True,max_err=10)
        for renderer in renderers:
            renderer.ResetCameraClippingRange()
        interact_with_renWin(renWin,renderers)
        for renderer in renderers:
            print_cam_props(renderer.GetActiveCamera())
    else:
        renWin, renderers = init_vtk()
        imf = vtkWindowToImageFilter()
        imf.SetInput(renWin)
        imf.Update()
        
        for i in range(6711,6930,1):
            actors = show_frames_vtk(results,renderers,11938,i,1)
            renWin.Render()
            
            writer = vtk.vtkPNGWriter()
            writer.SetInput(imf.GetOutput())
        
            imf.Modified()

            writer.SetInput(imf.GetOutput())
            fname = 'topvtk%06d.png'%i
            print 'saving',fname
            writer.SetFileName(fname)
            writer.Write()

            for renderer in renderers:
                for actor in actors:
                    renderer.RemoveActor(actor)
                
##            actors = ren1.GetActors()
##            print actors
##            while 1:
##                actor = actors.GetNextActor()
##                if actor is None:
##                    break
##                print actor

##                ren1.RemoveActor(actor)
