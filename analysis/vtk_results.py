#!/usr/bin/env python

import os
os.environ['__GL_FSAA_MODE']='4' # 4x gaussian multisampling on geForce3 linux
os.environ['__GL_DEFAULT_LOG_ANISO']='3'
opj=os.path.join
from vtkpython import *
from vtk.util.vtkImageImportFromArray import *
from vtk.util.colors import tomato, banana, azure, blue, \
     black, red, green, white, yellow, lime_green, cerulean, \
     light_grey, dark_orange, brown, light_beige
from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray
import math, random

import flydra.reconstruct as reconstruct

import numpy
import numarray as nx
import Numeric # vtkImageImportFromArray needs Numeric
import RandomArray

import cgtypes # tested with 1.2
array = nx.array

def init_vtk(stereo=False):

    renWin = vtkRenderWindow()
    if stereo:
        print 'renWin.GetStereoCapableWindow()',renWin.GetStereoCapableWindow()
        print 'renWin.StereoCapableWindowOn()',renWin.StereoCapableWindowOn()
        print 'renWin.GetStereoCapableWindow()',renWin.GetStereoCapableWindow()
        renWin.StereoRenderOn()
        renWin.SetStereoTypeToRedBlue()
        
    renderers = []
    for side_view in [True]:
        camera = vtkCamera()
        camera.SetParallelProjection(1)
        if side_view:

##            camera.SetParallelProjection(1)
##            camera.SetFocalPoint (72.14166197180748, 303.57498168945312, 39.668309688568115)
##            camera.SetPosition (-225.40723294994916, 191.86856708907058, 282.53932909594647)
##            camera.SetViewAngle(30.0)
##            camera.SetParallelScale(13.7499648363)


##            camera.SetParallelProjection(1)
##            camera.SetFocalPoint (183.28723427103145, 197.35179711218706, 179.54488158097655)
##            camera.SetPosition (103.94839384485147, 1090.2877280742541, 238.19531097666248)
##            camera.SetViewAngle(30.0)
##            camera.SetParallelScale(161)

            camera.SetParallelProjection(1)
            camera.SetFocalPoint (1355.1257952596316, 138.42352926242845, 241.79385230532296 )
            camera.SetPosition (1616.5195986326066, -491.41719908620058, -343.05060371955983 )
            camera.SetViewAngle(30.0)
            camera.SetViewUp (-8.2626711604279806e-05, -0.68046473229527904, 0.73278082758430629)
            camera.SetParallelScale(319.400653668)
            
            if 1:
                camera.SetViewUp (0,0,1)
            else:
                corner = array([  4.91559111,  54.73864537,  32.58650871])
                upc = array([   7.91709368,   64.14688249,  184.80049719])
                h=(upc-corner)
                camera.SetViewUp(*h)
            
        else:
            camera.SetFocalPoint (52.963163375854492, 117.89408111572266, 37.192019939422607)
            camera.SetPosition (52.963163375854492, 117.89408111572266, 437.19201993942261)
            camera.SetViewUp (0.0, 1.0, 0.0)
            camera.SetParallelScale(230.112510026)
        #camera.SetViewAngle(30.0)
        #camera.SetClippingRange (1e-3, 1e6)
        camera.SetClippingRange (1e-2, 1e5)

        ren1 = vtkRenderer()
        lk = vtkLightKit()
        if side_view:
            ren1.SetViewport(0.0,0,1.0,1.0)
        else:
            ren1.SetViewport(0.9,0.0,1.0,1)
        ren1.SetBackground( 1,1,1)
        #ren1.SetBackground( .6,.6,.75)
        #ren1.SetBackground( 0, 0x33/255.0, 0x33/255.0)

        ren1.SetActiveCamera( camera )

        renWin.AddRenderer( ren1 )
        renderers.append( ren1 )
        
    renWin.SetSize( 1024, 768 )
#    renWin.SetSize( 640,480)
#    renWin.SetSize( 320,240)

    return renWin, renderers

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

##def show_numpy_image(renderers,im,shared_vert,vert2,vert3):
##    if 1:
##        return
##    # compute 4th vertex
##    v1v2 = vert2 - shared_vert
##    vert4 = vert3 + v1v2

##    if len(im.shape) == 2:
##        im = nx.reshape( im, (im[0], im[1], 1) ) # returns view if possible
##    im = Numeric.asarray(im)
##    im = im.astype( Numeric.UInt8 )
##    iifa = vtkImageImportFromArray()
##    iifa.SetArray( im )

##    ia = vtk.vtkImageActor()
##    ia.SetInput(iifa.GetOutput())

##    # hmm
##    coords = nx.array([ shared_vert,
##                        vert2,
##                        vert3,
##                        vert4 ])
    
##    ia.SetDisplayExtent( min(coords[:,0]),
##                         max(coords[:,0]),
##                         min(coords[:,1]),
##                         max(coords[:,1]),
##                         min(coords[:,2]),
##                         min(coords[:,2]) )
    
##    # XXX not done
    
##    for renderer in renderers:
##        renderer.AddActor( ia )
        
##    actors = [ia]
    
##    return actors

def show_line(renderers,v1,v2,color,radius,nsides=20,opacity=1.0):
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
    profile.GetProperty().SetOpacity(opacity)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)
    
    for renderer in renderers:
        renderer.AddActor( profile )
    actors.append( profile )
    return actors

def show_texture(renderers,origin,ax_vert1,ax_vert2,shape):

    im = RandomArray.randint(0,2,shape=shape)
    im = (im*255).astype(Numeric.UInt8)
    imagedata  = vtkImageImportFromArray()
    imagedata.SetArray(im)
    atext = vtk.vtkTexture()
    atext.SetInput(imagedata.GetOutput())
    atext.InterpolateOff()

    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(*origin)
    plane.SetPoint1(*ax_vert1)
    plane.SetPoint2(*ax_vert2)
    profileData = plane.GetOutput()
        
    tmapper = vtk.vtkTextureMapToPlane()
    tmapper.SetInput( profileData )
    tmapper.SetOrigin(*origin)
    tmapper.SetPoint1(*ax_vert1)
    tmapper.SetPoint2(*ax_vert2)

    profileMapper = vtk.vtkDataSetMapper()
    profileMapper.SetInput(tmapper.GetOutput())
    
    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.SetTexture(atext)
    
    for renderer in renderers:
        renderer.AddActor( profile )
    return [profile]

def show_spheres(renderers,
                 sphere_coords=None,
                 sphere_radius=2.0,
                 sphere_color=blue,
                 theta_resolution=8,
                 phi_resolution=8,
                 ):
    points = vtk.vtkPoints()
    
    actors = []
    for X in sphere_coords:
        points.InsertNextPoint(*X)

    points_poly_data = vtkPolyData()
    points_poly_data.SetPoints(points)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(sphere_radius)
    sphere.SetThetaResolution(theta_resolution)
    sphere.SetPhiResolution(phi_resolution)

    sphere_glyphs = vtk.vtkGlyph3D()
    sphere_glyphs.SetInput(points_poly_data)
    sphere_glyphs.SetSource(sphere.GetOutput())

    sphere_glyph_mapper = vtkPolyDataMapper()
    sphere_glyph_mapper.SetInput( sphere_glyphs.GetOutput())
    sphereGlyphActor = vtk.vtkActor()
    sphereGlyphActor.SetMapper(sphere_glyph_mapper)
    sphereGlyphActor.GetProperty().SetDiffuseColor(sphere_color)
    sphereGlyphActor.GetProperty().SetSpecular(.3)
    sphereGlyphActor.GetProperty().SetSpecularPower(30)

    for renderer in renderers:
        renderer.AddActor( sphereGlyphActor )
    actors.append( sphereGlyphActor )
    return actors

class NotGiven: pass

def show_frames_vtk(results,renderers,
                    f1,f2=None,fstep=None,
                    typ=None,labels=True,
                    use_timestamps=False,
                    timed_force_table=None,
#                    plot_timed_forces=False,
                    timed_force_scaling_factor=1e6,
                    timed_force_color=red,
                    timed_force_radius=0.15,
                    orientation_corrected=True,
                    render_mode='ball_and_stick', # 'ball_and_stick' or 'triangles'
                    triangle_mode_data=None,
                    triangle_mode_color=red,
                    X_zero_frame=None,
                    bounding_box=True,
                    frame_no_offset=0, # for displaying frame numbers
                    color_change_frame=None,
                    show_warnings=True,
                    max_err=None,
                    # before trigger or all:
                    ball_color1=None,
                    line_color1=NotGiven,
                    # after trigger
                    ball_color2=None,
                    line_color2=NotGiven,
                    show_debug_info=False,
                    ): # only for 'ball_and_stick'

    if ball_color1 is None:
        ball_color1 = red
    if line_color1 == NotGiven:
        line_color1 = ( 0xd6/255.0, 0xec/255.0, 0x1c/255.0)        
    if ball_color2 is None:
        ball_color2 = blue
    if line_color2 == NotGiven:
        line_color2 = banana
    # 'triangles' render mode is always smoothed
    if typ is None:
        typ = 'best'
        
    if typ == 'fastest':
        data3d = results.root.data3d_fastest
    elif typ == 'best':
        data3d = results.root.data3d_best

    if timed_force_table is None:
        plot_timed_forces=False
    else:
        plot_timed_forces=True

    actors = []

    # Initialize VTK data structures
    
    timed_force_line_points = vtk.vtkPoints()

    if render_mode.startswith('ball_and_stick'):
        cog_points = vtk.vtkPoints() # 'center of gravity'
        body_line_points = vtk.vtkPoints()
        body_lines = vtk.vtkCellArray()
        body_point_num = 0
        #         if color_change_frame is not None:
            
        cog_points2 = vtk.vtkPoints() # 'center of gravity'
        body_line_points2 = vtk.vtkPoints()
        body_lines2 = vtk.vtkCellArray()
        body_point_num2 = 0
            
    elif render_mode.startswith('triangles'):
        tri_points = vtk.vtkPoints()
        tri_cells = vtk.vtkCellArray()
        tri_point_num = 0
        
        tri2_points = vtk.vtkPoints()
        tri2_cells = vtk.vtkCellArray()
        tri2_point_num = 0
    
    timed_force_lines = vtk.vtkCellArray()

    timed_force_point_num = 0
    
    unit_y_vector = nx.array([0,1,0],nx.Float)

    # Get data from results

    seq_args = [f1]
    if f2 is not None:
        seq_args.append(f2)
        if fstep is not None:
            seq_args.append(fstep)
    elif fstep is not None:
        print 'WARNING: fstep given, but not f2'

    if not data3d.cols.frame.index:
        data3d.cols.frame.createIndex()

    frame_nos = range(*seq_args)
    if render_mode.startswith('ball_and_stick'):
        Xs=[]
        line3ds=[]
        X_zero = None
        # XXX table must be in order?
        found_frame_nos = []

        for row in data3d.where( frame_nos[0] <= data3d.cols.frame <= frame_nos[-1] ):
            frame_no = row['frame']
            if frame_no not in frame_nos:
                continue
            X = row['x'],row['y'],row['z']
            if numpy.isnan(row['p0']):
                line3d = None
            else:
                line3d = row['p0'],row['p1'],row['p2'],row['p3'],row['p4'],row['p5']
            err = row['mean_dist']
            if max_err is not None:
                if err > max_err:
                    if show_warnings:
                        print 'WARNING: frame %d err too large'%row['frame']
                    X = None
                    line3d = None
            Xs.append(X)
            line3ds.append(line3d)
            found_frame_nos.append( row['frame'] )
            if X_zero_frame == frame_no:
                X_zero = X
        if show_debug_info:
            print 'found %d frames of 3d data'%(len(Xs),)
            #print '  mean position',numpy.mean(numpy.asarray(Xs),axis=0)
                
        if X_zero_frame is not None:
            # remove offset
            if X_zero is None:
                print 'WARNING: wanted X offset for frame, but data does not exist at that frame, skipping'
                return
            try:
                new_Xs = []
                for X in Xs:
                    if X is not None:
                        new_Xs.append( (X[0]-X_zero[0],X[1]-X_zero[1],X[2]-X_zero[2]) )
                    else:
                        new_Xs.append( None )
                Xs = new_Xs
            except:
                print 
                print 'X',X
                print 'X_zero',X_zero
                print
                raise
        for idx, frame_no in enumerate(frame_nos):
            if frame_no not in found_frame_nos:
                # XXX is idx right?
                Xs.insert( idx, None )
                line3ds.insert( idx, None )
                if show_warnings:
                    print 'WARNING: frame %d not found'%frame_no
        orient_infos = line3ds
    elif render_mode.startswith('triangles'):
        Xs=[]
        Qs = []
        for frame_no in frame_nos:
            X = None
            Q = None
            for row in triangle_mode_data:
#            for row in results.root.smooth_data_roll_fixed_lin:
#            for row in results.root.smooth_data_real:
#            for row in results.root.smooth_data:
                if row['frame'] != frame_no:
                    continue
                X = row['x'],row['y'],row['z']
                if not numpy.isnan(row['qw']):
                    Q = cgtypes.quat( row['qw'], row['qx'],
                                      row['qy'], row['qz'] )
                break
            Xs.append( X )
            Qs.append( Q )
        orient_infos = Qs
        
    timed_forces=[]
    fxyz = None
    for frame_no in frame_nos:
        fxyz = None
        if plot_timed_forces:
            for row in timed_force_table:
                if row['frame'] != frame_no:
                    continue
                fxyz = nx.array( (row['fx'], row['fy'], row['fz']) )
        timed_forces.append( fxyz )
    
    ok_Xs = nx.array([ X for X in Xs if X is not None ])
    if len(ok_Xs)==0:
        # no data, return empty list
        return actors
    
    xlim = min( ok_Xs[:,0] ), max( ok_Xs[:,0] )
    ylim = min( ok_Xs[:,1] ), max( ok_Xs[:,1] )
    zlim = min( ok_Xs[:,2] ), max( ok_Xs[:,2] )
        
    if 0:
        print 'x range:',xlim
        print 'y range:',ylim
        print 'z range:',zlim
    
    for frame_no,X,orient_info,timed_force in zip(frame_nos,Xs,orient_infos,timed_forces):
        in_orig_list = color_change_frame is None or (frame_no < color_change_frame)
        if X is not None:
            if render_mode.startswith('ball_and_stick'):
                if in_orig_list:
                    cog_points.InsertNextPoint(*X)
                else:
                    cog_points2.InsertNextPoint(*X)

        if render_mode.startswith('ball_and_stick'):
            line3d = orient_info
            if line3d is not None:

                L = line3d # Plucker coordinates
                U = reconstruct.line_direction(line3d)

                tube_length = 4

                if orientation_corrected:
                    pt1 = X-tube_length*U
                    pt2 = X
                else:
                    pt1 = X-tube_length*.5*U
                    pt2 = X+tube_length*.5*U

                if in_orig_list:
                    body_line_points.InsertNextPoint(*pt1)
                    body_point_num += 1
                    body_line_points.InsertNextPoint(*pt2)
                    body_point_num += 1

                    body_lines.InsertNextCell(2)
                    body_lines.InsertCellPoint(body_point_num-2)
                    body_lines.InsertCellPoint(body_point_num-1)
                else:
                    body_line_points2.InsertNextPoint(*pt1)
                    body_point_num2 += 1
                    body_line_points2.InsertNextPoint(*pt2)
                    body_point_num2 += 1

                    body_lines2.InsertNextCell(2)
                    body_lines2.InsertCellPoint(body_point_num2-2)
                    body_lines2.InsertCellPoint(body_point_num2-1)

        elif render_mode.startswith('triangles'):
            Q = orient_info

            # unit vectors in fly orientation
            fly_unit_x = cgtypes.quat(0, 1, 0, 0)
            fly_unit_y = cgtypes.quat(0, 0, 1, 0)
            fly_unit_z = cgtypes.quat(0, 0, 0, 1)

            def rotate(S3,u):
                V=S3*u*S3.inverse()
                return nx.array((V.x, V.y, V.z))
            
            fly_unit_x_world = rotate(Q,fly_unit_x)
            fly_unit_y_world = rotate(Q,fly_unit_y)
            fly_unit_z_world = rotate(Q,fly_unit_z)
        
            pt1 = X + 2*fly_unit_x_world
            pt2 = X - 2*fly_unit_x_world + 0.5*fly_unit_y_world
            pt3 = X - 2*fly_unit_x_world - 0.5*fly_unit_y_world

            tri_points.InsertNextPoint(*pt1)
            tri_point_num += 1
            tri_points.InsertNextPoint(*pt2)
            tri_point_num += 1
            tri_points.InsertNextPoint(*pt3)
            tri_point_num += 1

            tri_cells.InsertNextCell(3)
            tri_cells.InsertCellPoint(tri_point_num-3)
            tri_cells.InsertCellPoint(tri_point_num-2)
            tri_cells.InsertCellPoint(tri_point_num-1)

            # vertical "fin"
            
            pt1 = X - 1*fly_unit_x_world
            pt2 = X - 2*fly_unit_x_world + 0.5*fly_unit_z_world
            pt3 = X - 2*fly_unit_x_world

            tri2_points.InsertNextPoint(*pt1)
            tri2_point_num += 1
            tri2_points.InsertNextPoint(*pt2)
            tri2_point_num += 1
            tri2_points.InsertNextPoint(*pt3)
            tri2_point_num += 1

            tri2_cells.InsertNextCell(3)
            tri2_cells.InsertCellPoint(tri2_point_num-3)
            tri2_cells.InsertCellPoint(tri2_point_num-2)
            tri2_cells.InsertCellPoint(tri2_point_num-1)

        if plot_timed_forces and X is not None and timed_force is not None:
            pt1 = X+timed_force_scaling_factor*timed_force
            pt2 = X

            timed_force_line_points.InsertNextPoint(*pt1)
            timed_force_point_num += 1
            
            timed_force_line_points.InsertNextPoint(*pt2)
            timed_force_point_num += 1

            timed_force_lines.InsertNextCell(2)
            timed_force_lines.InsertCellPoint(timed_force_point_num-2)
            timed_force_lines.InsertCellPoint(timed_force_point_num-1)

    if render_mode.startswith('ball_and_stick'):
        # head rendering as ball

        for ptnum in [0,1]:
            if ptnum==0:
                points = cog_points
                line_points = body_line_points
                lines = body_lines
                ball_color = ball_color1
                line_color = line_color1
            else:
                line_points = body_line_points2
                points = cog_points2
                lines = body_lines2
                ball_color = ball_color2
                line_color = line_color2
                            
            points_poly_data = vtkPolyData()
            points_poly_data.SetPoints(points)

            head = vtk.vtkSphereSource()
            head.SetRadius(.5)
            #head.SetRadius(1.5)
            head.SetThetaResolution(8)
            head.SetPhiResolution(8)
    ##        head.SetThetaResolution(15)
    ##        head.SetPhiResolution(15)

            head_glyphs = vtk.vtkGlyph3D()
            head_glyphs.SetInput(points_poly_data)
            head_glyphs.SetSource(head.GetOutput())

            head_glyph_mapper = vtkPolyDataMapper()
            head_glyph_mapper.SetInput( head_glyphs.GetOutput())
            headGlyphActor = vtk.vtkActor()
            headGlyphActor.SetMapper(head_glyph_mapper)
            headGlyphActor.GetProperty().SetDiffuseColor(ball_color)
            headGlyphActor.GetProperty().SetSpecular(.3)
            headGlyphActor.GetProperty().SetSpecularPower(30)

            for renderer in renderers:
                renderer.AddActor( headGlyphActor )
            actors.append( headGlyphActor )

            if line_color is not None:
                # body line rendering 
                # ( see VTK demo Rendering/Python/CSpline.py )

                profileData = vtk.vtkPolyData()

                profileData.SetPoints(line_points)
                profileData.SetLines(lines)

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
                profile.GetProperty().SetDiffuseColor( line_color ) #0xd6/255.0, 0xec/255.0, 0x1c/255.0)
                #profile.GetProperty().SetDiffuseColor(cerulean)
                #profile.GetProperty().SetDiffuseColor(banana)
                profile.GetProperty().SetSpecular(.3)
                profile.GetProperty().SetSpecularPower(30)

                for renderer in renderers:
                    renderer.AddActor( profile )
                actors.append( profile )
        
    elif render_mode.startswith('triangles'):
        profileData = vtk.vtkPolyData()
        profileData.SetPoints(tri_points)
        profileData.SetPolys(tri_cells)
        
        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInput(profileData)

        profile = vtk.vtkActor()
        profile.SetMapper(profileMapper)
        profile.GetProperty().SetDiffuseColor(triangle_mode_color)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)
    
        for renderer in renderers:
            renderer.AddActor( profile )
        actors.append( profile )

        # fin
        
        profileData = vtk.vtkPolyData()
        profileData.SetPoints(tri2_points)
        profileData.SetPolys(tri2_cells)
        
        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInput(profileData)

        profile = vtk.vtkActor()
        profile.SetMapper(profileMapper)
        profile.GetProperty().SetDiffuseColor(black)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)
    
        for renderer in renderers:
            renderer.AddActor( profile )
        actors.append( profile )

    if plot_timed_forces:
        # timed_force line rendering 
        # ( see VTK demo Rendering/Python/CSpline.py )

        profileData = vtk.vtkPolyData()

        profileData.SetPoints(timed_force_line_points)
        profileData.SetLines(timed_force_lines)

        # Add thickness to the resulting line.
        profileTubes = vtk.vtkTubeFilter()
        profileTubes.SetNumberOfSides(8)
        profileTubes.SetInput(profileData)
        profileTubes.SetRadius(timed_force_radius)

        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInput(profileTubes.GetOutput())

        profile = vtk.vtkActor()
        profile.SetMapper(profileMapper)
        profile.GetProperty().SetDiffuseColor(timed_force_color)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)

        for renderer in renderers:
            renderer.AddActor( profile )
        actors.append( profile )

    # bounding box
    bbox_points = vtkPoints()
    if X_zero_frame is None:
        bbox_points.InsertNextPoint(400,0,0)
        bbox_points.InsertNextPoint(1000,300,300)
    else:
        bbox_points.InsertNextPoint(-150,-150,-150)
        bbox_points.InsertNextPoint(150,150,150)
    bbox_poly_data = vtkPolyData()
    bbox_poly_data.SetPoints(bbox_points)
    bbox_mapper = vtk.vtkPolyDataMapper()
    bbox_mapper.SetInput(bbox_poly_data)
    bbox=vtk.vtkActor()
    bbox.SetMapper(bbox_mapper)
##    for renderer in renderers:
##        renderer.AddActor( bbox )
##    actors.append( bbox )
##    print 'bbox drawn at',( xlim[0], ylim[0], zlim[0] )
##    print ( xlim[1], ylim[1], zlim[1] )

    if labels:
        for frame_no, X in zip(frame_nos,Xs):
            if X is None:
                continue
            if use_timestamps:
                if frame_no_offset == 0:
                    fdiff = f1
                else:
                    fdiff = frame_no_offset
                if (frame_no-fdiff)%10 != 0:
                    continue
                label = str((frame_no-fdiff)/100.0)
            else:
                if (frame_no-frame_no_offset)%10 != 0:
                    continue
                #print 'frame_no',frame_no
                #print 'frame_no_offset',frame_no_offset
                label = str(frame_no-frame_no_offset)
##            X = X.flat
            # labels
            
            tl = vtkTextActor()
            tl.SetInput( label )
            tl.GetPositionCoordinate().SetCoordinateSystemToWorld()
            tl.GetPositionCoordinate().SetValue(*X)
            tl.SetAlignmentPoint(0)
            if 0:
                tl.GetTextProperty().SetColor(0,0,0)
            else:
                tl.GetTextProperty().SetColor(white)
                tl.GetTextProperty().SetShadow(True)
            tl.GetTextProperty().SetJustificationToCentered() # does nothing?
            for renderer in renderers:
                renderer.AddActor( tl )
            actors.append( tl )

    if bounding_box:
        # from Annotation/Python/cubeAxes.py
        tprop = vtk.vtkTextProperty()
        tprop.SetColor(0,0,0)
        #tprop.ShadowOn()
        for renderer in renderers:
            axes2 = vtk.vtkCubeAxesActor2D()
            axes2.SetProp(bbox)
            axes2.SetCamera(renderer.GetActiveCamera())
            axes2.SetLabelFormat("%6.4g")
            #axes2.SetFlyModeToOuterEdges()
            axes2.SetFlyModeToClosestTriad()
            axes2.SetFontFactor(0.8)
            axes2.ScalingOff()
            axes2.SetAxisTitleTextProperty(tprop)
            axes2.SetAxisLabelTextProperty(tprop)
            axes2.GetProperty().SetColor(0,0,0)
            renderer.AddActor(axes2)
            actors.append( axes2 )
        
##    return bbox
    return actors
    
def interact_with_renWin(renWin):

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow( renWin )

    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.Initialize ()
    
    renWin.Render()
    
    iren.Start()

##def interact_with_stereo_renWin():
##    global stereo_renWin, iren
    
##    iren = vtkRenderWindowInteractor()
##    iren.SetInteractorStyle(None)
##    iren.SetRenderWindow( stereo_renWin )

##    #iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

##    iren.AddObserver("LeftButtonPressEvent", ButtonEvent)
##    iren.AddObserver("LeftButtonReleaseEvent", ButtonEvent)
##    iren.AddObserver("MiddleButtonPressEvent", ButtonEvent)
##    iren.AddObserver("MiddleButtonReleaseEvent", ButtonEvent)
##    iren.AddObserver("RightButtonPressEvent", ButtonEvent)
##    iren.AddObserver("RightButtonReleaseEvent", ButtonEvent)
##    iren.AddObserver("MouseMoveEvent", MouseMove)
    
##    iren.Initialize ()
    
##    stereo_renWin.Render()
    
##    iren.Start()
    
def print_cam_props(camera):
    print 'camera.SetParallelProjection(%s)'%str(camera.GetParallelProjection())
    print 'camera.SetFocalPoint',camera.GetFocalPoint()
    print 'camera.SetPosition',camera.GetPosition()        
    print 'camera.SetViewAngle(%s)'%str(camera.GetViewAngle())
    print 'camera.SetViewUp',camera.GetViewUp()
    print 'camera.SetClippingRange',camera.GetClippingRange()
    print 'camera.SetParallelScale(%s)'%str(camera.GetParallelScale())
    print

### Add the observers to watch for particular events. These invoke
### Python functions.
##Rotating = 0
##Panning = 0
##Zooming = 0

### Handle the mouse button events.
##def ButtonEvent(obj, event):
##    global Rotating, Panning, Zooming
##    if event == "LeftButtonPressEvent":
##        Rotating = 1
##    elif event == "LeftButtonReleaseEvent":
##        Rotating = 0
##    elif event == "MiddleButtonPressEvent":
##        Panning = 1
##    elif event == "MiddleButtonReleaseEvent":
##        Panning = 0
##    elif event == "RightButtonPressEvent":
##        Zooming = 1
##    elif event == "RightButtonReleaseEvent":
##        Zooming = 0

### General high-level logic
##def MouseMove(obj, event):
##    global Rotating, Panning, Zooming
##    global iren, stereo_renWin, center_camera

##    lastXYpos = iren.GetLastEventPosition()
##    lastX = lastXYpos[0]
##    lastY = lastXYpos[1]

##    xypos = iren.GetEventPosition()
##    x = xypos[0]
##    y = xypos[1]

##    center = stereo_renWin.GetSize()
##    centerX = center[0]/2.0
##    centerY = center[1]/2.0

##    if Rotating:
##        Rotate(center_ren,center_cam, x, y, lastX, lastY,
##               centerX, centerY)
##    elif Panning:
##        Pan(center_ren, center_cam, x, y, lastX, lastY, centerX,
##            centerY)
##    elif Zooming:
##        Dolly(center_ren, center_cam, x, y, lastX, lastY,
##              centerX, centerY)
##    else:
##        return
##    set_lr_cams_from_center( left_cam, right_cam, center_cam )
##    stereo_renWin.Render()

##def Rotate(renderer, camera, x, y, lastX, lastY, centerX, centerY):    
##    camera.Azimuth(lastX-x)
##    camera.Elevation(lastY-y)
##    camera.OrthogonalizeViewUp()

### Pan translates x-y motion into translation of the focal point and
### position.
##def Pan(renderer, camera, x, y, lastX, lastY, centerX, centerY):
##    FPoint = camera.GetFocalPoint()
##    FPoint0 = FPoint[0]
##    FPoint1 = FPoint[1]
##    FPoint2 = FPoint[2]

##    PPoint = camera.GetPosition()
##    PPoint0 = PPoint[0]
##    PPoint1 = PPoint[1]
##    PPoint2 = PPoint[2]

##    renderer.SetWorldPoint(FPoint0, FPoint1, FPoint2, 1.0)
##    renderer.WorldToDisplay()
##    DPoint = renderer.GetDisplayPoint()
##    focalDepth = DPoint[2]

##    APoint0 = centerX+(x-lastX)
##    APoint1 = centerY+(y-lastY)
    
##    renderer.SetDisplayPoint(APoint0, APoint1, focalDepth)
##    renderer.DisplayToWorld()
##    RPoint = renderer.GetWorldPoint()
##    RPoint0 = RPoint[0]
##    RPoint1 = RPoint[1]
##    RPoint2 = RPoint[2]
##    RPoint3 = RPoint[3]
    
##    if RPoint3 != 0.0:
##        RPoint0 = RPoint0/RPoint3
##        RPoint1 = RPoint1/RPoint3
##        RPoint2 = RPoint2/RPoint3

##    camera.SetFocalPoint( (FPoint0-RPoint0)/2.0 + FPoint0,
##                          (FPoint1-RPoint1)/2.0 + FPoint1,
##                          (FPoint2-RPoint2)/2.0 + FPoint2)
##    camera.SetPosition( (FPoint0-RPoint0)/2.0 + PPoint0,
##                        (FPoint1-RPoint1)/2.0 + PPoint1,
##                        (FPoint2-RPoint2)/2.0 + PPoint2)

### Dolly converts y-motion into a camera dolly commands.
##def Dolly(renderer, camera, x, y, lastX, lastY, centerX, centerY):
##    dollyFactor = pow(1.02,(0.5*(y-lastY)))
##    if camera.GetParallelProjection():
##        parallelScale = camera.GetParallelScale()*dollyFactor
##        camera.SetParallelScale(parallelScale)
##    else:
##        camera.Dolly(dollyFactor)
##        renderer.ResetCameraClippingRange()

if __name__=='__main__':
    import result_browser

    #results
    if 1:
        results = result_browser.get_results('DATA20060717_185535.h5',mode='r+')
        start_frame = 609150
        stop_frame = 609800
        
    
    if 1:

        #start_frame = 2
        #stop_frame = 50
        
        renWin, renderers = init_vtk(stereo=True)
        #show_cameras(results,renderers)

        if 0:
            CT=array([ 181.88106377,  221.06126383,  168.28886479])
            CB=array([ 188.25655514,  218.76102605,   30.89531996])
            show_line(renderers,CT,CB,black,4)
        if 0:
            NZ = array([   9.11331261,  117.08933803,   53.84209957])
            NY = array([  10.98416978,  392.19324712,   70.9049832 ])
            show_line(renderers,NZ,NY,blue,1)
        if 0:
            # bottom of area with pattern
            corner = array([  4.91559111,  54.73864537,  32.58650871])
            sfw = array([ -13.64048628,  335.36740794,   22.02908834])
            lwe = array([ 285.12425295,   60.4681217 ,   40.7247129 ])
            upc = array([   7.91709368,   64.14688249,  184.80049719])

            if 0:
                show_line(renderers,corner,sfw,blue,1)
                show_line(renderers,corner,lwe,blue,1)
                show_line(renderers,corner,upc,blue,1)

            show_texture(renderers, corner, upc, sfw, (32,16,1))
            show_texture(renderers, corner, upc, lwe, (32,16,1))
            show_texture(renderers, corner, lwe, sfw, (32,32,1))

            def mag(v):
                return math.sqrt( v[0]**2 + v[1]**2 + v[2]**2)
            
            h=(upc-corner)
            updir = h/mag(h)
            upside = 304.8 * updir # 304.8 mm = 1 foot


            # cap
            c1 = array([ 130.85457512,  169.45421191,   50.53490689])
            show_line(renderers,c1-updir*10,c1-updir*20,light_grey,4,opacity=0.5) # hmm, doctored position

            le=(lwe-corner)
            wdir = le/mag(le)
            wside = 2*304.8*wdir

            se=(sfw-corner)
            sdir = se/mag(se)
            sside = 304.8*sdir

            if 1:
                line_color = light_grey

                show_line(renderers,corner,corner+upside,line_color,1)
                show_line(renderers,corner,corner+wside,line_color,1)
                show_line(renderers,corner,corner+sside,line_color,1)

                show_line(renderers,corner+wside,corner+wside+upside,line_color,1)
                show_line(renderers,corner+sside, corner+sside+upside,line_color,1)
                show_line(renderers,corner+sside, corner+sside+wside,line_color,1)
                show_line(renderers,corner+sside+wside, corner+sside+wside+upside,line_color,1)
                show_line(renderers,corner+wside,corner+wside+sside,line_color,1)
            
##            A = nx.zeros( (32,32,1), nx.UInt8 )
##            for row in range(A.shape[0]):
##                for col in range(A.shape[1]):
##                    if random.random() > 0.5:
##                        A[row,col,0] = 255
##                    else:
##                        A[row,col,0] = 0
##            show_numpy_image( renderers, A, corner, upc, lwe )
            
####        show_frames_vtk(results,renderers,start_frame,stop_frame,1,
####                        orientation_corrected=True,
####                        #timed_force_table=results.root.resultant_forces,
####                        #timed_force_color=red,
####                        use_timestamps=True,max_err=10)
##        if 0:
##            show_frames_vtk(results,renderers,start_frame,stop_frame,1,
##                            orientation_corrected=True,
##                            timed_force_table=results.root.real_resultant_forces,
##                            timed_force_color=green,
##                            use_timestamps=True,max_err=10)
        if 0:
            show_frames_vtk(results,renderers,start_frame,stop_frame,1,
                            orientation_corrected=True,
                            labels=True,
                            #timed_force_table=results.root.drag_force_linear,
                            #timed_force_table=results.root.real_timed_forces,
                            timed_force_color=green,
                            timed_force_scaling_factor=5e5,
                            render_mode='triangles',
                            triangle_mode_data=results.root.smooth_data_roll_fixed_lin,
                            triangle_mode_color=red,
                            use_timestamps=True)
        if 0:
            show_frames_vtk(results,renderers,start_frame,stop_frame,1,
                            orientation_corrected=True,
                            labels=True,
                            #timed_force_table=results.root.resultants,
                            timed_force_color=blue,
                            timed_force_scaling_factor=5e5,
                            timed_force_radius=0.3,
                            render_mode='triangles',
                            triangle_mode_data=results.root.smooth_data,
                            #triangle_mode_data=results.root.smooth_data_real,
                            triangle_mode_color=green,
                            use_timestamps=True)
        if 1:
            show_frames_vtk(results,renderers,start_frame,stop_frame,1,
                            render_mode='ball_and_stick',
                            labels=True,#False,
                            orientation_corrected=False,
                            use_timestamps=True,max_err=10)
            
        for renderer in renderers:
            renderer.ResetCameraClippingRange()
        if 1:
            interact_with_renWin(renWin)
            for renderer in renderers:
                print_cam_props(renderer.GetActiveCamera())
        else:
            deg2rad = math.pi/180.0
            if 0:
                az1 = list(-nx.arange( 0.0, 90.0, 1.0))
                az2 = [az1[-1]]*20
                az3 = az1[::-1]
                az4 = [az3[-1]]*20
            else:
                az1 = list(-nx.arange( 0.0, 90.0, 10.0))
                az2 = []
                az3 = az1[::-1]
                az4 = []
            azs = az1 + az2 + az3 + az4
            save_to_file = True
            
            if save_to_file:
                imf = vtkWindowToImageFilter()
                imf.SetInput(renWin)
                imf.Update()
                
            if 1:
                for frame_no,az in enumerate(azs):
                    for renderer in renderers:
                        camera = renderer.GetActiveCamera()
                        camera.SetPosition (103.94839384485147, 1090.2877280742541, 238.19531097666248)
                        camera.Azimuth(az)
                        renderer.ResetCameraClippingRange()
                    renWin.Render()

                    if save_to_file:

                        writer = vtk.vtkPNGWriter()
                        #writer.SetInput(imf.GetOutput())

                        imf.Modified()

                        writer.SetInput(imf.GetOutput())
                        fname = 'small_vtk%03d.png'%(frame_no+1,)
                        print 'saving',fname
                        writer.SetFileName(fname)
                        writer.Write()

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
