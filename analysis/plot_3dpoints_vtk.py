#!/usr/bin/env python

import os
os.environ['__GL_FSAA_MODE']='5' # 4x gaussian multisampling on geForce3 linux
opj=os.path.join
from vtkpython import *
import numarray as na

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    lines = buf.split('\n')[:-1]
    return na.array([map(float,line.split()) for line in lines])

X=load_ascii_matrix('X.recalc.dat')

renWin = vtkRenderWindow()
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow( renWin )

camera = vtkCamera()
camera.SetFocalPoint(11.653, 118.202, 300.084)
camera.SetPosition(-2176.26, -171.996, 537.824)
camera.SetViewAngle( 30.0 )
camera.SetViewUp(0.124543, -0.13351, 0.983191)
camera.SetClippingRange( 1842.94, 2532.1)

ren1 = vtkRenderer()
lk = vtkLightKit()
ren1.SetViewport(0.0,0.0,1.0,1.0)
ren1.SetBackground( 1,1,1)

ren1.SetActiveCamera( camera )

renWin.AddRenderer( ren1 )

imf = vtkWindowToImageFilter()
imf.SetInput(renWin)
imf.Update()
writer = vtk.vtkPNGWriter()

SAVE=1

#for CUR_PT in range(X.shape[0]):
#for CUR_PT in range(395,472):
for CUR_PT in range(410,472):

    points = vtk.vtkPoints()
    pt = X[CUR_PT,:]
    points.InsertNextPoint(*pt)

    profile = vtkPolyData()
    profile.SetPoints(points)

    ball = vtk.vtkSphereSource()
    ball.SetRadius(10.0)
    ball.SetThetaResolution(15)
    ball.SetPhiResolution(15)
    balls = vtk.vtkGlyph3D()
    balls.SetInput(profile)
    balls.SetSource(ball.GetOutput())
    mapBalls = vtkPolyDataMapper()
    mapBalls.SetInput( balls.GetOutput())
    ballActor = vtk.vtkActor()
    ballActor.SetMapper(mapBalls)

    ren1.AddActor( ballActor )

##    if CUR_PT>0:
##        for i in range(CUR_PT-1):
##            pt1=X[i]
##            pt2=X[i+1]

##            line = vtk.vtkLineSource()
##            line.SetPoint1(*pt1)
##            line.SetPoint2(*pt2)
##            lineMapper = vtk.vtkPolyDataMapper()
##            lineMapper.SetInput(line.GetOutput())
##            lineActor = vtk.vtkActor()
##            lineActor.SetMapper(lineMapper)
##            lineActor.GetProperty().SetColor(0, 0, 0)
##            lineActor.GetProperty().SetLineWidth(1)
##            ren1.AddActor( lineActor )

    renWin.SetSize( 160, 480 )

    renWin.Render()

    if SAVE:
        imf.Modified()
        writer.SetInput(imf.GetOutput())
        fname = 'im%04d.png'%CUR_PT
        print 'saving',fname
        writer.SetFileName(fname)
        writer.Write()
        print 'done'
    else:
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.Initialize ()
        iren.Start()

        #print camera

if SAVE:
    inc=5.0
    ii=0
    for i in na.arange(0.0,720.0,inc):
        ii+=1
        renWin.Render()
        ren1.GetActiveCamera().Azimuth( inc )
        
        imf.Modified()
        writer.SetInput(imf.GetOutput())
        fname = 'rot%03d.png'%(ii,)
        print 'saving',fname
        writer.SetFileName(fname)
        writer.Write()
