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
camera.SetFocalPoint(0,0,300)
#camera.SetFocalPoint(135.822, -15.4543, 231.17)
camera.SetPosition(-1749.21, 741.993, 1339.98)
camera.SetViewAngle( 20.0 )
camera.SetViewUp(0.48599, -0.0662295, 0.871451)
camera.SetClippingRange( 1108.07, 3026.28 )

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

for CUR_PT in range(X.shape[0]):

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

    renWin.SetSize( 800, 600 )

    renWin.Render()

    imf.Modified()
    writer.SetInput(imf.GetOutput())
    fname = 'im%04d.png'%CUR_PT
    print 'saving',fname
    writer.SetFileName(fname)
    writer.Write()
    print 'done'

##    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
##    iren.Initialize ()
##    iren.Start()

##    print camera
