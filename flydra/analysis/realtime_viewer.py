#!/usr/bin/env python
import Numeric as nx
import math
from wxPython.wx import *
import wxVTKRenderWindow
from vtkpython import *
from vtk.util.colors import tomato, banana, azure
import time, sys, threading, socket, struct
import flydra.reconstruct as reconstruct

#hostname = socket.gethostbyname('mainbrain')
hostname = '192.168.1.199'
sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

port = 28931

sockobj.bind(( hostname, port))
print 'listening on',hostname,port

EVT_TRIG_ID = wxNewId()
def EVT_TRIG(win, func):
    win.Connect(-1, -1, EVT_TRIG_ID, func)
class TrigEvent(wxPyEvent):
    def __init__(self,data):
        wxPyEvent.__init__(self)
        self.SetEventType(EVT_TRIG_ID)
        self.data = data

class Listener(threading.Thread):
    def __init__(self,wxapp):
        threading.Thread.__init__(self)
        self.quit_now = threading.Event()
        self.fmt = 'ifffffffffd'
        self.wxapp = wxapp
    def run(self):
        tick = 0
        print 'listening...'
        while not self.quit_now.isSet():
            data, addr = sockobj.recvfrom(1024)
            tmp = struct.unpack(self.fmt,data)
            corrected_framenumber,x,y,z = tmp[:4]
            line3d = tmp[4:10]
            timestamp = tmp[10]

            self.wxapp.AddPendingEvent(TrigEvent((corrected_framenumber, (x,y,z), line3d, timestamp )))

    def quit(self):
        self.quit_now.set()

class App(wxApp):
    def OnInit(self,*args,**kw):
        wxInitAllImageHandlers()
        self.frame = wxFrame(None, -1, "Flydra Realtime Data Viewer",size=(640,480))

        #########################################

        self.vtk_render_window = wxVTKRenderWindow.wxVTKRenderWindow(self.frame, -1)
        ren = vtkRenderer()
        ren.SetViewport(0.0,0.5,1.0,1.0)
        self.vtk_render_window.GetRenderWindow().AddRenderer(ren)
        ren2 = vtkRenderer()
        ren2.SetViewport(0.0,0.0,1.0,0.5)
        self.vtk_render_window.GetRenderWindow().AddRenderer(ren2)

        #########################################

        C = nx.array((1250, 150, 470))
        camera = vtkCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(*C)
        camera.SetPosition(C+nx.array((0,-400,0)))
        camera.SetViewAngle( 30.0 )
        camera.SetViewUp(0,0,1)
        camera.SetClippingRange( 115,2314 )
        camera.SetParallelScale(276)
        self.cam1 = camera
        ren.SetActiveCamera( camera )

        ren.SetBackground( .7,.7,.8 )
        #ren.SetBackground( 1,1,1)

        camera = vtkCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(*C)
        camera.SetPosition(C+nx.array((0,0,400)))
        #camera.SetPosition(-857, -380, 202)
        camera.SetViewAngle( 30.0 )
        camera.SetViewUp(0,1,0)
        camera.SetClippingRange( 115,2314 )
        camera.SetParallelScale(209)

        self.cam2 = camera
        ren2.SetActiveCamera( camera )
        ren2.SetBackground( .7,.7,.8 )
        #ren2.SetBackground( 1,1,1)

        # point rendering #######################

        points_poly_data = vtkPolyData()
#        points_poly_data.SetPoints(self.cog_points)

        ball = vtk.vtkSphereSource()
        ball.SetRadius(3.0)
        ball.SetThetaResolution(15)
        ball.SetPhiResolution(15)
        self.balls = vtk.vtkGlyph3D()
        self.balls.SetInput(points_poly_data)
        self.balls.SetSource(ball.GetOutput())
        mapBalls = vtkPolyDataMapper()
        mapBalls.SetInput( self.balls.GetOutput())
        ballActor = vtk.vtkActor()
        ballActor.SetMapper(mapBalls)

        ren.AddActor( ballActor )
        ren2.AddActor( ballActor )

        # point rendering #######################

        profileData = vtk.vtkPolyData()

#        profileData.SetPoints(self.line_points)
#        profileData.SetLines(self.lines)

        self.profileTubes = vtk.vtkTubeFilter()
        self.profileTubes.SetNumberOfSides(8)
        self.profileTubes.SetInput(profileData)
        self.profileTubes.SetRadius(1.0)

        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInput(self.profileTubes.GetOutput())

        profile = vtk.vtkActor()
        profile.SetMapper(profileMapper)
        profile.GetProperty().SetDiffuseColor(banana)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)

        ren.AddActor( profile )
        ren2.AddActor( profile )

        # bounding box
        bbox_points = vtkPoints()
        datarange = nx.array((200,100,100))
        bbox_points.InsertNextPoint(C-datarange) # minimum values
        bbox_points.InsertNextPoint(C+datarange) # max
        bbox_poly_data = vtkPolyData()
        bbox_poly_data.SetPoints(bbox_points)
        bbox_mapper = vtk.vtkPolyDataMapper()
        bbox_mapper.SetInput(bbox_poly_data)
        bbox=vtk.vtkActor()
        bbox.SetMapper(bbox_mapper)
        # (don't render)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(0,0,0)
        #tprop.ShadowOn()

        axes2 = vtk.vtkCubeAxesActor2D()
        axes2.SetProp(bbox)
        axes2.SetCamera(camera)
        axes2.SetLabelFormat("%6.4g")
        axes2.SetFlyModeToOuterEdges()
        #axes2.SetFlyModeToClosestTriad()
        axes2.SetFontFactor(0.8)
        axes2.ScalingOff()
        axes2.SetAxisTitleTextProperty(tprop)
        axes2.SetAxisLabelTextProperty(tprop)
        axes2.GetProperty().SetColor(0,0,0)
        ren.AddProp(axes2)
        ren2.AddProp(axes2)

        #########################################
        self.current_data = []
        #########################################

        ID_Timer  = wxNewId()
        self.timer = wxTimer(self,      # object to send the event to
                             ID_Timer)  # event id to use
        EVT_TIMER(self,  ID_Timer, self.OnTimer)
        self.update_interval=200
        self.timer.Start(self.update_interval) # call every n msec
        #EVT_IDLE(self.frame, self.OnIdle)

        ID_Timer2  = wxNewId()
        self.timer2 = wxTimer(self,      # object to send the event to
                              ID_Timer2)  # event id to use
        EVT_TIMER(self,  ID_Timer2, self.OnDraw)
        self.timer2.Start(200) # call every n msec

        #########################################

        EVT_TRIG(self, self.OnTrig)

        self.frame.Show()
        self.SetTopWindow(self.frame)

        return True

    def OnTrig(self,event):
        self.current_data.append( event.data )
        #print 'appending',event.data
        #print 'len(self.current_data)',len(self.current_data)

    def OnTimer(self,event):
        # prune data
        now = time.time()
        too_old_sec = 10.0
        while len(self.current_data):
            if (now - self.current_data[0][3]) > too_old_sec:
                # remove data older than too_old_sec seconds
                val = self.current_data.pop(0)
                #print 'removing',val
            else:
                # Because the list is ordered, first point that is
                # recent enough means the rest will be too.
                break

    def OnDraw(self,event=None):

        cog_points = vtk.vtkPoints() # 'center of gravity'
        line_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        current_line_point = 0

        N = len(self.current_data)
        cog_points.SetNumberOfPoints( N )

        #print 'drawing:',len(self.current_data)
        # self.cog_points acts like a FIFO
        for i,tmp in enumerate(self.current_data):
            corrected_framenumber, (x,y,z), line3d, timestamp = tmp
            X = nx.array((x,y,z))
            #print '  drawing',X

            # body
            cog_points.SetPoint(i,x,y,z )
            #cog_points.InsertNextPoint( x,y,z )
            if 0:
                U = reconstruct.line_direction(line3d)

                # line
                line_points.InsertNextPoint(*(X+20*U))
                current_line_point += 1
                line_points.InsertNextPoint(*(X-20*U))
                current_line_point += 1

                lines.InsertNextCell(2)
                lines.InsertCellPoint(current_line_point-2)
                lines.InsertCellPoint(current_line_point-1)

        if 1:
            points_poly_data = vtkPolyData()
            points_poly_data.SetPoints(cog_points)
            self.balls.SetInput(points_poly_data)

            if 0:
                profileData = vtk.vtkPolyData()
                profileData.SetPoints(line_points)
                profileData.SetLines(lines)
                self.profileTubes.SetInput(profileData)

            self.vtk_render_window.Render()

def print_cam_props(camera):
    print 'camera.SetParallelProjection',camera.GetParallelProjection()
    print 'camera.SetFocalPoint',camera.GetFocalPoint()
    print 'camera.SetPosition',camera.GetPosition()
    print 'camera.SetViewAngle',camera.GetViewAngle()
    print 'camera.SetViewUp',camera.GetViewUp()
    print 'camera.SetClippingRange',camera.GetClippingRange()
    print 'camera.SetParallelScale',camera.GetParallelScale()
    print

def main():
    # initialize GUI
    #app = App(redirect=1,filename='viewer_log.txt')
    app = App(redirect=0)
    #app = App()

    listener = Listener(app)
    listener.setDaemon(True) # don't let this thread keep app alive
    listener.start()

    app.MainLoop()

    print_cam_props(app.cam1)
    print_cam_props(app.cam2)
    del app
##    listener.quit()

if __name__ == '__main__':
    main()
