#!/usr/bin/env python
# $Id$
import numarray as nx
import math
from wxPython.wx import *
import wxVTKRenderWindow
from vtkpython import *
from vtk.util.colors import tomato, banana, azure
import time, sys, threading, socket, struct
import flydra.reconstruct as reconstruct

hostname = socket.gethostbyname('mainbrain')
sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

port = 28931

listen_socket = sockobj.bind(( hostname, port))
print 'listening on',hostname,port

incoming_data_lock = threading.Lock()
incoming_data = []

class Listener(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.quit_now = threading.Event()
##        self.is_done = threading.Event()
        self.fmt = '<ifffffffffdi'
    def run(self):
        global incoming_data_lock, incoming_data
        tick = 0
        while not self.quit_now.isSet():
            data, addr = sockobj.recvfrom(1024)
            tmp = struct.unpack(self.fmt,data)
            corrected_framenumber,x,y,z = tmp[:4]
            line3d = tmp[4:10]
            timestamp, n_cams = tmp[10:12]
            #time.sleep(0.5)

            incoming_data_lock.acquire()
            incoming_data.append( (corrected_framenumber, (x,y,z), line3d, timestamp, n_cams ))
            incoming_data_lock.release()
                
##        self.is_done.set()
    def quit(self):
        self.quit_now.set()
##        # send packet to self (allow blocking call to quit)
##        tmp_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
##        tmp_socket.sendto('x'*struct.calcsize(self.fmt),
##                          ( hostname, port))
##        self.is_done.wait(0.1)
##        print
            
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

        C = nx.array((119, 16, -59))
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

        ren.SetBackground( 1,1,1)

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
        ren2.SetBackground( 1,1,1)

        #########################################
    
        self.current_cog_point = 0
        self.allocated_cog_points = 0

        self.max_cog_points = 10
        
        self.current_line_point = 0
        self.allocated_line_points = 0

        self.max_line_points = 20
        
        self.update_freq_sec = 0.05
        
        self.cog_points = vtk.vtkPoints() # 'center of gravity'
        self.line_points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()

        self.last_update = 0.0
        
        # point rendering #######################
    
        points_poly_data = vtkPolyData()
        points_poly_data.SetPoints(self.cog_points)

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
    
        profileData.SetPoints(self.line_points)
        profileData.SetLines(self.lines)

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
        bbox_points.InsertNextPoint(-200,-100,-150) # minimum values
        bbox_points.InsertNextPoint(500,450,200) # max
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

        ID_Timer  = wxNewId() 	         
        self.timer = wxTimer(self,      # object to send the event to 	 
                             ID_Timer)  # event id to use 	 
        EVT_TIMER(self,  ID_Timer, self.OnIdle)
        self.update_interval=30
        self.timer.Start(self.update_interval) # call every n msec
        EVT_IDLE(self.frame, self.OnIdle)
        
        #########################################

        self.frame.Show()
        self.SetTopWindow(self.frame)

        return True
    
    def OnIdle(self,event):
        global incoming_data_lock, incoming_data

        incoming_data_lock.acquire()
        local_data = incoming_data[:] # copy
        incoming_data = [] # delete
        incoming_data_lock.release()

        # self.cog_points acts like a FIFO
        for tmp in local_data:
            corrected_framenumber, (x,y,z), line3d, timestamp, n_cams = tmp
            X = nx.array((x,y,z))
            U = reconstruct.line_direction(line3d)
            if self.allocated_cog_points < self.max_cog_points:
                self.cog_points.InsertNextPoint( x,y,z )

                self.current_cog_point += 1
                self.allocated_cog_points += 1
                # line
                self.line_points.InsertNextPoint(*(X+20*U))
                self.current_line_point += 1
                self.allocated_line_points += 1
                
                self.line_points.InsertNextPoint(*(X-20*U))
                self.current_line_point += 1
                self.allocated_line_points += 1

                self.lines.InsertNextCell(2)
                self.lines.InsertCellPoint(self.current_line_point-2)
                self.lines.InsertCellPoint(self.current_line_point-1)
            else:
                if self.current_cog_point >= self.max_cog_points:
                    self.current_cog_point = 0
                self.cog_points.SetPoint( self.current_cog_point, x,y,z )
                self.current_cog_point += 1
                # line
                if self.current_line_point >= self.max_line_points:
                    self.current_line_point = 0
                self.line_points.SetPoint(self.current_line_point, *(X+20*U))
                self.current_line_point += 1
                
                self.line_points.SetPoint(self.current_line_point, *(X-20*U))
                self.current_line_point += 1
                
        if len(local_data):
            now = time.time()
            if now-self.last_update > self.update_freq_sec: # hand-tuned to prevent CPU overload
                
                points_poly_data = vtkPolyData()
                points_poly_data.SetPoints(self.cog_points)
                self.balls.SetInput(points_poly_data)

                profileData = vtk.vtkPolyData()
                profileData.SetPoints(self.line_points)
                profileData.SetLines(self.lines)
                self.profileTubes.SetInput(profileData)
                
                self.vtk_render_window.Render()
                self.last_update = now

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
    listener = Listener()
    listener.setDaemon(True) # don't let this thread keep app alive
    listener.start()
    
    # initialize GUI
    #app = App(redirect=1,filename='viewer_log.txt')
    app = App() 

    app.MainLoop()

    print_cam_props(app.cam1)
    print_cam_props(app.cam2)
    del app
    listener.quit()

if __name__ == '__main__':
    main()
