#!/usr/bin/env python
import thread
import time
import socket
import os
import numarray
import Pyro.core
import DynamicImageCanvas
Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

from wxPython.wx import *
from wxPython.lib import newevent
from wxPython.xrc import *

RESDIR = os.path.split(os.path.abspath(sys.argv[0]))[0]
RESFILE = os.path.join(RESDIR,'flydra_server.xrc')
hydra_image_file = os.path.join(RESDIR,'hydra.gif')
RES = wxXmlResource(RESFILE)

class FlydraBrainPyroServer( Pyro.core.ObjBase ):
    def set_wxApp(self,wxApp):
        self.wxApp = wxApp
        self.servlets = {}
        self.update_wx()
    def update_wx(self):
        self.wxApp.statusbar.SetStatusText('%d camera servlet(s)'%len(self.servlets),2)
    def get_URI_for_new_camera_servlet(self):
        cam_serv = CameraServlet()
        cam_serv.set_wxApp( self.wxApp )
        daemon = self.getDaemon()
        URI=daemon.connect(cam_serv) # start serving cam_serv
        self.servlets[URI]=cam_serv
        self.update_wx()
        return URI
    def delete_servlet_by_URI(self,URI):
        cam_serv = self.servlets[URI]
        del self.servlets[URI]
        cam_serv.close()
        self.update_wx()
        
class CameraServlet( Pyro.core.ObjBase ):
    """Communication between camPanel and Pyro"""
    
    # -=-=-=-=-= start remote access -=-=-=-=-=
    
    def set_cam_info(self, cam_id, scalar_control_info):
        # this is like a 2nd __init__
        self.cam_id = cam_id
        self.scalar_control_info = scalar_control_info

        self.my_container = self.wxApp.all_cam_panel
        self.start_time = time.time()

        self.n_frames = 0
        self.last_measurement_time = time.time()
        self.updates = {}

        self.init_gui()

    def get_updates(self):
        result = self.updates.items()
        self.updates = {} # clear queue
        return result

    def push_image(self, image):
        self.cam_image_canvas.update_texture(image)
        self.n_frames += 1

    def set_current_fps(self, acquired_fps):
        #self.current_fps = fps
        acquired_fps_label = XRCCTRL(self.camPanel,'acquired_fps_label') # get container
        acquired_fps_label.SetLabel('Frames per second (acquired): %.1f'%acquired_fps)
        
        now = time.time()
        elapsed = now-self.last_measurement_time
        self.last_measurement_time = now
        displayed_fps = self.n_frames/elapsed
        displayed_fps_label = XRCCTRL(self.camPanel,'displayed_fps_label') # get container
        displayed_fps_label.SetLabel('Frames per second (displayed): %.1f'%displayed_fps)
        
        self.n_frames = 0

    # -=-=-=-=-= end remote access -=-=-=-=-=
        
    def set_wxApp(self, wxApp):
        self.wxApp = wxApp

    def disconnect_camera(self, event):
        self.updates['quit']=True

    def init_gui(self):
        """build GUI"""
        
        self.camPanel = RES.LoadPanel(self.my_container,"PerCameraPanel")
        # Add myself to my_container's sizer
        acp_box = self.my_container.GetSizer()
        acp_box.Add(self.camPanel,1,wxEXPAND | wxALL,border=10)

        if 0:
            box = self.camPanel.GetSizer()
            static_box = box.GetStaticBox()
            static_box.SetLabel( 'Camera ID: %s'%self.cam_id )


        caller_addr= self.daemon.getLocalStorage().caller.addr
        caller_ip, caller_port = caller_addr
        fqdn = socket.getfqdn(caller_ip)

        cam_info_label = XRCCTRL(self.camPanel,'cam_info_label')
        cam_info_label.SetLabel('Camera: %s:%d'%(fqdn,caller_port))

        quit_camera = XRCCTRL(self.camPanel,"quit_camera") # get container
        EVT_BUTTON(quit_camera, quit_camera.GetId(), self.disconnect_camera)
        
        dynamic_image_panel = XRCCTRL(self.camPanel,"DynamicImagePanel") # get container

        self.cam_image_canvas = DynamicImageCanvas.DynamicImageCanvas(dynamic_image_panel,-1) # put GL window in container
        #self.cam_image_canvas = wxButton(dynamic_image_panel,-1,"Button") # put GL window in container

        box = wxBoxSizer(wxVERTICAL)
        #box.Add(self.cam_image_canvas,1,wxEXPAND|wxSHAPED) # keep aspect ratio
        box.Add(self.cam_image_canvas,1,wxEXPAND)
        dynamic_image_panel.SetSizer(box)
        dynamic_image_panel.Layout()

        per_cam_controls_panel = XRCCTRL(self.camPanel,"PerCameraControlsContainer") # get container
        box = wxBoxSizer(wxVERTICAL)

        for param in self.scalar_control_info:
            current_value, min_value, max_value = self.scalar_control_info[param]
            scalarPanel = RES.LoadPanel(per_cam_controls_panel,"ScalarControlPanel") # frame main panel
            box.Add(scalarPanel,1,wxEXPAND)
            
            label = XRCCTRL(scalarPanel,'scalar_control_label')
            label.SetLabel( param )
            
            slider = XRCCTRL(scalarPanel,'scalar_control_slider')
            slider.SetRange( min_value, max_value )
            slider.SetValue( current_value )
            
            class ParamSliderHelper:
                def __init__(self, name, slider, parent):
                    self.name=name
                    self.slider=slider
                    self.parent = parent
                def onScroll(self, event):
                    self.parent.updates[self.name] = self.slider.GetValue()
            
            psh = ParamSliderHelper(param,slider,self)
            EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
        
        per_cam_controls_panel.SetSizer(box)
        self.my_container.Layout()

    def close(self):
        if hasattr(self,'camPanel'):
            #print 'closing...'
            self.camPanel.DestroyChildren()
            self.camPanel.Destroy()
            del self.camPanel
        #else:
        #    print 'WARNING: calling close 2nd time'
        
class App(wxApp):
    def OnInit(self,*args,**kw):
    
        wxInitAllImageHandlers()
        frame = wxFrame(None, -1, "Flydra Control Center",size=(700,500))
        
        self.statusbar = frame.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        menuBar = wxMenuBar()
        filemenu = wxMenu()
        ID_quit = wxNewId()
        filemenu.Append(ID_quit, "Quit\tCtrl-Q", "Quit application")
        EVT_MENU(self, ID_quit, self.OnQuit)
        menuBar.Append(filemenu, "&File")
        frame.SetMenuBar(menuBar)

        panel = RES.LoadPanel(frame,"FlydraPanel") # frame main panel
        panel.SetFocus()

##        hydra_bitmap = XRCCTRL(panel,'hydra_pic') # get wxStaticBitmap
##        #if hydra_image_file.endswith('.gif'):
##        if 1:
##            tmp = wxImage(hydra_image_file, wxBITMAP_TYPE_GIF).ConvertToBitmap()
##            hydra_bitmap.SetBitmap(tmp)
##            #hydra_bitmap.SetSize((200,200))

        frame_box = wxBoxSizer(wxVERTICAL)
        frame_box.Add(panel,1,wxEXPAND)
        frame.SetSizer(frame_box)
        frame.Layout()

        #####################################

        self.all_cam_panel = XRCCTRL(panel,"AllCamPanel")

        acp_box = wxBoxSizer(wxVERTICAL)
        self.all_cam_panel.SetSizer(acp_box)
        
        acp_box.Add(wxStaticText(self.all_cam_panel,-1,"This is the main panel"),0,wxEXPAND)
        self.all_cam_panel.Layout()

        #########################################

        frame.SetAutoLayout(true)

        frame.Show()
        self.SetTopWindow(frame)
        self.frame = frame

        self.start_pyro_server()

        #EVT_IDLE(self, self.OnTimer)
        
        ID_Timer  = wxNewId()
        self.timer = wxTimer(self,      # object to send the event to
                             ID_Timer)  # event id to use
        EVT_TIMER(self,  ID_Timer, self.OnTimer)
        self.timer.Start(10)
        
        return True
    
    def OnQuit(self, event):
        self.frame.Close(True)

    def OnTimer(self, event):
        self.daemon.handleRequests(0) # don't block
        
    def start_pyro_server(self):
        Pyro.core.initServer(banner=0)
        hostname = socket.gethostbyname(socket.gethostname())
        fqdn = socket.getfqdn(hostname)
        port = 9832
        self.daemon = Pyro.core.Daemon(host=hostname,port=port)
        self.flydra_brain = FlydraBrainPyroServer()
        self.flydra_brain.set_wxApp(self)
        URI=self.daemon.connect(self.flydra_brain,'flydra_brain')
        self.statusbar.SetStatusText("flydra_brain at %s:%d"%(fqdn,port), 1)
        self.frame.SetFocus()

def main():
    #app = App(redirect=1,filename='flydra_log.txt')
    app = App()
    app.MainLoop()

if __name__ == '__main__':
    main()

