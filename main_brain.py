#!/usr/bin/env python
import sys
import threading
import time
import socket
import os
import copy
from MainBrain import MainBrain
from MatplotlibPanel import PlotPanel
import DynamicImageCanvas
from wxPython.wx import *
from wxPython.xrc import *
import ErrorDialog
import numarray
import opencv

RESDIR = os.path.split(os.path.abspath(sys.argv[0]))[0]
RESFILE = os.path.join(RESDIR,'flydra_server.xrc')
hydra_image_file = os.path.join(RESDIR,'hydra.gif')
RES = wxXmlResource(RESFILE)

        
class App(wxApp):
    def OnInit(self,*args,**kw):
    
        wxInitAllImageHandlers()
        frame = wxFrame(None, -1, "Flydra Main Brain",size=(800,600))

        # statusbar ----------------------------------
        self.statusbar = frame.CreateStatusBar()
        self.statusbar.SetFieldsCount(2)
        
        # menubar ------------------------------------
        menuBar = wxMenuBar()
        #   File
        filemenu = wxMenu()
        ID_quit = wxNewId()
        filemenu.Append(ID_quit, "Quit\tCtrl-Q", "Quit application")
        EVT_MENU(self, ID_quit, self.OnQuit)
        menuBar.Append(filemenu, "&File")

        #   View
        viewmenu = wxMenu()
        ID_toggle_image_tinting = wxNewId()
        viewmenu.Append(ID_toggle_image_tinting, "Tint clipped data",
                        "Tints clipped pixels blue", wx.ITEM_CHECK)
        EVT_MENU(self, ID_toggle_image_tinting, self.OnToggleTint)
        menuBar.Append(viewmenu, "&View")

        # finish menubar -----------------------------
        frame.SetMenuBar(menuBar)

        # main panel ----------------------------------
        self.main_panel = RES.LoadPanel(frame,"FLYDRA_PANEL") # make frame main panel
        self.main_panel.SetFocus()

        frame_box = wxBoxSizer(wxVERTICAL)
        frame_box.Add(self.main_panel,1,wxEXPAND)
        frame.SetSizer(frame_box)
        frame.Layout()

        nb = XRCCTRL(self.main_panel,"MAIN_NOTEBOOK")
        self.cam_preview_panel = RES.LoadPanel(nb,"CAM_PREVIEW_PANEL") # make camera preview panel
        nb.AddPage(self.cam_preview_panel,"Camera Preview/Settings")
        
        self.calibration_panel = RES.LoadPanel(nb,"CALIBRATION_PANEL") # make camera preview panel
        nb.AddPage(self.calibration_panel,"Calibration")
        self.InitCalibrationPanel()
        
        self.record_raw_panel = RES.LoadPanel(nb,"RECORD_RAW_PANEL") # make camera preview panel
        nb.AddPage(self.record_raw_panel,"Record raw video")

        temp_panel = RES.LoadPanel(nb,"UNDER_CONSTRUCTION_PANEL") # make camera preview panel
        nb.AddPage(temp_panel,"Realtime 3D tracking")

        nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanged)
        self.main_notebook = nb
        self.current_page = 'preview'

        #####################################

        self.all_cam_panel = XRCCTRL(self.cam_preview_panel,"AllCamPanel")

        acp_box = wxBoxSizer(wxHORIZONTAL) # all camera panel (for camera controls, e.g. gain)
        self.all_cam_panel.SetSizer(acp_box)
        
        #acp_box.Add(wxStaticText(self.all_cam_panel,-1,"This is the main panel"),0,wxEXPAND)
        self.all_cam_panel.Layout()

        #########################################

        frame.SetAutoLayout(true)

        frame.Show()
        self.SetTopWindow(frame)
        self.frame = frame

        dynamic_image_panel = XRCCTRL(self.main_panel,"DynamicImagePanel") # get container
        self.cam_image_canvas = DynamicImageCanvas.DynamicImageCanvas(dynamic_image_panel,-1) # put GL window in container
        viewmenu.Check(ID_toggle_image_tinting,self.cam_image_canvas.get_clipping())
        #self.cam_image_canvas = wxButton(dynamic_image_panel,-1,"Button") # put GL window in container

        box = wxBoxSizer(wxVERTICAL)
        #box.Add(self.cam_image_canvas,1,wxEXPAND|wxSHAPED) # keep aspect ratio
        box.Add(self.cam_image_canvas,1,wxEXPAND)
        dynamic_image_panel.SetSizer(box)
        dynamic_image_panel.Layout()
        
        EVT_IDLE(self.frame, self.OnIdle)

        self.cameras = {} #OrderedDict()
        self.wx_id_2_cam_id = {}

        self.update_wx()

        return True

    def OnPageChanged(self, event):
        page = event.GetSelection()
        self.statusbar.SetStatusText('',0)
        if page==0:
            self.current_page = 'preview'
            print 'cam preview'
        elif page==1:
            self.current_page = 'calibration'
            #self.OnEnterCalibrationPage()
        elif page==2:
            self.current_page = 'record'
            print 'record raw video'
        else:
            self.current_page = 'unknown'
            print 'unknown page'

    def InitCalibrationPanel(self):
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        find_chessboard_button = XRCCTRL(self.calibration_panel,
                                         "find_chessboard_button")
        EVT_BUTTON(find_chessboard_button, find_chessboard_button.GetId(),
                   self.OnFindChessboard)
        calibration_plot = XRCCTRL(self.calibration_panel,"calibration_plot")
        sizer = wxBoxSizer(wxVERTICAL)
        
        # matplotlib panel itself
        self.plotpanel = PlotPanel(calibration_plot)
        self.plotpanel.init_plot_data()
        
        # wx boilerplate
        sizer.Add(self.plotpanel, 1, wxEXPAND)
        calibration_plot.SetSizer(sizer)
        calibration_plot.Fit()

    def OnFindChessboard(self,event):
        try:
            calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                             "calibration_cam_choice")
            cam_id = calibration_cam_choice.GetStringSelection()
            if cam_id == '':
                return
            print 'get_image_sync(%s)'%cam_id
            frame = self.main_brain.get_image_sync(cam_id)
            print 'got frame:',frame.shape
            height = frame.shape[0]
            self.plotpanel.set_image(frame/255.0)
    #        self.plotpanel.set_image(frame[::-1]/255.0)

            etalon_width = int(XRCCTRL(self.calibration_panel,
                                       "etalon_width").GetValue())
            etalon_height = int(XRCCTRL(self.calibration_panel,
                                        "etalon_height").GetValue())
            etalon_size = etalon_width, etalon_height
            print 'etalon_size',etalon_size
            print 'enter opencv'
            found_all,corners = opencv.find_corners(frame,etalon_size)
            print 'return from opencv'
            
            status_string = 'Found %d corners, OpenCV '%len(corners)
            if found_all:
                status_string += 'thinks it found all.'
            else:
                status_string += 'does not think it found all.'
            self.statusbar.SetStatusText(status_string,0)
            if len(corners.shape)==2:
                self.plotpanel.set_data(corners[:,0],height-corners[:,1])
            else:
                self.plotpanel.set_data([0,0,0],[0,0,0])
            print 'entering draw'
            self.plotpanel.draw()
            print 'exit draw'
        except Exception, err:
            ErrorDialog.ShowErrorDialog(err)
            
    def OnToggleTint(self, event):
        self.cam_image_canvas.set_clipping( event.IsChecked() )

    def attach_and_start_main_brain(self,main_brain):
        self.main_brain = main_brain
        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening()

    def update_wx(self):
        self.statusbar.SetStatusText('%d camera(s)'%len(self.cameras),1)
        
    def OnQuit(self, event):
        print 'wxApp quit'
        del self.main_brain
        self.frame.Close(True)

    def OnIdle(self, event):
        
        if not hasattr(self,'main_brain'):
            return # quitting
        self.main_brain.service_pending() # may call OnNewCamera, OnOldCamera, etc
        if self.current_page == 'preview':
            for cam_id in self.cameras.keys():
                self.main_brain.request_image_async(cam_id)

                cam = self.cameras[cam_id]
                camPanel = cam['camPanel']
                image = None
                show_fps = None
                try:
                    image, show_fps = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
                except KeyError:
                    # unexpected disconnect
                    pass # may have lost camera since call to service_pending
                if image is not None:
                    self.cam_image_canvas.update_image(cam_id,image)
                if show_fps is not None:
                    show_fps_label = XRCCTRL(camPanel,'acquired_fps_label') # get container
                    show_fps_label.SetLabel('fps: %.1f'%show_fps)
            self.cam_image_canvas.OnDraw()

        
            if sys.platform != 'win32' or isinstance(event,wxIdleEventPtr):
                event.RequestMore()
            
        else:
            # do other stuff
            pass
            
    def OnNewCamera(self, cam_id, scalar_control_info):
        # add self to WX
        camPanel = RES.LoadPanel(self.all_cam_panel,"PerCameraPanel")
        acp_box = self.all_cam_panel.GetSizer()
        acp_box.Add(camPanel,1,wxEXPAND | wxALL,border=10)

        # set staticbox label
        box = camPanel.GetSizer()
        static_box = box.GetStaticBox()
        static_box.SetLabel( cam_id.split(':')[0] )

        quit_camera = XRCCTRL(camPanel,"quit_camera") # get container
        EVT_BUTTON(quit_camera, quit_camera.GetId(), self.OnCloseCamera)
        self.wx_id_2_cam_id.update( {quit_camera.GetId():cam_id} )
        
        per_cam_controls_panel = XRCCTRL(camPanel,"PerCameraControlsContainer") # get container
        box = wxBoxSizer(wxVERTICAL)

        for param in scalar_control_info.keys():
            current_value, min_value, max_value = scalar_control_info[param]
            scalarPanel = RES.LoadPanel(per_cam_controls_panel,"ScalarControlPanel") # frame main panel
            box.Add(scalarPanel,1,wxEXPAND)
            
            label = XRCCTRL(scalarPanel,'scalar_control_label')
            label.SetLabel( param )
            
            slider = XRCCTRL(scalarPanel,'scalar_control_slider')
            #slider.SetToolTip(wxToolTip('adjust %s'%param))
            slider.SetRange( min_value, max_value )
            slider.SetValue( current_value )
            
            class ParamSliderHelper:
                def __init__(self, name, cam_id, slider, main_brain,label_if_shutter=None):
                    self.name=name
                    self.cam_id=cam_id
                    self.slider=slider
                    self.main_brain=main_brain
                    self.label_if_shutter=label_if_shutter
                def onScroll(self, event):
                    current_value = self.slider.GetValue()
                    self.main_brain.send_set_camera_property(
                        self.cam_id,self.name,current_value)
                    if self.label_if_shutter is not None: # this is the shutter control
                        self.label_if_shutter.SetLabel('Exposure (msec): %.3f'%(current_value*0.02,))

            if param.lower() == 'shutter':
                label_if_shutter = XRCCTRL(camPanel,'exposure_label') # get label
                label_if_shutter.SetLabel('Exposure (msec): %.3f'%(current_value*0.02,))
            else:
                label_if_shutter = None
            psh = ParamSliderHelper(param,cam_id,slider,self.main_brain,label_if_shutter)
            EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
      
        per_cam_controls_panel.SetSizer(box)
        self.all_cam_panel.Layout()

        # bookkeeping
        self.cameras[cam_id] = {'scalar_control_info':scalar_control_info,
                                'camPanel':camPanel,
                                }
        # XXX should tell self.cam_image_canvas


        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        orig_selection = calibration_cam_choice.GetStringSelection()
        cam_list = [calibration_cam_choice.GetString(i) for i in
                     range(calibration_cam_choice.GetCount())]
        
        while not calibration_cam_choice.IsEmpty():
            calibration_cam_choice.Delete(0)
            
        cam_list.append(cam_id)
        cam_list.sort()
        cam_list.reverse()
        for cam_id in cam_list:
            calibration_cam_choice.Insert(cam_id,0,'')
        if orig_selection != '':
            calibration_cam_choice.SetStringSelection(orig_selection)

        calibration_cam_choice.GetParent().GetSizer.Fit()
            
        self.update_wx()

    def OnCloseCamera(self, event):
        cam_id = self.wx_id_2_cam_id[event.GetId()]
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera
    
    def OnOldCamera(self, cam_id):
        print 'a camera was unregistered:',cam_id
        try:
            self.cam_image_canvas.delete_image(cam_id)
        
        except KeyError:
            # camera never sent frame??
            pass
        camPanel=self.cameras[cam_id]['camPanel']
        camPanel.DestroyChildren()
        camPanel.Destroy()

        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        i=calibration_cam_choice.FindString(cam_id)
        calibration_cam_choice.Delete(i)
        
        del self.cameras[cam_id]
        
        self.update_wx()
    
def main():
    # initialize GUI
    #app = App(redirect=1,filename='flydra_log.txt')
    app = App() 
    
    # create main_brain server (not started yet)
    main_brain = MainBrain()

    try:
        # connect server to GUI
        app.attach_and_start_main_brain(main_brain)

        # hand control to GUI
        app.MainLoop()
        del app

    finally:
        # stop main_brain server
        main_brain.quit()
        print 'done'
    
if __name__ == '__main__':
    main()
