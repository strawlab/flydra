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
from wxPython.lib.scrolledpanel import wxScrolledPanel
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
        frame = wxFrame(None, -1, "Flydra Main Brain",size=(1000,700))

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
                        "Tints clipped pixels blue", wxITEM_CHECK)
        EVT_MENU(self, ID_toggle_image_tinting, self.OnToggleTint)
        menuBar.Append(viewmenu, "&View")

        # finish menubar -----------------------------
        frame.SetMenuBar(menuBar)

        # main panel ----------------------------------
        self.main_panel = RES.LoadPanel(frame,"APP_PANEL") # make frame main panel
        self.main_panel.SetFocus()

        frame_box = wxBoxSizer(wxVERTICAL)
        frame_box.Add(self.main_panel,1,wxEXPAND)
        frame.SetSizer(frame_box)
        frame.Layout()

        nb = XRCCTRL(self.main_panel,"MAIN_NOTEBOOK")

        # setup notebook pages
        
        self.cam_preview_panel = RES.LoadPanel(nb,"PREVIEW_PANEL")
        nb.AddPage(self.cam_preview_panel,"Camera Preview/Settings")
        self.InitPreviewPanel()
        
        viewmenu.Check(ID_toggle_image_tinting,self.cam_image_canvas.get_clipping())
        
        self.calibration_panel = RES.LoadPanel(nb,"CALIBRATION_PANEL")
        nb.AddPage(self.calibration_panel,"Calibration")
        self.InitCalibrationPanel()
        
        self.record_raw_panel = RES.LoadPanel(nb,"RECORD_RAW_PANEL")
        nb.AddPage(self.record_raw_panel,"Record raw video")
        self.InitRecordRawPanel()

        temp_panel = RES.LoadPanel(nb,"UNDER_CONSTRUCTION_PANEL")
        nb.AddPage(temp_panel,"Realtime 3D tracking")
        
        EVT_NOTEBOOK_PAGE_CHANGED(nb,nb.GetId(),self.OnPageChanged)
        self.main_notebook = nb
        self.current_page = 'preview'

        # finalize wx stuff

        frame.SetAutoLayout(true)

        frame.Show()
        self.SetTopWindow(frame)
        self.frame = frame

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
        elif page==1:
            self.current_page = 'calibration'
            #self.OnEnterCalibrationPage()
        elif page==2:
            self.current_page = 'record'
        else:
            self.current_page = 'unknown'

    def InitPreviewPanel(self):
        # setup dynamic image (OpenGL) panel
        dynamic_image_panel = XRCCTRL(self.cam_preview_panel,"PreviewDynamicImagePanel") # get container
        box = wxBoxSizer(wxVERTICAL)
        dynamic_image_panel.SetSizer(box)
        
        self.cam_image_canvas = DynamicImageCanvas.DynamicImageCanvas(dynamic_image_panel,-1) # put GL window in container
        box.Add(self.cam_image_canvas,1,wxEXPAND)
        dynamic_image_panel.Layout()

        # setup per-camera container panel
        container = XRCCTRL(self.cam_preview_panel,"PreviewPerCamPanel")
        sizer = container.GetSizer()

        scrolled_container = wxScrolledPanel(container,-1)
        sizer.Add(scrolled_container,1,wxEXPAND)
        
        ##
        sizer = wxBoxSizer(wxHORIZONTAL)
        scrolled_container.SetSizer(sizer)
        scrolled_container.SetAutoLayout(1)
        scrolled_container.SetupScrolling()
        self.preview_per_cam_scrolled_container = scrolled_container

        ###

        self.preview_per_cam_scrolled_container.Layout()

    def PreviewPerCamInit(self,cam_id):
        scalar_control_info=self.cameras[cam_id]['scalar_control_info']
        
        # add self to WX
        PreviewPerCamPanel = RES.LoadPanel(self.preview_per_cam_scrolled_container,"preview_per_cam_panel")
        acp_box = self.preview_per_cam_scrolled_container.GetSizer()
        acp_box.Add(PreviewPerCamPanel,0,wxEXPAND | wxALL,border=10)

        # set staticbox label
        box = PreviewPerCamPanel.GetSizer()
        static_box = box.GetStaticBox()
        static_box.SetLabel( cam_id )

        quit_camera = XRCCTRL(PreviewPerCamPanel,"quit_camera") # get container
        EVT_BUTTON(quit_camera, quit_camera.GetId(), self.OnCloseCamera)
        self.wx_id_2_cam_id.update( {quit_camera.GetId():cam_id} )
        
        per_cam_controls_panel = XRCCTRL(PreviewPerCamPanel,"PerCameraControlsContainer") # get container
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
                label_if_shutter = XRCCTRL(PreviewPerCamPanel,'exposure_label') # get label
                label_if_shutter.SetLabel('Exposure (msec): %.3f'%(current_value*0.02,))
            else:
                label_if_shutter = None
            psh = ParamSliderHelper(param,cam_id,slider,self.main_brain,label_if_shutter)
            EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
      
        per_cam_controls_panel.SetSizer(box)
        self.preview_per_cam_scrolled_container.Layout()
        self.cameras[cam_id]['PreviewPerCamPanel']=PreviewPerCamPanel

    def PreviewPerCamClose(self,cam_id):
        PreviewPerCamPanel=self.cameras[cam_id]['PreviewPerCamPanel']
        PreviewPerCamPanel.DestroyChildren()
        PreviewPerCamPanel.Destroy()

    def InitCalibrationPanel(self):
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        find_chessboard_button = XRCCTRL(self.calibration_panel,
                                         "find_chessboard_button")
        fns_chessboard_button = XRCCTRL(self.calibration_panel,
                                        "find_and_save_chessboard_button")
        EVT_BUTTON(find_chessboard_button, find_chessboard_button.GetId(),
                   self.OnFindChessboard)
        EVT_BUTTON(fns_chessboard_button, fns_chessboard_button.GetId(),
                   self.OnFindAndSaveChessboard)
        calibration_plot = XRCCTRL(self.calibration_panel,"calibration_plot")
        sizer = wxBoxSizer(wxVERTICAL)
        
        # matplotlib panel itself
        self.plotpanel = PlotPanel(calibration_plot)
        self.plotpanel.init_plot_data()
        
        # wx boilerplate
        sizer.Add(self.plotpanel, 1, wxEXPAND)
        calibration_plot.SetSizer(sizer)
        #calibration_plot.Fit()

        # lower panel container --------------
        container = XRCCTRL(self.calibration_panel,
                            "calib_intrinsic_per_cam_container")
        sizer = container.GetSizer()

        scrolled_container = wxScrolledPanel(container,-1)
        sizer.Add(scrolled_container,1,wxEXPAND)

        ##
        sizer = wxBoxSizer(wxVERTICAL)
        scrolled_container.SetSizer(sizer)
        scrolled_container.SetAutoLayout(1)
        scrolled_container.SetupScrolling(scroll_x=False)
        self.calibration_per_cam_scrolled_container = scrolled_container

    def CalibrationPerCamInit(self,cam_id):
        # Choice control
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        orig_selection = calibration_cam_choice.GetStringSelection()
        cam_list = [calibration_cam_choice.GetString(i) for i in
                     range(calibration_cam_choice.GetCount())]
        
        while not 0==calibration_cam_choice.GetCount():
            calibration_cam_choice.Delete(0)
            
        cam_list.append(cam_id)
        cam_list.sort()
        for tmp_cam_id in cam_list:
            calibration_cam_choice.Append(tmp_cam_id)
        if orig_selection != '':
            calibration_cam_choice.SetStringSelection(orig_selection)
        else:
            calibration_cam_choice.SetStringSelection(cam_id)

        calibration_cam_choice.GetParent().GetSizer().Layout()

        #  per cameral panel
        container = self.calibration_per_cam_scrolled_container
        sizer = container.GetSizer()
        panel = RES.LoadPanel(container,
                              "calib_intrinsic_per_cam_panel")
        panel.GetSizer().GetStaticBox().SetLabel(cam_id )
        listbox = XRCCTRL(panel,"calib_intrinsic_listbox")
        self.cameras[cam_id]['calib_intrinsic_listbox']=listbox
        sizer.Add(panel,0,wxEXPAND)

        per_cam_view = XRCCTRL(panel,"per_cam_intrinsic_view")
        EVT_BUTTON(per_cam_view, per_cam_view.GetId(),
                   self.OnCalibPerCamSavedPointsView)
        per_cam_delete = XRCCTRL(panel,"per_cam_intrinsic_delete")
        EVT_BUTTON(per_cam_delete, per_cam_delete.GetId(),
                   self.OnCalibPerCamSavedPointsDelete)

        per_cam_calc = XRCCTRL(panel,"per_cam_intrinsic_calc")
        EVT_BUTTON(per_cam_calc, per_cam_calc.GetId(),
                   self.OnCalibPerCamCalcIntrinsics)

        container.GetParent().GetSizer().Layout()

    def OnCalibPerCamCalcIntrinsics(self, event):
        parent_panel = event.GetEventObject().GetParent()
        cam_id = parent_panel.GetSizer().GetStaticBox().GetLabel()
        listbox = self.cameras[cam_id]['calib_intrinsic_listbox']
        n_images = listbox.GetCount()
        image_points = [] # list of lists
        for i in range(n_images):
            image_points.append( listbox.GetClientData(i)[1] )
        #opencv.calibrate_camera(image_points)
        try:
            raise NotImplementedError("Not implemented yet")
        except Exception,x:
            ErrorDialog.ShowErrorDialog(x)
        
    def OnCalibPerCamSavedPointsView(self, event):
        parent_panel = event.GetEventObject().GetParent()
        cam_id = parent_panel.GetSizer().GetStaticBox().GetLabel()
        listbox = self.cameras[cam_id]['calib_intrinsic_listbox']
        selection = listbox.GetSelection()
        if selection == -1:
            return
        selection_string = listbox.GetString(selection)
        frame,corners = listbox.GetClientData(selection)
        height = frame.shape[0]
        self.plotpanel.set_image(frame/255.0)
        self.plotpanel.set_data(corners[:,0],height-corners[:,1])
        self.plotpanel.draw()
        self.statusbar.SetStatusText('Viewing %s %s'%(
            cam_id,selection_string),0)
        
    def OnCalibPerCamSavedPointsDelete(self, event):
        parent_panel = event.GetEventObject().GetParent()
        cam_id = parent_panel.GetSizer().GetStaticBox().GetLabel()
        listbox = self.cameras[cam_id]['calib_intrinsic_listbox']
        selection = listbox.GetSelection()
        if selection == -1:
            return
        listbox.Delete(selection)

    def CalibrationPerCamClose(self,cam_id):
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        i=calibration_cam_choice.FindString(cam_id)
        calibration_cam_choice.Delete(i)

    def InitRecordRawPanel(self):
        record_raw_record = XRCCTRL(self.record_raw_panel,
                                    "record_raw_record")
        EVT_BUTTON(record_raw_record, record_raw_record.GetId(), self.OnRecordRaw)
        record_raw_stop = XRCCTRL(self.record_raw_panel,
                                    "record_raw_stop")
        EVT_BUTTON(record_raw_stop, record_raw_stop.GetId(), self.OnRecordRawStop)
        self._currently_recording_cams = []

    def OnRecordRaw(self, event):
        if len(self._currently_recording_cams) != 0:
            raise RuntimeError("currently recording!")
        
        cam_choice = XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
        filename_text_entry = XRCCTRL(self.record_raw_panel,
                                      "record_raw_filename")
        filename = filename_text_entry.GetValue()
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))

        try:
            for cam_id in cam_ids:
                self.main_brain.start_recording(cam_id,filename)
                self._currently_recording_cams.append(cam_id)
            self.statusbar.SetStatusText('Recording started',0)
        except:
            self.statusbar.SetStatusText('Failed to start recording: see console',0)
            raise

    def OnRecordRawStop(self, event):
        if not len(self._currently_recording_cams):
            self.statusbar.SetStatusText('Not recording - cannot stop',0)
            return

        try:
            for cam_id in self._currently_recording_cams[:]:
                self.main_brain.stop_recording(cam_id)
                self._currently_recording_cams.remove(cam_id)
            self.statusbar.SetStatusText('Recording stopped',0)
        except:
            self.statusbar.SetStatusText('Failed to stop recording: see console',0)
            raise

    def RecordRawPerCamInit(self,cam_id):
        # Choice control
        cam_choice = XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
        cam_list = []
        for i in range(cam_choice.GetCount()):
            string_val =cam_choice.GetString(i)
            check_val = cam_choice.IsChecked(i)
            client_data = cam_choice.GetClientData(i)
            cam_list.append( (string_val,check_val,client_data) )
        
        while not 0==cam_choice.GetCount():
            cam_choice.Delete(0)

        new_string_val = cam_id
        new_check_val = True
        new_client_data = None
        cam_list.append((new_string_val,
                         new_check_val,
                         new_client_data))
        #cam_list.sort()
        for i in range(len(cam_list)):
            string_val, check_val, client_data=cam_list[i]
            cam_choice.Append(string_val,client_data)
            cam_choice.Check(i,check_val)
            
        cam_choice.GetParent().GetSizer().Layout()

    def RecordRawPerCamClose(self,cam_id):
        pass
    
    def OnFindAndSaveChessboard(self,event):
        frame, corners = self.OnFindChessboard(event,return_find=True)
        if len(corners.shape)!=2:
            self.statusbar.SetStatusText('No corners found, data not saved',0)
            return
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        cam_id = calibration_cam_choice.GetStringSelection()
        if cam_id == '':
            return
        listbox=self.cameras[cam_id]['calib_intrinsic_listbox']
        quick_string = '%d pts %s'%(len(corners),
                                    time.strftime('%H:%M:%S'))
        listbox.Append(quick_string,(frame,corners))
        
    def OnFindChessboard(self,event,return_find=False):
        calibration_cam_choice = XRCCTRL(self.calibration_panel,
                                         "calibration_cam_choice")
        cam_id = calibration_cam_choice.GetStringSelection()
        if cam_id == '':
            return
        frame = self.main_brain.get_image_sync(cam_id)
        height = frame.shape[0]
        self.plotpanel.set_image(frame/255.0)

        etalon_width = int(XRCCTRL(self.calibration_panel,
                                   "etalon_width").GetValue())
        etalon_height = int(XRCCTRL(self.calibration_panel,
                                    "etalon_height").GetValue())
        etalon_size = etalon_width, etalon_height
        found_all,corners = opencv.find_corners(frame,etalon_size)

        status_string = 'Found %d corners, '%len(corners)
        if found_all:
            status_string += 'OpenCV thinks it found all.'
        else:
            status_string += 'OpenCV does not think it found all.'
        self.statusbar.SetStatusText(status_string,0)
        if len(corners.shape)==2:
            self.plotpanel.set_data(corners[:,0],height-corners[:,1])
        else:
            self.plotpanel.set_data([],[])#0,0,0],[0,0,0])
        self.plotpanel.draw()
        if return_find:
            return frame, corners
            
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
                if not cam.has_key('PreviewPerCamPanel'):
                    # not added yet
                    continue
                PreviewPerCamPanel = cam['PreviewPerCamPanel']
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
                    show_fps_label = XRCCTRL(PreviewPerCamPanel,'acquired_fps_label') # get container
                    show_fps_label.SetLabel('fps: %.1f'%show_fps)
            self.cam_image_canvas.OnDraw()

        
            if sys.platform != 'win32' or isinstance(event,wxIdleEventPtr):
                event.RequestMore()
            
        else:
            # do other stuff
            pass
            
    def OnNewCamera(self, cam_id, scalar_control_info):
        # bookkeeping
        self.cameras[cam_id] = {'scalar_control_info':scalar_control_info,
                                }

        # XXX should tell self.cam_image_canvas
        self.PreviewPerCamInit(cam_id)
        self.CalibrationPerCamInit(cam_id)
        self.RecordRawPerCamInit(cam_id)
        self.update_wx()

    def OnCloseCamera(self, event):
        cam_id = self.wx_id_2_cam_id[event.GetId()]
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera
    
    def OnOldCamera(self, cam_id):
        try:
            self.cam_image_canvas.delete_image(cam_id)
        
        except KeyError:
            # camera never sent frame??
            pass

        self.PreviewPerCamClose(cam_id)
        self.CalibrationPerCamClose(cam_id)
        self.RecordRawPerCamClose(cam_id)
        
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
    
if __name__ == '__main__':
    main()
