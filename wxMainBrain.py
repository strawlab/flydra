#!/usr/bin/env python
# $Id$

# TODO:
#
# hotkeys for
#   running average on all cameras (DONE)
#   take bg image on all cameras (DONE)
#   clean bg image on all cameras (DONE)
#   record data (auto filename) (DONE)
#   record movie (auto filename)
# make sound when tracking (DONE)
# toggle for save movie only when point found
# toggle for external trigger (DONE)

# In other files:
#  don't crash camera servers when hard-drives not mounted
#  use ROI surrounding maximum point (DONE)
#  save clear_threshold and diff_threshold camera settings (really?)

import sys
import threading
import time
import os
import copy
import MainBrain
from MatplotlibPanel import PlotPanel
import DynamicImageCanvas
from wxPython.wx import *
from wxPython.lib.scrolledpanel import wxScrolledPanel
from wxPython.xrc import *
import ErrorDialog

RESDIR = os.path.split(os.path.abspath(sys.argv[0]))[0]
RESFILE = os.path.join(RESDIR,'flydra_server.xrc')
hydra_image_file = os.path.join(RESDIR,'hydra.gif')
RES = wxXmlResource(RESFILE)
DETECT_SND = wxSound(os.path.join(RESDIR,'detect.wav'))
DETECT_SND.Play()

class App(wxApp):
    def OnInit(self,*args,**kw):
        wxInitAllImageHandlers()
        frame = wxFrame(None, -1, "Flydra Main Brain",size=(650,700))

        # statusbar ----------------------------------
        self.statusbar = frame.CreateStatusBar()
        self.statusbar.SetFieldsCount(4)
        
        # menubar ------------------------------------
        menuBar = wxMenuBar()
        #   File
        filemenu = wxMenu()
        
        ID_open_cam_config = wxNewId()
        filemenu.Append(ID_open_cam_config, "Open Camera Configuration...\tCtrl-O")
        EVT_MENU(self, ID_open_cam_config, self.OnOpenCamConfig)
        
        ID_save_cam_config = wxNewId()
        filemenu.Append(ID_save_cam_config, "Save Camera Configuration...\tCtrl-S")
        EVT_MENU(self, ID_save_cam_config, self.OnSaveCamConfig)

        filemenu.AppendItem(wxMenuItem(kind=wxITEM_SEPARATOR))
        
        ID_start_calibration = wxNewId()
        filemenu.Append(ID_start_calibration, "Start calibration...", "Start saving calibration points")
        EVT_MENU(self, ID_start_calibration, self.OnStartCalibration)
        
        filemenu.AppendItem(wxMenuItem(kind=wxITEM_SEPARATOR))
        
        ID_quit = wxNewId()
        filemenu.Append(ID_quit, "Quit\tCtrl-Q", "Quit application")
        EVT_MENU(self, ID_quit, self.OnQuit)
        
        menuBar.Append(filemenu, "&File")

        #   Data logging
        data_logging_menu = wxMenu()

        ID_start_saving_data = wxNewId()
        data_logging_menu.Append(ID_start_saving_data, "Start saving data...",
                                 "Collect & save all 2D and 3D data")
        EVT_MENU(self, ID_start_saving_data, self.OnStartSavingData)

        ID_stop_saving_data = wxNewId()
        data_logging_menu.Append(ID_stop_saving_data, "Stop saving data",
                                 "Stop saving data")
        EVT_MENU(self, ID_stop_saving_data, self.OnStopSavingData)

        menuBar.Append(data_logging_menu, "Data &Logging")
        
        #   Cameras
        cammenu = wxMenu()
        
        ID_fake_sync = wxNewId()
        cammenu.Append(ID_fake_sync, "Fake synchronization")
        EVT_MENU(self, ID_fake_sync, self.OnFakeSync)

        # XXX not finished
        #ID_stop_all_collecting_bg = wxNewId()
        #cammenu.Append(ID_stop_all_collecting_bg, "Stop all running background collection")
        #EVT_MENU(self, ID_stop_all_collecting_bg, self.OnStopAllCollectingBg)
        
        menuBar.Append(cammenu, "&Cameras")

        #   View
        viewmenu = wxMenu()
        ID_debug_cameras = wxNewId()
        viewmenu.Append(ID_debug_cameras, "Camera debug mode",
                        "Enter camera debug mode", wxITEM_CHECK)
        EVT_MENU(self, ID_debug_cameras, self.OnToggleDebugCameras)

        ID_toggle_image_tinting = wxNewId()
        viewmenu.Append(ID_toggle_image_tinting, "Tint clipped data",
                        "Tints clipped pixels green", wxITEM_CHECK)
        EVT_MENU(self, ID_toggle_image_tinting, self.OnToggleTint)

        ID_draw_points = wxNewId()
        viewmenu.Append(ID_draw_points, "Draw points",
                        "Draw 2D points and orientation", wxITEM_CHECK)
        EVT_MENU(self, ID_draw_points, self.OnToggleDrawPoints)

        ID_set_timer = wxNewId()
        viewmenu.Append(ID_set_timer, "Set update timer...",
                        "Sets interval at which display is updated")
        EVT_MENU(self, ID_set_timer, self.OnSetTimer)

        ID_set_timer2 = wxNewId()
        viewmenu.Append(ID_set_timer2, "Set raw image update timer...",
                        "Sets interval at which images are updated")
        EVT_MENU(self, ID_set_timer2, self.OnSetTimer2)

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
        self.cam_preview_panel.SetAutoLayout(True)
        nb.AddPage(self.cam_preview_panel,"Camera Preview/Settings")
        self.InitPreviewPanel()
        
        viewmenu.Check(ID_toggle_image_tinting,self.cam_image_canvas.get_clipping())
        viewmenu.Check(ID_draw_points,self.cam_image_canvas.get_display_points())
        
        self.snapshot_panel = RES.LoadPanel(nb,"SNAPSHOT_PANEL")
        nb.AddPage(self.snapshot_panel,"Snapshot")
        self.InitSnapshotPanel()
        
        self.record_raw_panel = RES.LoadPanel(nb,"RECORD_RAW_PANEL")
        nb.AddPage(self.record_raw_panel,"Record raw video")
        self.InitRecordRawPanel()

        self.tracking_panel = RES.LoadPanel(nb,"REALTIME_TRACKING_PANEL")
        nb.AddPage(self.tracking_panel,"Realtime 3D tracking")
        self.InitTrackingPanel()
        
        #temp_panel = RES.LoadPanel(nb,"UNDER_CONSTRUCTION_PANEL")
        #nb.AddPage(temp_panel,"Under construction")
        
        EVT_NOTEBOOK_PAGE_CHANGED(nb,nb.GetId(),self.OnPageChanged)
        self.main_notebook = nb
        self.current_page = 'preview'

        # finalize wx stuff

        frame.SetAutoLayout(true)

        frame.Show()
        self.SetTopWindow(frame)
        self.frame = frame

        ID_Timer  = wxNewId() 	         
        self.timer = wxTimer(self,      # object to send the event to 	 
                             ID_Timer)  # event id to use 	 
        EVT_TIMER(self,  ID_Timer, self.OnIdle)
        self.update_interval=100
        self.timer.Start(self.update_interval) # call every n msec
##        EVT_IDLE(self.frame, self.OnIdle)

        # raw image update timer
        ID_Timer2  = wxNewId() 	         
        self.timer2 = wxTimer(self,      # object to send the event to 	 
                              ID_Timer2)  # event id to use 	 
        EVT_TIMER(self,  ID_Timer2, self.OnUpdateRawImages)
        self.update_interval2=2000
        self.timer2.Start(self.update_interval2) # call every n msec

        self.cameras = {} #OrderedDict()

        self.Bind(EVT_KEY_DOWN, self.OnKeyDown)

        self.update_wx()

        self.collecting_background_buttons = {}
        self.take_background_buttons = {}
        self.clear_background_buttons = {}
        return True

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if not (27 < keycode < 256):
            # propagate event up the chain...
            event.Skip()
            return
        
        keyname = "%s" % chr(keycode)
        if keyname == 'R':
            for cam_id in self.cameras.keys():
                widget = self.collecting_background_buttons[cam_id]
                widget.SetValue( not widget.GetValue() )
                id = widget.GetId()
                event = wxCommandEvent(EVT_CHECKBOX.evtType[0],id)
                event.SetEventObject( widget )
                widget.Command( event )
            self.statusbar.SetStatusText('running BG collection toggled',0)
        elif keyname == 'S':
            if str(self.statusbar.GetStatusText(2)) == '':
                self.OnStartSavingData()
            else:
                self.OnStopSavingData()
        elif keyname == 'T':
            for cam_id in self.cameras.keys():
                widget = self.take_background_buttons[cam_id]
                id = widget.GetId()
                event = wxCommandEvent(EVT_BUTTON.evtType[0],id)
                event.SetEventObject( widget )
                widget.Command( event )
            self.statusbar.SetStatusText('took BG images',0)
        elif keyname == 'C':
            for cam_id in self.cameras.keys():
                widget = self.clear_background_buttons[cam_id]
                id = widget.GetId()
                event = wxCommandEvent(EVT_BUTTON.evtType[0],id)
                event.SetEventObject( widget )
                widget.Command( event )
            self.statusbar.SetStatusText('cleared BG images',0)
        else:
            # propagate event up the chain...
            event.Skip()
        
    def OnPageChanged(self, event):
        page = event.GetSelection()
        self.statusbar.SetStatusText('',0)
        if page==0:
            self.current_page = 'preview'
        elif page==1:
            self.current_page = 'snapshot'
        elif page==2:
            self.current_page = 'record'
        elif page==3:
            self.current_page = 'tracking'
        else:
            self.current_page = 'unknown'

    def InitPreviewPanel(self):
        # setup dynamic image (OpenGL) panel
        dynamic_image_panel = XRCCTRL(self.cam_preview_panel,"PreviewDynamicImagePanel") # get container
        box = wxBoxSizer(wxVERTICAL)
        dynamic_image_panel.SetSizer(box)
        
        self.cam_image_canvas = DynamicImageCanvas.DynamicImageCanvas(dynamic_image_panel,-1)
        box.Add(self.cam_image_canvas,1,wxEXPAND)
        dynamic_image_panel.Layout()

        # setup per-camera container panel
        container = XRCCTRL(self.cam_preview_panel,"PreviewPerCamPanel")
        sizer = wxBoxSizer(wxHORIZONTAL)
        container.SetSizer(sizer)

        if 1:
            scrolled_container = wxScrolledPanel(container,-1)
        else:
            scrolled_container = wxPanel(container,-1)
        sizer.Add(scrolled_container,1,wxEXPAND)
        
        sizer = wxBoxSizer(wxHORIZONTAL)
        scrolled_container.SetSizer(sizer)
        #scrolled_container.SetAutoLayout(True)
        if isinstance(scrolled_container,wxScrolledPanel):
            scrolled_container.SetupScrolling()
        self.preview_per_cam_scrolled_container = scrolled_container

        self.preview_per_cam_scrolled_container.Layout()

    def PreviewPerCamInit(self,cam_id):
        scalar_control_info=self.cameras[cam_id]['scalar_control_info']
        
        # add self to WX
        previewPerCamPanel = RES.LoadPanel(self.preview_per_cam_scrolled_container,
                                           "preview_per_cam_panel")
        acp_box = self.preview_per_cam_scrolled_container.GetSizer()
        all_cams = self.cameras.keys()
        all_cams.sort()
        cam_number = all_cams.index(cam_id)
        acp_box.Insert(cam_number,previewPerCamPanel,1,wxEXPAND | wxALL,border=3)

        # set staticbox label
        box = previewPerCamPanel.GetSizer()
        static_box = box.GetStaticBox()
        static_box.SetLabel( cam_id )

        collecting_background = XRCCTRL(previewPerCamPanel,"COLLECTING_BACKGROUND")
        self.collecting_background_buttons[cam_id] = collecting_background
        EVT_CHECKBOX(collecting_background, collecting_background.GetId(),
                     self.OnCollectingBackground)
        
        take_background = XRCCTRL(previewPerCamPanel,"take_background")
        self.take_background_buttons[cam_id] = take_background
        EVT_BUTTON(take_background, take_background.GetId(),
                   self.OnTakeBackground)
        
        clear_background = XRCCTRL(previewPerCamPanel,"clear_background")
        self.clear_background_buttons[cam_id] = clear_background
        EVT_BUTTON(clear_background, clear_background.GetId(),
                   self.OnClearBackground)
        
        find_Rcenter = XRCCTRL(previewPerCamPanel,"find_Rcenter")
        EVT_BUTTON(find_Rcenter, find_Rcenter.GetId(),
                   self.OnFindRCenter)
        
        set_roi = XRCCTRL(previewPerCamPanel,"set_roi")
        EVT_BUTTON(set_roi, set_roi.GetId(), self.OnSetROI)

        quit_camera = XRCCTRL(previewPerCamPanel,"quit_camera")
        EVT_BUTTON(quit_camera, quit_camera.GetId(), self.OnCloseCamera)

        threshold_value = XRCCTRL(previewPerCamPanel,"threshold_value")
        val = scalar_control_info['diff_threshold']
        threshold_value.SetValue( str( val ) )
        EVT_TEXT(threshold_value, threshold_value.GetId(), self.OnSetCameraThreshold)

        threshold_clear_value = XRCCTRL(previewPerCamPanel,"threshold_clear_value")
        val = scalar_control_info['clear_threshold']
        threshold_clear_value.SetValue( str( val ) )
        EVT_TEXT(threshold_clear_value, threshold_clear_value.GetId(),
                 self.OnSetCameraClearThreshold)

        arena_control = XRCCTRL(previewPerCamPanel,
                             "ARENA_CONTROL")
        EVT_CHECKBOX(arena_control, arena_control.GetId(), self.OnArenaControl)
        
        ext_trig = XRCCTRL(previewPerCamPanel,
                           "EXT_TRIG")
        val = scalar_control_info['trigger_source']
        ext_trig.SetValue( val )
        EVT_CHECKBOX(ext_trig, ext_trig.GetId(), self.OnExtTrig)
        
        per_cam_controls_panel = XRCCTRL(previewPerCamPanel,
                                         "PerCameraControlsContainer")
        grid = wxFlexGridSizer(0,3,0,0) # 3 columns
        grid.AddGrowableCol(2)

        params = scalar_control_info.keys()
        params.sort()
        params.reverse()
        for param in params:
            if param not in ['shutter','gain','brightness']:
                continue
            current_value, min_value, max_value = scalar_control_info[param]
            grid.Add( wxStaticText(per_cam_controls_panel,wxNewId(),param),
                     0,wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL )
            
            txtctrl = wxTextCtrl( per_cam_controls_panel, wxNewId(),
                                  size=(40,20))
            txtctrl.SetValue(str(current_value))
            grid.Add( txtctrl,0,wxALIGN_LEFT )
            
            slider = wxSlider( per_cam_controls_panel, wxNewId(),
                               current_value, min_value, max_value,
                               style= wxSL_HORIZONTAL )
            grid.Add( slider,1,wxEXPAND )
            
            class ParamSliderHelper:
                def __init__(self, name, cam_id, txtctrl, slider,
                             main_brain,label_if_shutter=None):
                    self.name=name
                    self.cam_id=cam_id
                    self.txtctrl=txtctrl
                    self.slider=slider
                    self.main_brain=main_brain
                    self.label_if_shutter=label_if_shutter
                def update(self, value):
                    self.main_brain.send_set_camera_property(
                        self.cam_id,self.name,value)
                    if self.label_if_shutter is not None:
                        # this is the shutter control
                        self.label_if_shutter.SetLabel(
                            'Exposure (msec): %.1f'%(value*0.02,))
                def onScroll(self, event):
                    current_value = self.slider.GetValue()
                    self.txtctrl.SetValue(str(current_value))
                    self.update(current_value)
                def onText(self, event):
                    val = self.txtctrl.GetValue()
                    if val == '':
                        return
                    current_value = int(val)
                    if (current_value >= self.slider.GetMin() and
                        current_value <= self.slider.GetMax()):
                        self.slider.SetValue(current_value)
                        self.update(current_value)
                def external_set(self,new_value):
                    self.slider.SetValue(new_value)
                    self.txtctrl.SetValue(str(new_value))
                    self.update(new_value)

            if param.lower() == 'shutter':
                label_if_shutter = XRCCTRL(previewPerCamPanel,
                                           'exposure_label') # get label
                label_if_shutter.SetLabel(
                    'Exposure (msec): %.3f'%(current_value*0.02,))
            else:
                label_if_shutter = None
            psh = ParamSliderHelper(param,cam_id,txtctrl,slider,
                                    self.main_brain,label_if_shutter)
            EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
            EVT_TEXT(txtctrl, txtctrl.GetId(), psh.onText)
            self.cameras[cam_id]['sliderHelper_%s'%param]=psh
      
        per_cam_controls_panel.SetSizer(grid)
        self.preview_per_cam_scrolled_container.Layout()
        self.cameras[cam_id]['previewPerCamPanel']=previewPerCamPanel
        self.cam_preview_panel.Layout()

    def PreviewPerCamUpdateSetting(self,cam_id,property_name,value):
        previewPerCamPanel=self.cameras[cam_id]['previewPerCamPanel']
        param = property_name
        if param in ['shutter','gain','brightness']:
            current_value, min_value, max_value = value
            psh = self.cameras[cam_id]['sliderHelper_%s'%param]
            psh.external_set(current_value)
        elif param == 'clear_threshold':
            threshold_clear_value = XRCCTRL(previewPerCamPanel,"threshold_clear_value")
            threshold_clear_value.SetValue( str( value ) )
        elif param == 'diff_threshold':
            threshold_diff_value = XRCCTRL(previewPerCamPanel,"threshold_value")
            threshold_diff_value.SetValue( str( value ) )
        elif param == 'trigger_source':
            ext_trig = XRCCTRL(previewPerCamPanel,
                               "EXT_TRIG")
            ext_trig.SetValue( value )
        else:
            if param not in ('roi','width','height'):
                print 'WARNING: could not update panel display for',param
        
    def PreviewPerCamClose(self,cam_id):
        previewPerCamPanel=self.cameras[cam_id]['previewPerCamPanel']
        previewPerCamPanel.DestroyChildren()
        previewPerCamPanel.Destroy()

    def InitSnapshotPanel(self):
        snapshot_cam_choice = XRCCTRL(self.snapshot_panel,
                                         "snapshot_cam_choice")
        snapshot_button = XRCCTRL(self.snapshot_panel,
                                  "snapshot_button")
        EVT_BUTTON(snapshot_button, snapshot_button.GetId(),
                   self.OnSnapshot)
        EVT_LISTBOX(snapshot_cam_choice, snapshot_cam_choice.GetId(),
                   self.OnSnapshot)
        EVT_LISTBOX_DCLICK(snapshot_cam_choice, snapshot_cam_choice.GetId(),
                           self.OnSnapshot)
        snapshot_plot = XRCCTRL(self.snapshot_panel,"snapshot_plot")
        sizer = wxBoxSizer(wxVERTICAL)
        
        # matplotlib panel itself
        self.plotpanel = PlotPanel(snapshot_plot)
        self.plotpanel.init_plot_data()
        
        # wx boilerplate
        sizer.Add(self.plotpanel, 1, wxEXPAND)
        snapshot_plot.SetSizer(sizer)

    def SnapshotPerCamInit(self,cam_id):
        # Choice control
        snapshot_cam_choice = XRCCTRL(self.snapshot_panel,
                                         "snapshot_cam_choice")
        orig_selection = snapshot_cam_choice.GetStringSelection()
        cam_list = [snapshot_cam_choice.GetString(i) for i in
                     range(snapshot_cam_choice.GetCount())]
        
        while not 0==snapshot_cam_choice.GetCount():
            snapshot_cam_choice.Delete(0)
            
        cam_list.append(cam_id)
        cam_list.sort()
        for tmp_cam_id in cam_list:
            if tmp_cam_id != '': # XXX workaround for weird wx 2.5 behavior
                snapshot_cam_choice.Append(tmp_cam_id)
        if orig_selection != '':
            snapshot_cam_choice.SetStringSelection(orig_selection)
        else:
            snapshot_cam_choice.SetStringSelection(cam_id)

        snapshot_cam_choice.GetParent().GetSizer().Layout()

    def SnapshotPerCamClose(self,cam_id):
        snapshot_cam_choice = XRCCTRL(self.snapshot_panel,
                                         "snapshot_cam_choice")
        i=snapshot_cam_choice.FindString(cam_id)
        snapshot_cam_choice.Delete(i)

    def InitRecordRawPanel(self):
        self.record_raw = XRCCTRL(self.record_raw_panel,
                                  "record_raw")
        EVT_CHECKBOX(self.record_raw, self.record_raw.GetId(),
                     self.OnRecordRaw)
        
        self._currently_recording_cams = []

    def OnRecordRaw(self,event):
        if self.record_raw.IsChecked():
            self.OnRecordRawStart()
        else:
            self.OnRecordRawStop()

    def OnRecordRawStart(self):
        if len(self._currently_recording_cams) != 0:
            raise RuntimeError("currently recording!")
        
        cam_choice = XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
        filename_text_entry = XRCCTRL(self.record_raw_panel,
                                      "record_raw_filename")
        raw_filename = filename_text_entry.GetValue()
        if raw_filename.endswith('.fmf'):
            bg_filename = raw_filename[:-4] + '_bg.fmf'
        else:
            bg_filename = raw_filename + '.bg.fmf'
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))
        if len(cam_ids)==0:
            record_raw = XRCCTRL(self.record_raw_panel,
                                 "record_raw")
            record_raw.SetValue(False)
            return
        try:
            for cam_id in cam_ids:
                self.main_brain.start_recording(cam_id, raw_filename, bg_filename)
                self._currently_recording_cams.append(cam_id)
            self.statusbar.SetStatusText('Recording started on %d cameras'%(
                len(self._currently_recording_cams),),0)
        except Exception, exc:
            try:
                for tmp_cam_id in self._currently_recording_cams[:]:
                    self.main_brain.stop_recording(tmp_cam_id)
                    self._currently_recording_cams.remove(tmp_cam_id)
            finally:
                record_raw = XRCCTRL(self.record_raw_panel,
                                     "record_raw")
                record_raw.SetValue(False)

                self.statusbar.SetStatusText(
                    'Failed to start recording (%s): see console'%(cam_id,),0)
                raise exc

    def OnRecordRawStop(self,warn=True):
        if warn and not len(self._currently_recording_cams):
            self.statusbar.SetStatusText('Not recording - cannot stop',0)
            return
        try:
            n_stopped = 0
            for cam_id in self._currently_recording_cams[:]:
                try:
                    self.main_brain.stop_recording(cam_id)
                except KeyError, x:
                    print '%s: %s'%(x.__class__,str(x)),
                    MainBrain.DEBUG()
                self._currently_recording_cams.remove(cam_id)
                n_stopped+=1
            self.statusbar.SetStatusText('Recording stopped on %d cameras'%(
                n_stopped,))
            record_raw = XRCCTRL(self.record_raw_panel,
                                 "record_raw")
            record_raw.SetValue(False)
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
            if string_val == '': # XXX workaround for weird wx 2.5 behavior
                continue
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
        cam_choice = XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
        i=cam_choice.FindString(cam_id)
        cam_choice.Delete(i)
    
    def InitTrackingPanel(self):
        load_cal = XRCCTRL(self.tracking_panel,
                           "LOAD_CAL")
        EVT_BUTTON(load_cal, load_cal.GetId(), self.OnLoadCal)

    def OnLoadCal(self,event):
        doit=False
        dlg = wxDirDialog( self.frame, "Select directory with calibration data",
                           style = wxDD_DEFAULT_STYLE,
                           defaultPath = os.environ.get('HOME','')
                           )
        try:
            if dlg.ShowModal() == wxID_OK:
                calib_dir = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
        if doit:
            self.main_brain.load_calibration(calib_dir)
            cal_status_check = XRCCTRL(self.tracking_panel,
                                       "CAL_STATUS_CHECK")
            cal_status_check.Enable(True)
            cal_status_check.SetValue(True)
            cal_status_check.Enable(False)

    def OnSnapshot(self,event):
        snapshot_cam_choice = XRCCTRL(self.snapshot_panel,
                                      "snapshot_cam_choice")
        cam_id = snapshot_cam_choice.GetStringSelection()
        if cam_id == '':
            return
        image, show_fps, points = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
        if image is None:
            return
        height = image.shape[0]
        self.plotpanel.set_image(image)
        self.plotpanel.set_points(points)
        self.plotpanel.draw()

    def OnFakeSync(self, event):
        print 'sending fake sync command...'
        self.main_brain.fake_synchronize()

    def OnToggleDebugCameras(self, event):
        self.main_brain.set_all_cameras_debug_mode( event.IsChecked() )

    def OnStartSavingData(self, event=None):
        display_save_filename = time.strftime( 'DATA%Y%m%d_%H%M%S.h5' )
        save_dir = os.environ.get('HOME','')
        test_dir = os.path.join( save_dir, 'ORIGINAL_DATA' )
        if os.path.exists(test_dir):
            save_dir = test_dir
        save_filename = os.path.join( save_dir, display_save_filename )
        if 1:
            try:
                self.main_brain.start_saving_data(save_filename)
                self.statusbar.SetStatusText("Saving data to '%s'"%save_filename)
                self.statusbar.SetStatusText(display_save_filename,2)
            except:
                self.statusbar.SetStatusText("Error saving data to '%s', see console"%save_filename)
                self.statusbar.SetStatusText("",2)
                raise

    def OnStopSavingData(self, event=None):
        self.main_brain.stop_saving_data()
        self.statusbar.SetStatusText("Saving stopped")        
        self.statusbar.SetStatusText("",2)
        
    def OnToggleTint(self, event):
        self.cam_image_canvas.set_clipping( event.IsChecked() )

    def OnToggleDrawPoints(self, event):
        self.cam_image_canvas.set_display_points( event.IsChecked() )
        
    def OnSetTimer(self, event):
        dlg=wxTextEntryDialog(self.frame, 'What interval should the display be updated at (msec)?',
                              'Set display update interval',str(self.update_interval))
        try:
            if dlg.ShowModal() == wxID_OK:
                self.update_interval = int(dlg.GetValue())
                self.timer.Start(self.update_interval)
        finally:
            dlg.Destroy()

    def OnSetTimer2(self, event):
        dlg=wxTextEntryDialog(self.frame, 'What interval should raw frames be grabbed (msec)?',
                              'Set display update interval',str(self.update_interval2))
        try:
            if dlg.ShowModal() == wxID_OK:
                self.update_interval2 = int(dlg.GetValue())
                self.timer2.Start(self.update_interval2)
        finally:
            dlg.Destroy()

    def OnSetROI(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        dlg = RES.LoadDialog(self.frame,"ROI_DIALOG") # make frame main panel
        
        dlg_ok = XRCCTRL(dlg,"ROI_OK")
        dlg_cam_id = XRCCTRL(dlg,"ROI_cam_id")
        dlg_cam_id.SetLabel(cam_id)

        lbrt = self.main_brain.get_roi(cam_id)
        width, height = self.main_brain.get_widthheight(cam_id)
        
        l,b,r,t = lbrt
        XRCCTRL(dlg,"ROI_LEFT").SetValue(str(l))
        XRCCTRL(dlg,"ROI_BOTTOM").SetValue(str(b))
        XRCCTRL(dlg,"ROI_RIGHT").SetValue(str(r))
        XRCCTRL(dlg,"ROI_TOP").SetValue(str(t))
        
        def OnROIOK(event):
            dlg.left = int(XRCCTRL(dlg,"ROI_LEFT").GetValue())
            dlg.right = int(XRCCTRL(dlg,"ROI_RIGHT").GetValue())
            dlg.bottom = int(XRCCTRL(dlg,"ROI_BOTTOM").GetValue())
            dlg.top = int(XRCCTRL(dlg,"ROI_TOP").GetValue())
            dlg.EndModal( wxID_OK )
        EVT_BUTTON(dlg_ok, dlg_ok.GetId(),
                   OnROIOK)
        try:
            if dlg.ShowModal() == wxID_OK:
                l,b,r,t = dlg.left,dlg.bottom,dlg.right,dlg.top
                lbrt = l,b,r,t
                if l >= r or b >= t or r >= width or t >= height:
                    raise ValueError("ROI dimensions not possible")
                self.main_brain.send_set_camera_property(cam_id,'roi',lbrt)
                self.cam_image_canvas.set_lbrt(cam_id,lbrt)
        finally:
            dlg.Destroy()

    def attach_and_start_main_brain(self,main_brain):
        self.main_brain = main_brain
        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening()

    def update_wx(self):
        self.statusbar.SetStatusText('%d camera(s)'%len(self.cameras),1)
        self.frame.Layout()

    def OnOpenCamConfig(self, event):
        doit=False
        dlg = wxFileDialog( self.frame, "Select file from which to open camera config data",
                            style = wxOPEN,
                            defaultDir = os.environ.get('HOME',''),
                            defaultFile = 'flydra_cameras.cfg',
                            wildcard = '*.cfg',
                            )
        try:
            if dlg.ShowModal() == wxID_OK:
                open_filename = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
        if doit:
            fd = open(open_filename,'rb')
            buf = fd.read()
            all_params = eval(buf)
            try:
                for cam_id, params in all_params.iteritems():
                    for property_name, value in params.iteritems():
                        self.main_brain.send_set_camera_property(cam_id,property_name,value)
                        self.PreviewPerCamUpdateSetting(cam_id,property_name,value)
                        if property_name=='roi':
                            lbrt = value
                            self.cam_image_canvas.set_lbrt(cam_id,lbrt)
            except KeyError,x:
                dlg2 = wxMessageDialog( self.frame, 'Error opening configuration data:\n'\
                                        '%s: %s'%(x.__class__,x),
                                        'Error', wxOK | wxICON_ERROR )
                try:
                    dlg2.ShowModal()
                finally:
                    dlg2.Destroy()
                    
    def OnSaveCamConfig(self, event):
        all_params = self.main_brain.get_all_params()
        doit=False
        dlg = wxFileDialog( self.frame, "Select file to save camera config data",
                            #style = wxSAVE | wxOVERWRITE_PROMPT,
                            style = wxDD_DEFAULT_STYLE,
                            defaultDir = os.environ.get('HOME',''),
                            defaultFile = 'flydra_cameras.cfg',
                            wildcard = '*.cfg',
                            )
        try:
            if dlg.ShowModal() == wxID_OK:
                save_filename = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
        if doit:
            fd = open(save_filename,'wb')
            fd.write(repr(all_params))
            fd.close()
        
    def OnStartCalibration(self, event):
        doit = False
        dlg = wxDirDialog( self.frame, "Calibration save directory",
                           style = wxDD_DEFAULT_STYLE | wxDD_NEW_DIR_BUTTON,
                           defaultPath = os.environ.get('HOME',''),
                           )
        try:
            if dlg.ShowModal() == wxID_OK:
                calib_dir = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
        if doit:
            self.main_brain.start_calibrating(calib_dir)
            dlg = wxMessageDialog( self.frame, 'Acquiring calibration points',
                                   'calibration', wxOK | wxICON_INFORMATION )
            try:
                dlg.ShowModal()
            finally:
                self.main_brain.stop_calibrating()
                dlg.Destroy()
        
    def OnQuit(self, event):
        self.timer.Stop()
        self.timer2.Stop()
        
        if hasattr(self,'main_brain'):
            self.main_brain.quit()
            del self.main_brain
        #self.frame.Close(True)
        self.frame.Destroy()
        print "why doesn't the app stop now??"

    def OnUpdateRawImages(self, event):
        if self.current_page in ['preview','snapshot']:
            for cam_id in self.cameras.keys():
                self.main_brain.request_image_async(cam_id)

    def OnIdle(self, event):
        if not hasattr(self,'main_brain'):
            return # quitting
        self.main_brain.service_pending() # may call OnNewCamera, OnOldCamera, etc
        if self.current_page in ['tracking','preview','snapshot']:
            realtime_data=MainBrain.get_best_realtime_data()
            if realtime_data is not None:
                data3d,line3d,cam_ids_used=realtime_data
                if self.current_page == 'tracking':
                    XRCCTRL(self.tracking_panel,'x_pos').SetValue('% 8.1f'%data3d[0])
                    XRCCTRL(self.tracking_panel,'y_pos').SetValue('% 8.1f'%data3d[1])
                    XRCCTRL(self.tracking_panel,'z_pos').SetValue('% 8.1f'%data3d[2])
                elif self.current_page == 'preview':
                    r=self.main_brain.reconstructor
                    for cam_id in self.cameras.keys():
                        pt,ln=r.find2d(cam_id,data3d,
                                       Lcoords=line3d,distorted=True)
                        self.cam_image_canvas.set_reconstructed_points(cam_id,([pt],[ln]))
            if self.current_page == 'preview':
                for cam_id in self.cameras.keys():
                    cam = self.cameras[cam_id]
                    if not cam.has_key('previewPerCamPanel'):
                        # not added yet
                        continue
                    previewPerCamPanel = cam['previewPerCamPanel']
                    image = None
                    show_fps = None
                    try:
                        image, show_fps, points = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
                    except KeyError:
                        # unexpected disconnect
                        pass # may have lost camera since call to service_pending
                    if image is not None:
                        self.cam_image_canvas.update_image(cam_id,image)
                    if show_fps is not None:
                        show_fps_label = XRCCTRL(previewPerCamPanel,'acquired_fps_label') # get container
                        show_fps_label.SetLabel('fps: %.1f'%show_fps)
                    self.cam_image_canvas.set_draw_points(cam_id,points)

                self.cam_image_canvas.OnDraw()

                if isinstance(event,wxIdleEventPtr):
                    event.RequestMore()
            
    def OnNewCamera(self, cam_id, scalar_control_info, fqdnport):
        # bookkeeping
        self.cameras[cam_id] = {'scalar_control_info':scalar_control_info,
                                }
        # XXX should tell self.cam_image_canvas
        self.PreviewPerCamInit(cam_id)
        self.SnapshotPerCamInit(cam_id)
        self.RecordRawPerCamInit(cam_id)
        self.update_wx()

    def _get_cam_id_for_button(self, button):
        container = button.GetParent()
        box = container.GetSizer()
        static_box = box.GetStaticBox()
        return static_box.GetLabel()

    def OnCollectingBackground(self, event):
        widget = event.GetEventObject()
        cam_id = self._get_cam_id_for_button(widget)
        self.main_brain.set_collecting_background( cam_id, widget.IsChecked() )

    def OnStopAllCollectingBg(self, event):
        # XXX not finished
        pass

    def OnTakeBackground(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.take_background(cam_id)

    def OnClearBackground(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.clear_background(cam_id)

    def OnFindRCenter(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.find_r_center(cam_id)

    def OnCloseCamera(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera
    
    def OnSetCameraThreshold(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.send_set_camera_property(cam_id,'diff_threshold',value)

    def OnSetCameraClearThreshold(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.send_set_camera_property(cam_id,'clear_threshold',value)

    def OnArenaControl(self, event):
        widget = event.GetEventObject()
        cam_id = self._get_cam_id_for_button(widget)
        self.main_brain.set_use_arena( cam_id, widget.IsChecked() )

    def OnExtTrig(self, event):
        widget = event.GetEventObject()
        cam_id = self._get_cam_id_for_button(widget)
        self.main_brain.send_set_camera_property( cam_id, 'trigger_source', widget.IsChecked() )

    def OnOldCamera(self, cam_id):
        self.OnRecordRawStop(warn=False)
        
        try:
            self.cam_image_canvas.delete_image(cam_id)
        
        except KeyError:
            # camera never sent frame??
            pass

        del self.collecting_background_buttons[cam_id]
        del self.take_background_buttons[cam_id]
        del self.clear_background_buttons[cam_id]

        self.PreviewPerCamClose(cam_id)
        self.SnapshotPerCamClose(cam_id)
        self.RecordRawPerCamClose(cam_id)
        
        del self.cameras[cam_id]
        
        self.preview_per_cam_scrolled_container.Layout()
        self.update_wx()
    
def main():
    # initialize GUI
    #app = App(redirect=1,filename='flydra_log.txt')
    app = App() 
    
    # create main_brain server (not started yet)
    main_brain = MainBrain.MainBrain()

    try:
        # connect server to GUI
        app.attach_and_start_main_brain(main_brain)

        # hand control to GUI
        app.MainLoop()
        print 'mainloop over'
        del app
    finally:
        # stop main_brain server
        main_brain.quit()
        print '2nd(?) call to quit?'
    
if __name__ == '__main__':
    main()
