#!/usr/bin/env python
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
import numarray

RESDIR = os.path.split(os.path.abspath(sys.argv[0]))[0]
RESFILE = os.path.join(RESDIR,'flydra_server.xrc')
hydra_image_file = os.path.join(RESDIR,'hydra.gif')
RES = wxXmlResource(RESFILE)

class App(wxApp):
    def OnInit(self,*args,**kw):
        wxInitAllImageHandlers()
        frame = wxFrame(None, -1, "Flydra Main Brain",size=(1100,750))

        # statusbar ----------------------------------
        self.statusbar = frame.CreateStatusBar()
        self.statusbar.SetFieldsCount(2)
        
        # menubar ------------------------------------
        menuBar = wxMenuBar()
        #   File
        filemenu = wxMenu()
        
        ID_start_calibration = wxNewId()
        filemenu.Append(ID_start_calibration, "Start calibration...", "Start saving calibration points")
        EVT_MENU(self, ID_start_calibration, self.OnStartCalibration)
        
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

        ID_set_timer = wxNewId()
        viewmenu.Append(ID_set_timer, "Set update timer",
                        "Sets interval at which display is updated")#, wxITEM_CHECK)
        EVT_MENU(self, ID_set_timer, self.OnSetTimer)

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
        
        self.snapshot_panel = RES.LoadPanel(nb,"SNAPSHOT_PANEL")
        nb.AddPage(self.snapshot_panel,"Snapshot")
        self.InitSnapshotPanel()
        
        self.record_raw_panel = RES.LoadPanel(nb,"RECORD_RAW_PANEL")
        nb.AddPage(self.record_raw_panel,"Record raw video")
        self.InitRecordRawPanel()

        self.tracking_panel = RES.LoadPanel(nb,"REALTIME_TRACKING_PANEL")
        nb.AddPage(self.tracking_panel,"Realtime 3D tracking")
        
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
        self.update_interval=500
        self.timer.Start(self.update_interval) # call every n msec
        EVT_IDLE(self.frame, self.OnIdle)

        self.cameras = {} #OrderedDict()

        self.update_wx()

        return True

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
        
        self.cam_image_canvas = DynamicImageCanvas.DynamicImageCanvas(dynamic_image_panel,-1) # put GL window in container
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
        
        ##
        sizer = wxBoxSizer(wxHORIZONTAL)
        scrolled_container.SetSizer(sizer)
        scrolled_container.SetAutoLayout(True)
        if isinstance(scrolled_container,wxScrolledPanel):
            scrolled_container.SetupScrolling()
        self.preview_per_cam_scrolled_container = scrolled_container

        ###

        self.preview_per_cam_scrolled_container.Layout()

    def PreviewPerCamInit(self,cam_id):
        scalar_control_info=self.cameras[cam_id]['scalar_control_info']
        
        # add self to WX
        PreviewPerCamPanel = RES.LoadPanel(self.preview_per_cam_scrolled_container,
                                           "preview_per_cam_panel")
        acp_box = self.preview_per_cam_scrolled_container.GetSizer()
        all_cams = self.cameras.keys()
        all_cams.sort()
        cam_number = all_cams.index(cam_id)
        acp_box.Insert(cam_number,PreviewPerCamPanel,1,wxEXPAND | wxALL,border=3)

        # set staticbox label
        box = PreviewPerCamPanel.GetSizer()
        static_box = box.GetStaticBox()
        static_box.SetLabel( cam_id )

        collect_background = XRCCTRL(PreviewPerCamPanel,"collect_background")
        EVT_BUTTON(collect_background, collect_background.GetId(),
                   self.OnCollectBackground)
        
        clear_background = XRCCTRL(PreviewPerCamPanel,"clear_background")
        EVT_BUTTON(clear_background, clear_background.GetId(),
                   self.OnClearBackground)
        
        quit_camera = XRCCTRL(PreviewPerCamPanel,"quit_camera")
        EVT_BUTTON(quit_camera, quit_camera.GetId(), self.OnCloseCamera)

        threshold_value = XRCCTRL(PreviewPerCamPanel,"threshold_value")
        threshold_value.SetValue(
            str( self.cameras[cam_id]['scalar_control_info']['threshold'] ) )
        EVT_TEXT(threshold_value, threshold_value.GetId(), self.OnSetCameraThreshold)
        
        per_cam_controls_panel = XRCCTRL(PreviewPerCamPanel,
                                         "PerCameraControlsContainer")
        grid = wxFlexGridSizer(0,3,0,0) # 3 columns
        grid.AddGrowableCol(2)

        for param in scalar_control_info.keys():
            if param == 'threshold':
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
                def change_param(self, value):
                    self.main_brain.send_set_camera_property(
                        self.cam_id,self.name,value)
                    if self.label_if_shutter is not None:
                        # this is the shutter control
                        self.label_if_shutter.SetLabel(
                            'Exposure (msec): %.3f'%(value*0.02,))
                def onScroll(self, event):
                    current_value = self.slider.GetValue()
                    self.txtctrl.SetValue(str(current_value))
                    self.change_param(current_value)
                def onText(self, event):
                    val = self.txtctrl.GetValue()
                    if val == '':
                        return
                    current_value = int(val)
                    if (current_value >= self.slider.GetMin() and
                        current_value <= self.slider.GetMax()):
                        self.slider.SetValue(current_value)
                        self.change_param(current_value)

            if param.lower() == 'shutter':
                label_if_shutter = XRCCTRL(PreviewPerCamPanel,
                                           'exposure_label') # get label
                label_if_shutter.SetLabel(
                    'Exposure (msec): %.3f'%(current_value*0.02,))
            else:
                label_if_shutter = None
            psh = ParamSliderHelper(param,cam_id,txtctrl,slider,
                                    self.main_brain,label_if_shutter)
            EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
            EVT_TEXT(txtctrl, txtctrl.GetId(), psh.onText)
      
        per_cam_controls_panel.SetSizer(grid)
        self.preview_per_cam_scrolled_container.Layout()
        self.cameras[cam_id]['PreviewPerCamPanel']=PreviewPerCamPanel
        self.cam_preview_panel.Layout()
        
    def PreviewPerCamClose(self,cam_id):
        PreviewPerCamPanel=self.cameras[cam_id]['PreviewPerCamPanel']
        PreviewPerCamPanel.DestroyChildren()
        PreviewPerCamPanel.Destroy()

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
        record_raw = XRCCTRL(self.record_raw_panel,
                             "record_raw")
        EVT_CHECKBOX(record_raw, record_raw.GetId(), self.OnRecordRaw)
        
        self._currently_recording_cams = []

    def OnRecordRaw(self,event):
        if event.IsChecked():
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
        filename = filename_text_entry.GetValue()
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))
        try:
            for cam_id in cam_ids:
                self.main_brain.start_recording(cam_id,filename)
                self._currently_recording_cams.append(cam_id)
            self.statusbar.SetStatusText('Recording started on %d cameras'%(
                len(self._currently_recording_cams),),0)
        except Exception,x:
            try:
                for tmp_cam_id in self._currently_recording_cams[:]:
                    self.main_brain.stop_recording(tmp_cam_id)
                    self._currently_recording_cams.remove(tmp_cam_id)
            finally:
                self.statusbar.SetStatusText(
                    'Failed to start recording (%s): see console'%(cam_id,),0)
                raise x

    def OnRecordRawStop(self):
        if not len(self._currently_recording_cams):
            self.statusbar.SetStatusText('Not recording - cannot stop',0)
            return
        try:
            n_stopped = 0
            for cam_id in self._currently_recording_cams[:]:
                self.main_brain.stop_recording(cam_id)
                self._currently_recording_cams.remove(cam_id)
                n_stopped+=1
            self.statusbar.SetStatusText('Recording stopped on %d cameras'%(
                n_stopped,))
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
    
    def OnSnapshot(self,event):
        snapshot_cam_choice = XRCCTRL(self.snapshot_panel,
                                      "snapshot_cam_choice")
        cam_id = snapshot_cam_choice.GetStringSelection()
        if cam_id == '':
            return
        frame, points = self.main_brain.get_image_sync(cam_id)
        height = frame.shape[0]
        self.plotpanel.set_image(frame)
        self.plotpanel.set_points(points)
        self.plotpanel.draw()
            
    def OnToggleTint(self, event):
        self.cam_image_canvas.set_clipping( event.IsChecked() )

    def OnSetTimer(self, event):
        dlg=wxTextEntryDialog(self.frame, 'What interval should the display be updated at (msec)?',
                              'Set display update interval',str(self.update_interval))
        try:
            if dlg.ShowModal() == wxID_OK:
                self.update_interval = int(dlg.GetValue())
                self.timer.Start(self.update_interval)
        finally:
            dlg.Destroy()

    def attach_and_start_main_brain(self,main_brain):
        self.main_brain = main_brain
        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening()

    def update_wx(self):
        self.statusbar.SetStatusText('%d camera(s)'%len(self.cameras),1)
        
    def OnStartCalibration(self, event):
        doit = False
        dlg = wxDirDialog( self.frame, "Calibration save directory",
                           style = wxDD_DEFAULT_STYLE | wxDD_NEW_DIR_BUTTON,
                           defaultPath = os.environ.get('HOME','')
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
                    image, show_fps, points = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
                except KeyError:
                    # unexpected disconnect
                    pass # may have lost camera since call to service_pending
                if image is not None:
                    self.cam_image_canvas.update_image(cam_id,image)
                if show_fps is not None:
                    show_fps_label = XRCCTRL(PreviewPerCamPanel,'acquired_fps_label') # get container
                    show_fps_label.SetLabel('fps: %.1f'%show_fps)
                self.cam_image_canvas.set_draw_points(cam_id,points)
                
            self.cam_image_canvas.OnDraw()

            if isinstance(event,wxIdleEventPtr):
                event.RequestMore()
                
        elif self.current_page == 'tracking':
            data3d=MainBrain.get_realtime_data()
            if data3d is not None:
                XRCCTRL(self.tracking_panel,'x_pos').SetValue('% 8.1f'%data3d[0])
                XRCCTRL(self.tracking_panel,'y_pos').SetValue('% 8.1f'%data3d[1])
                XRCCTRL(self.tracking_panel,'z_pos').SetValue('% 8.1f'%data3d[2])
        else:
            # do other stuff
            pass
            
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

    def OnCollectBackground(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.collect_background(cam_id)

    def OnClearBackground(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.clear_background(cam_id)

    def OnCloseCamera(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera
    
    def OnSetCameraThreshold(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.set_diff_threshold(cam_id,value) # eventually calls OnOldCamera
    
    def OnOldCamera(self, cam_id):
        try:
            self.cam_image_canvas.delete_image(cam_id)
        
        except KeyError:
            # camera never sent frame??
            pass

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
        del app

    finally:
        # stop main_brain server
        main_brain.quit()
    
if __name__ == '__main__':
    main()
