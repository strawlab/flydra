#!/usr/bin/env python
import sys, threading, time, os, copy, socket
import traceback
import MainBrain
from MainBrain import DEBUG
import wx
from wx import xrc
import pyglet.gl.lib
import numpy
import flydra.a2.xml_stimulus

PLOTPANEL = True
if PLOTPANEL:
    from MatplotlibPanel import PlotPanel
import common_variables
import flydra.kalman.dynamic_models
from flydra.kalman.point_prob import some_rough_negative_log_likelihood
import flydra.data_descriptions

import pkg_resources
from optparse import OptionParser
import pprint

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor


SCROLLED=True
if SCROLLED:
    from wx.lib.scrolledpanel import ScrolledPanel

import motmot.wxvalidatedtext.wxvalidatedtext as wxvt

 # trigger extraction
RESFILE = pkg_resources.resource_filename(__name__,"flydra_server.xrc")
pkg_resources.resource_filename(__name__,"flydra_server_art.png")
pkg_resources.resource_filename(__name__,"detect.wav")

RESDIR = os.path.split(RESFILE)[0]
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

class AudioNotification(traits.HasTraits):
    enabled = traits.Bool(False)
    traits_view = View( Group( ( Item('enabled'),
                                 )),
                        title = 'Audio notification',
                        )

class StatusTraits(traits.HasTraits):
    audio_notification = traits.Instance( AudioNotification, args=() )

    traits_view = View( Group( ( Item( 'audio_notification', style='custom',
                                       show_label=False),
                                 )),
                        title = 'Status',
                        )

def my_loadpanel(parent,panel_name):
    orig_dir = os.path.abspath(os.curdir)
    os.chdir(RESDIR)
    try:
        result = RES.LoadPanel(parent,panel_name)
    finally:
        os.chdir(orig_dir)
    return result

def validate_positive_float(val_str):
    try:
        val = float(val_str)
    except ValueError,err:
        return False
    if val>0.0:
        return True
    else:
        return False

def wrap_loud( wxparent, func ):
    class LoudWrapper:
        """show a message dialog when a function call fails"""
        def __init__(self,wxparent,func):
            self.func = func
            self.wxparent = wxparent
        def __call__(self,*args,**kw):
            try:
                return self.func(*args,**kw)
            except Exception, err:
                dlg2 = wx.MessageDialog( self.wxparent, 'Error: %s'%str(err),
                                         'Unexpected error!',
                                         wx.OK | wx.ICON_ERROR )
                try:
                    dlg2.ShowModal()
                finally:
                    dlg2.Destroy()
                    raise
            except:
                dlg2 = wx.MessageDialog( self.wxparent, 'unknown error',
                                         'Unexpected error!',
                                         wx.OK | wx.ICON_ERROR )
                try:
                    dlg2.ShowModal()
                finally:
                    dlg2.Destroy()
                    raise
    return LoudWrapper( wxparent, func)

class wxMainBrainApp(wx.App):
    def OnInit(self,*args,**kw):
        global use_opengl, wxglvideo

        if use_opengl:
            import motmot.wxglvideo.simple_overlay as wxglvideo
        else:
            import motmot.wxvideo.wxvideo as wxglvideo

        self.pass_all_keystrokes = False
        wx.InitAllImageHandlers()
        frame = wx.Frame(None, -1, "Flydra Main Brain",size=(650,600))

        # statusbar ----------------------------------
        self.statusbar = frame.CreateStatusBar()
        self.statusbar.SetFieldsCount(4)

        # menubar ------------------------------------
        menuBar = wx.MenuBar()
        #   File
        filemenu = wx.Menu()

        ID_open_cam_config = wx.NewId()
        filemenu.Append(ID_open_cam_config, "Open Camera Configuration...\tCtrl-O")
        wx.EVT_MENU(self, ID_open_cam_config, self.OnOpenCamConfig)

        ID_save_cam_config = wx.NewId()
        filemenu.Append(ID_save_cam_config, "Save Camera Configuration...\tCtrl-S")
        wx.EVT_MENU(self, ID_save_cam_config, self.OnSaveCamConfig)

        filemenu.AppendItem(wx.MenuItem(kind=wx.ITEM_SEPARATOR))

        ID_about = wx.NewId()
        filemenu.Append(ID_about, "About", "About wx.MainBrain")
        wx.EVT_MENU(self, ID_about, self.OnAboutMainBrain)

        filemenu.AppendItem(wx.MenuItem(kind=wx.ITEM_SEPARATOR))

        ID_quit = wx.NewId()
        filemenu.Append(ID_quit, "Quit\tCtrl-Q", "Quit application")
        wx.EVT_MENU(self, ID_quit, self.OnQuit)
        wx.EVT_CLOSE(frame, self.OnWindowClose)

        menuBar.Append(filemenu, "&File")

        #   Data logging
        data_logging_menu = wx.Menu()

        ID_change_save_data_dir = wx.NewId()
        data_logging_menu.Append(ID_change_save_data_dir, "Set save data dir...")
        wx.EVT_MENU(self, ID_change_save_data_dir, self.OnChangeSaveDataDir)

        ID_start_saving_data = wx.NewId()
        data_logging_menu.Append(ID_start_saving_data, "Start saving data...",
                                 "Collect & save all 2D and 3D data")
        wx.EVT_MENU(self, ID_start_saving_data, self.OnStartSavingData)

        ID_stop_saving_data = wx.NewId()
        data_logging_menu.Append(ID_stop_saving_data, "Stop saving data",
                                 "Stop saving data")
        wx.EVT_MENU(self, ID_stop_saving_data, self.OnStopSavingData)

        ID_toggle_debugging_text = wx.NewId()
        data_logging_menu.Append(ID_toggle_debugging_text, "Toggle emitting debug data to console")
        wx.EVT_MENU(self, ID_toggle_debugging_text, self.OnToggleDebuggingText)

        ID_toggle_show_overall_latency = wx.NewId()
        data_logging_menu.Append(ID_toggle_show_overall_latency,
                                 "Toggle emitting overall latency data to console")
        wx.EVT_MENU(self, ID_toggle_show_overall_latency, self.OnToggleShowOverallLatency)

        menuBar.Append(data_logging_menu, "Data &Logging")

        #   Cameras
        cammenu = wx.Menu()

        # XXX not finished
        #ID_stop_all_collecting_bg = wx.NewId()
        #cammenu.Append(ID_stop_all_collecting_bg, "Stop all running background collection")
        #wx.EVT_MENU(self, ID_stop_all_collecting_bg, self.OnStopAllCollectingBg)

        ID_set_fps = wx.NewId()
        cammenu.Append(ID_set_fps, "Set framerate...")
        wx.EVT_MENU(self, ID_set_fps,
                    wrap_loud(frame,self.OnSetFps))

        menuBar.Append(cammenu, "&Cameras")

        #   View
        viewmenu = wx.Menu()

        ID_toggle_image_tinting = wx.NewId()
        viewmenu.Append(ID_toggle_image_tinting, "Tint clipped data",
                        "Tints clipped pixels green", wx.ITEM_CHECK)
        wx.EVT_MENU(self, ID_toggle_image_tinting, self.OnToggleTint)

        ID_toggle_show_likely_points_only = wx.NewId()
        viewmenu.Append(ID_toggle_show_likely_points_only, "Show only likely points",
                        "Show only likely points", wx.ITEM_CHECK)
        wx.EVT_MENU(self, ID_toggle_show_likely_points_only, self.OnShowLikelyPointsOnly)
        self.show_likely_points_only = False

        ID_draw_points = wx.NewId()
        viewmenu.Append(ID_draw_points, "Draw points",
                        "Draw 2D points and orientation", wx.ITEM_CHECK)
        wx.EVT_MENU(self, ID_draw_points, self.OnToggleDrawPoints)

        self.show_xml_stim = None
        ID_show_xml_stimulus = wx.NewId()
        viewmenu.Append(ID_show_xml_stimulus, "Show XML stimulus...",
                        "Show XML based stimulus")
        wx.EVT_MENU(self, ID_show_xml_stimulus, self.OnShowXMLStimulus)

        ID_hide_xml_stimulus = wx.NewId()
        viewmenu.Append(ID_hide_xml_stimulus, "Hide XML stimulus",
                        "Hide XML based stimulus")
        wx.EVT_MENU(self, ID_hide_xml_stimulus, self.OnHideXMLStimulus)

        ID_set_timer = wx.NewId()
        viewmenu.Append(ID_set_timer, "Set update timer...",
                        "Sets interval at which display is updated")
        wx.EVT_MENU(self, ID_set_timer, self.OnSetTimer)

        ID_set_timer2 = wx.NewId()
        viewmenu.Append(ID_set_timer2, "Set raw image update timer...",
                        "Sets interval at which images are updated")
        wx.EVT_MENU(self, ID_set_timer2, self.OnSetTimer2)

        menuBar.Append(viewmenu, "&View")

        # finish menubar -----------------------------
        frame.SetMenuBar(menuBar)

        try:
            self.detect_sound = wx.Sound(os.path.join(RESDIR,'detect.wav'))
            self.detect_sound.Play()
        except NameError: #wx.Sound not in some versions of wx
            self.detect_sound = None

        # main panel ----------------------------------
        self.main_panel = my_loadpanel(frame,"APP_PANEL") # make frame main panel
        self.main_panel.SetFocus()

        frame_box = wx.BoxSizer(wx.VERTICAL)
        frame_box.Add(self.main_panel,1,wx.EXPAND)
        frame.SetSizer(frame_box)
        frame.Layout()

        nb = xrc.XRCCTRL(self.main_panel,"MAIN_NOTEBOOK")

        # setup notebook pages

        self.cam_preview_panel = my_loadpanel(nb,"PREVIEW_PANEL")
        self.cam_preview_panel.SetAutoLayout(True)
        nb.AddPage(self.cam_preview_panel,"Camera Preview/Settings")
        self.InitPreviewPanel()

        if hasattr(self.cam_image_canvas,'get_clipping'):
            viewmenu.Check(ID_toggle_image_tinting,self.cam_image_canvas.get_clipping())
            viewmenu.Check(ID_draw_points,self.cam_image_canvas.get_display_points())
        else:
            viewmenu.Enable(ID_toggle_image_tinting,False)
            viewmenu.Enable(ID_draw_points,False)

        self.snapshot_panel = my_loadpanel(nb,"SNAPSHOT_PANEL")
        nb.AddPage(self.snapshot_panel,"Snapshot")
        self.InitSnapshotPanel()

        self.record_raw_panel = my_loadpanel(nb,"RECORD_RAW_PANEL")
        nb.AddPage(self.record_raw_panel,"Record raw video")
        self.InitRecordRawPanel()

        self.status_traits = StatusTraits()
        self.status_panel = my_loadpanel(nb,"STATUS_PANEL")
        nb.AddPage(self.status_panel,"Status")
        self.InitStatusPanel()

        #temp_panel = my_loadpanel(nb,"UNDER_CONSTRUCTION_PANEL")
        #nb.AddPage(temp_panel,"Under construction")

        wx.EVT_NOTEBOOK_PAGE_CHANGED(nb,nb.GetId(),self.OnPageChanged)
        self.main_notebook = nb
        self.current_page = 'preview'

        # finalize wx stuff

        frame.SetAutoLayout(True)

        frame.Show()
        self.SetTopWindow(frame)
        self.frame = frame

        ID_Timer  = wx.NewId()
        self.timer = wx.Timer(self,      # object to send the event to
                              ID_Timer)  # event id to use
        wx.EVT_TIMER(self,  ID_Timer, wrap_loud(self.frame,self.OnTimer))
        self.update_interval=100
        self.timer.Start(self.update_interval) # call every n msec
##        wx.EVT_IDLE(self.frame, self.OnIdle)

        # raw image update timer
        ID_Timer2  = wx.NewId()
        self.timer2 = wx.Timer(self,      # object to send the event to
                               ID_Timer2)  # event id to use
        wx.EVT_TIMER(self,  ID_Timer2, self.OnUpdateRawImages)
        self.update_interval2=2000
        self.timer2.Start(self.update_interval2) # call every n msec

        self.cameras = {} #OrderedDict()

        wx.EVT_KEY_DOWN(self, self.OnKeyDown)

        self.update_wx()

        self.collecting_background_buttons = {}
        self.take_background_buttons = {}
        self.clear_background_buttons = {}
        self.last_sound_time = time.time()

        self.save_data_dir = os.environ.get('HOME','')
        test_dir = os.path.join( self.save_data_dir, 'ORIGINAL_DATA' )
        if os.path.exists(test_dir):
            self.save_data_dir = test_dir

        return True

    def OnAboutMainBrain(self, event):
        disp = 'Loaded modules (.eggs):\n'
        disp += '----------------------\n'
        for d in pkg_resources.working_set:
            disp += str(d) + '\n'
        dlg = wx.MessageDialog(self.frame, disp,
                               'About wxMainBrain',
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def OnKeyDown(self, event):
        if self.pass_all_keystrokes:
            # propagate event up the chain...
            event.Skip()
            return

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
                event = wx.CommandEvent(wx.wxEVT_COMMAND_CHECKBOX_CLICKED,id)
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
                event = wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED,id)
                event.SetEventObject( widget )
                widget.Command( event )
            self.statusbar.SetStatusText('took BG images',0)
        elif keyname == 'C':
            for cam_id in self.cameras.keys():
                widget = self.clear_background_buttons[cam_id]
                id = widget.GetId()
                event = wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED,id)
                event.SetEventObject( widget )
                widget.Command( event )
            self.statusbar.SetStatusText('cleared BG images',0)

        # Propagate event up the chain. (We don't mind propagating
        # keys that we've processed because normally wx doesn't do
        # anything with these keys, but when we're in a text entry
        # dialog box, we do want wx to grab them.)
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
        # init "load calibration..." button
        load_cal = xrc.XRCCTRL(self.cam_preview_panel,
                           "LOAD_CAL_DIR")
        wx.EVT_BUTTON(load_cal, load_cal.GetId(), self.OnLoadCalDir)

        load_cal = xrc.XRCCTRL(self.cam_preview_panel,
                           "LOAD_CAL_FILE")
        wx.EVT_BUTTON(load_cal, load_cal.GetId(), self.OnLoadCalFile)

        clear_cal = xrc.XRCCTRL(self.cam_preview_panel,
                           "CLEAR_CAL")
        wx.EVT_BUTTON(clear_cal, clear_cal.GetId(), self.OnClearCal)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,
                           "MANUAL_TRIGGER_DEVICE_PREVIEW1") # EXT TRIG1
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice1)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,
                           "MANUAL_TRIGGER_DEVICE_PREVIEW2") # EXT TRIG2
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice2)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,
                           "MANUAL_TRIGGER_DEVICE_PREVIEW3") # EXT TRIG3
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice3)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,
                           "MANUAL_RECORD_RAW_TOGGLE")
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),self.OnRecordRawButton)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,
                           "MANUAL_RECORD_SMALL_TOGGLE")
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),self.OnRecordSmallButton)

        ctrl = xrc.XRCCTRL(self.cam_preview_panel,'SYNCHRONIZE_BUTTON')
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      wrap_loud(self.cam_preview_panel,self.OnSynchronizeButton))

        # setup dynamic image (OpenGL) panel
        dynamic_image_panel = xrc.XRCCTRL(self.cam_preview_panel,"PreviewDynamicImagePanel") # get container
        box = wx.BoxSizer(wx.VERTICAL)
        dynamic_image_panel.SetSizer(box)

        if int(os.environ.get('WXGLVIDEO_FORCE_ATTRIBLIST','0')):
            child_kwargs={}
            child_kwargs['attribList']=0
        else:
            child_kwargs=None
        self.cam_image_canvas = wxglvideo.DynamicImageCanvas(dynamic_image_panel,-1,child_kwargs=child_kwargs)
        box.Add(self.cam_image_canvas,1,wx.EXPAND)
        dynamic_image_panel.Layout()

        # setup per-camera container panel
        container = xrc.XRCCTRL(self.cam_preview_panel,"PreviewPerCamPanel")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        container.SetSizer(sizer)

        if SCROLLED:
            scrolled_container = ScrolledPanel(container,-1)
        else:
            scrolled_container = wx.Panel(container,-1)
        sizer.Add(scrolled_container,1,wx.EXPAND)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        scrolled_container.SetSizer(sizer)
        scrolled_container.SetAutoLayout(True)

        if SCROLLED:
            scrolled_container.SetupScrolling()
        self.preview_per_cam_scrolled_container = scrolled_container

        self.preview_per_cam_scrolled_container.Layout()

    def PreviewPerCamInit(self,cam_id):
        scalar_control_info=self.cameras[cam_id]['scalar_control_info']

        # add self to WX
        previewPerCamPanel = my_loadpanel(self.preview_per_cam_scrolled_container,
                                           "preview_per_cam_panel")
        acp_box = self.preview_per_cam_scrolled_container.GetSizer()
        all_cams = self.cameras.keys()
        all_cams.sort()
        cam_number = all_cams.index(cam_id)
        acp_box.Insert(cam_number,previewPerCamPanel,0,wx.EXPAND | wx.ALL,border=3)
        # arrgh, cannot make the scrolled container expand... This doesn't do it:
        acp_box.Fit(self.preview_per_cam_scrolled_container)

        # set staticbox label
        box = previewPerCamPanel.GetSizer()
        static_box = box.GetStaticBox()
        static_box.SetLabel( cam_id )

        collecting_background = xrc.XRCCTRL(previewPerCamPanel,"COLLECTING_BACKGROUND")
        self.collecting_background_buttons[cam_id] = collecting_background
        collecting_background.SetValue(scalar_control_info['collecting_background'])
        wx.EVT_CHECKBOX(collecting_background, collecting_background.GetId(),
                     self.OnCollectingBackground)

        take_background = xrc.XRCCTRL(previewPerCamPanel,"take_background")
        self.take_background_buttons[cam_id] = take_background
        wx.EVT_BUTTON(take_background, take_background.GetId(),
                   self.OnTakeBackground)

        clear_background = xrc.XRCCTRL(previewPerCamPanel,"clear_background")
        self.clear_background_buttons[cam_id] = clear_background
        wx.EVT_BUTTON(clear_background, clear_background.GetId(),
                   self.OnClearBackground)

        set_roi = xrc.XRCCTRL(previewPerCamPanel,"set_roi")
        wx.EVT_BUTTON(set_roi, set_roi.GetId(), self.OnSetROI)

        quit_camera = xrc.XRCCTRL(previewPerCamPanel,"quit_camera")
        wx.EVT_BUTTON(quit_camera, quit_camera.GetId(), self.OnCloseCamera)

        ctrl = xrc.XRCCTRL(previewPerCamPanel,"n_sigma")
        val = scalar_control_info['n_sigma']
        ctrl.SetValue( str( val ) )
        wx.EVT_TEXT(ctrl, ctrl.GetId(), self.OnSetCameraNSigma)

        trigger_mode_value = xrc.XRCCTRL(previewPerCamPanel,"trigger_mode_number")
        val = 0 #TODO: scalar_control_info['trigger_mode']
        N_trigger_modes = scalar_control_info.get('N_trigger_modes',2) # hack for old camnodes
        assert (0<= val) and (val < N_trigger_modes)
        #trigger_mode_value.SetValue( str( val ) )
        wx.EVT_TEXT(trigger_mode_value, trigger_mode_value.GetId(), self.OnSetTriggerModeNumber)

        threshold_value = xrc.XRCCTRL(previewPerCamPanel,"threshold_value")
        val = scalar_control_info['diff_threshold']
        threshold_value.SetValue( str( val ) )
        wx.EVT_TEXT(threshold_value, threshold_value.GetId(), self.OnSetCameraThreshold)

        threshold_clear_value = xrc.XRCCTRL(previewPerCamPanel,"threshold_clear_value")
        val = scalar_control_info['clear_threshold']
        threshold_clear_value.SetValue( str( val ) )
        wx.EVT_TEXT(threshold_clear_value, threshold_clear_value.GetId(),
                 self.OnSetCameraClearThreshold)

        max_framerate = xrc.XRCCTRL(previewPerCamPanel,"MAX_FRAMERATE")
        if 'max_framerate' in scalar_control_info:
            val = scalar_control_info['max_framerate']
            max_framerate.SetValue( str( val ) )
            wx.EVT_TEXT(max_framerate, max_framerate.GetId(),
                     self.OnSetMaxFramerate)
        else:
            max_framerate.Enable(False)

        view_image_choice = xrc.XRCCTRL(previewPerCamPanel,"view_image_display")
        wx.EVT_CHOICE(view_image_choice, view_image_choice.GetId(),
                   self.OnSetViewImageChoice)

        per_cam_controls_panel = xrc.XRCCTRL(previewPerCamPanel,
                                         "PerCameraControlsContainer")
        grid = wx.FlexGridSizer(0,3,0,0) # 3 columns
        grid.AddGrowableCol(2)

        params = scalar_control_info.keys()
        params.sort()
        params.reverse()
        camprops = scalar_control_info['camprops']
        camprops.sort()
        for param in camprops:
            current_value, min_value, max_value = scalar_control_info[param]
            grid.Add( wx.StaticText(per_cam_controls_panel,wx.NewId(),param),
                     0,wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL )

            txtctrl = wx.TextCtrl( per_cam_controls_panel, wx.NewId(),
                                  size=(40,20))
            txtctrl.SetValue(str(current_value))
            grid.Add( txtctrl,0,wx.ALIGN_LEFT )

            slider = wx.Slider( per_cam_controls_panel, wx.NewId(),
                               current_value, min_value, max_value,
                               style= wx.SL_HORIZONTAL )
            grid.Add( slider,1,wx.EXPAND )

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
                label_if_shutter = xrc.XRCCTRL(previewPerCamPanel,
                                           'exposure_label') # get label
                label_if_shutter.SetLabel(
                    'Exposure (msec): %.3f'%(current_value*0.02,))
            else:
                label_if_shutter = None
            psh = ParamSliderHelper(param,cam_id,txtctrl,slider,
                                    self.main_brain,label_if_shutter)
            wx.EVT_COMMAND_SCROLL(slider, slider.GetId(), psh.onScroll)
            wx.EVT_TEXT(txtctrl, txtctrl.GetId(), psh.onText)
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
            threshold_clear_value = xrc.XRCCTRL(previewPerCamPanel,"threshold_clear_value")
            threshold_clear_value.SetValue( str( value ) )
        elif param == 'diff_threshold':
            threshold_diff_value = xrc.XRCCTRL(previewPerCamPanel,"threshold_value")
            threshold_diff_value.SetValue( str( value ) )
        elif param == 'n_sigma':
            ctrl = xrc.XRCCTRL(previewPerCamPanel,"n_sigma")
            ctrl.SetValue( str( value ) )
        elif param == 'trigger_mode':
            print 'WARNING: skipping trigger mode update in display'
        else:
            if param not in ('roi','width','height'):
                print 'WARNING: could not update panel display for',param

    def OnKalmanParametersChange(self,event=None):
        ctrl = xrc.XRCCTRL(self.status_panel,
                           "kalman_parameters_choice")
        kalman_param_string = ctrl.GetStringSelection()
        name=str(kalman_param_string)

        MainBrain.rc_params['kalman_model'] = name
        MainBrain.save_rc_params()

        if self.main_brain.reconstructor is not None:
            print 'setting model to',name
            self.main_brain.set_new_tracker(kalman_model_name=name)
        else:
            print 'no reconstructor, not setting kalman model'

    def PreviewPerCamClose(self,cam_id):
        previewPerCamPanel=self.cameras[cam_id]['previewPerCamPanel']
        previewPerCamPanel.DestroyChildren()
        previewPerCamPanel.Destroy()

    def InitSnapshotPanel(self):
        snapshot_cam_choice = xrc.XRCCTRL(self.snapshot_panel,
                                          "snapshot_cam_choice")
        snapshot_button = xrc.XRCCTRL(self.snapshot_panel,
                                  "snapshot_button")
        snapshot_colormap = xrc.XRCCTRL(self.snapshot_panel,
                                  "snapshot_colormap")
        fixed_color_range = xrc.XRCCTRL(self.snapshot_panel,
                                    "fixed_color_range")
        wx.EVT_CHECKBOX(fixed_color_range, fixed_color_range.GetId(),
                        self.OnFixedColorRange)
        wx.EVT_BUTTON(snapshot_button, snapshot_button.GetId(),
                      self.OnSnapshot)
        wx.EVT_CHOICE(snapshot_colormap, snapshot_colormap.GetId(),
                      self.OnSnapshotColormap)
        wx.EVT_LISTBOX(snapshot_cam_choice, snapshot_cam_choice.GetId(),
                       self.OnSnapshot)
        wx.EVT_LISTBOX_DCLICK(snapshot_cam_choice, snapshot_cam_choice.GetId(),
                              self.OnSnapshot)
        snapshot_plot = xrc.XRCCTRL(self.snapshot_panel,"snapshot_plot")
        sizer = wx.BoxSizer(wx.VERTICAL)

        # matplotlib panel itself
        if PLOTPANEL:
            self.plotpanel = PlotPanel(snapshot_plot)
            self.plotpanel.init_plot_data()

            # wx boilerplate
            sizer.Add(self.plotpanel, 1, wx.EXPAND)
        snapshot_plot.SetSizer(sizer)

    def SnapshotPerCamInit(self,cam_id):
        # Choice control
        snapshot_cam_choice = xrc.XRCCTRL(self.snapshot_panel,
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
        snapshot_cam_choice = xrc.XRCCTRL(self.snapshot_panel,
                                         "snapshot_cam_choice")
        i=snapshot_cam_choice.FindString(cam_id)
        snapshot_cam_choice.Delete(i)

    def InitRecordRawPanel(self):
        self.record_raw = xrc.XRCCTRL(self.record_raw_panel,
                                  "record_raw")
        wx.EVT_CHECKBOX(self.record_raw, self.record_raw.GetId(),
                     self.OnRecordRaw)

##        filename_text_entry = xrc.XRCCTRL(self.record_raw_panel,
##                                      "record_raw_filename")

        self.record_small = xrc.XRCCTRL(self.record_raw_panel,
                                  "record_small")
        wx.EVT_CHECKBOX(self.record_small, self.record_small.GetId(),
                     self.OnRecordSmall)

##        small_filename_text_entry = xrc.XRCCTRL(self.record_raw_panel,
##                                            "record_small_filename")

##        wx.EVT_SET_FOCUS(filename_text_entry,
##                      self.OnFilenameSetFocus)

##        wx.EVT_KILL_FOCUS(filename_text_entry,
##                      self.OnFilenameKillFocus)

##        wx.EVT_SET_FOCUS(small_filename_text_entry,
##                      self.OnFilenameSetFocus)

##        wx.EVT_KILL_FOCUS(small_filename_text_entry,
##                      self.OnFilenameKillFocus)

        self._currently_recording_cams = []
        self._currently_recording_small_cams = []

    def OnFilenameSetFocus(self,event):
        #print 'OnFilenameSetFocus'
        self.pass_all_keystrokes = True
        event.Skip()

    def OnFilenameKillFocus(self,event):
        #print 'OnFilenameKillFocus'
        self.pass_all_keystrokes = False
        event.Skip()

    def OnRecordRawButton(self,event):
        if self.record_raw.IsChecked():
            #toggle value
            self.record_raw.SetValue(False)
        else:
            self.record_raw.SetValue(True)
        self.OnRecordRaw(None)

    def OnRecordSmallButton(self,event):
        if self.record_small.IsChecked():
            #toggle value
            self.record_small.SetValue(False)
        else:
            self.record_small.SetValue(True)
        self.OnRecordSmall(None)

    def OnRecordRaw(self,event):
        if self.record_raw.IsChecked():
            self.OnRecordRawStart()
        else:
            self.OnRecordRawStop()

    def OnRecordSmall(self,event):
        if self.record_small.IsChecked():
            self.OnRecordSmallStart()
        else:
            self.OnRecordSmallStop()

    def OnRecordRawStart(self):
        if len(self._currently_recording_cams) != 0:
            raise RuntimeError("currently recording!")

        cam_choice = xrc.XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
##        filename_text_entry = xrc.XRCCTRL(self.record_raw_panel,
##                                      "record_raw_filename")
##        raw_filename = filename_text_entry.GetValue()
##        if raw_filename.endswith('.fmf'):
##            bg_filename = raw_filename[:-4] + '_bg.fmf'
##        else:
##            bg_filename = raw_filename + '.bg.fmf'
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))
        if len(cam_ids)==0:
            self.record_raw.SetValue(False)
            return
        try:
            nowstr = time.strftime( '%Y%m%d_%H%M%S' )
            for cam_id in cam_ids:
                basename = '~/FLYDRA_LARGE_MOVIES/full_%s_%s'%(nowstr,cam_id)
                self.main_brain.start_recording(cam_id,
                                                basename)
                self._currently_recording_cams.append(cam_id)
            self.statusbar.SetStatusText('Recording started on %d cameras'%(
                len(self._currently_recording_cams),),0)
        except Exception, exc:
            try:
                for tmp_cam_id in self._currently_recording_cams[:]:
                    self.main_brain.stop_recording(tmp_cam_id)
                    self._currently_recording_cams.remove(tmp_cam_id)
            finally:
                self.record_raw.SetValue(False)

                self.statusbar.SetStatusText(
                    'Failed to start recording (%s): see console'%(cam_id,),0)
                raise exc

    def OnRecordSmallStart(self):
        if len(self._currently_recording_small_cams) != 0:
            raise RuntimeError("currently recording!")

        cam_choice = xrc.XRCCTRL(self.record_raw_panel,
                             "record_raw_cam_select_checklist")
##        filename_text_entry = xrc.XRCCTRL(self.record_raw_panel,
##                                      "record_small_filename")
##        small_filename = filename_text_entry.GetValue()
##        if small_filename.endswith('.fmf'):
##            small_datafile_filename = small_filename[:-4] + '.smd'
##        else:
##            small_datafile_filename = small_filename + '.smd'
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))
        if len(cam_ids)==0:
            self.record_small.SetValue(False)
            return
        try:
            nowstr = time.strftime( '%Y%m%d_%H%M%S' )
            for cam_id in cam_ids:
                basename = '~/FLYDRA_SMALL_MOVIES/small_%s_%s'%(nowstr,cam_id)
                self.main_brain.start_small_recording(cam_id,
                                                      basename)
                self._currently_recording_small_cams.append(cam_id)
            self.statusbar.SetStatusText('Small recording started on %d cameras'%(
                len(self._currently_recording_small_cams),),0)
        except Exception, exc:
            try:
                for tmp_cam_id in self._currently_recording_small_cams[:]:
                    self.main_brain.stop_small_recording(tmp_cam_id)
                    self._currently_recording_small_cams.remove(tmp_cam_id)
            finally:
                self.record_small.SetValue(False)

                self.statusbar.SetStatusText(
                    'Failed to start small recording (%s): see console'%(cam_id,),0)
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
            self.record_raw.SetValue(False)
        except:
            self.statusbar.SetStatusText('Failed to stop recording: see console',0)
            raise

    def OnRecordSmallStop(self,warn=True):
        if warn and not len(self._currently_recording_small_cams):
            self.statusbar.SetStatusText('Not recording small FMFs - cannot stop',0)
            return
        try:
            n_stopped = 0
            for cam_id in self._currently_recording_small_cams[:]:
                try:
                    self.main_brain.stop_small_recording(cam_id)
                except KeyError, x:
                    print '%s: %s'%(x.__class__,str(x)),
                    MainBrain.DEBUG()
                self._currently_recording_small_cams.remove(cam_id)
                n_stopped+=1
            self.statusbar.SetStatusText('Small recording stopped on %d cameras'%(
                n_stopped,))
            self.record_small.SetValue(False)
        except:
            self.statusbar.SetStatusText('Failed to stop small recording: see console',0)
            raise

    def RecordRawPerCamInit(self,cam_id):
        # Choice control
        cam_choice = xrc.XRCCTRL(self.record_raw_panel,
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
        cam_choice = xrc.XRCCTRL(self.record_raw_panel,
                                 "record_raw_cam_select_checklist")
        i=cam_choice.FindString(cam_id)
        cam_choice.Delete(i)

    def InitStatusPanel(self):
        ctrl = xrc.XRCCTRL(self.status_panel,
                           "kalman_parameters_choice")
        wx.EVT_CHOICE(ctrl, ctrl.GetId(),
                      self.OnKalmanParametersChange)

        ctrl = xrc.XRCCTRL(self.status_panel,
                           "MANUAL_TRIGGER_DEVICE_STATUS1") # EXT TRIG1
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice1)

        ctrl = xrc.XRCCTRL(self.status_panel,
                           "MANUAL_TRIGGER_DEVICE_STATUS2") # EXT TRIG2
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice2)

        ctrl = xrc.XRCCTRL(self.status_panel,
                           "MANUAL_TRIGGER_DEVICE_STATUS3") # EXT TRIG3
        wx.EVT_BUTTON(ctrl, ctrl.GetId(),
                      self.OnManualTriggerDevice3)

        panel = xrc.XRCCTRL(self.status_panel,'TRAITED_STATUS_PANEL')
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        control = self.status_traits.edit_traits( parent=panel,
                                                  kind='subpanel',
                                                  ).control
        #control.GetParent().SetMinSize(control.GetMinSize())
        sizer.Add(control, 1, wx.EXPAND)
        panel.SetSizer( sizer )

    def OnHypothesisTestMaxError(self,event):
        ctrl = xrc.XRCCTRL(self.status_panel,
                       "HYPOTHESIS_TEST_MAX_ERR")
        val = float(ctrl.GetValue())
        self.main_brain.set_hypothesis_test_max_error(val)

    def OnManualTriggerDevice1(self,event):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto('1',(MainBrain.hostname,common_variables.trigger_network_socket_port))

    def OnManualTriggerDevice2(self,event):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto('2',(MainBrain.hostname,common_variables.trigger_network_socket_port))

    def OnManualTriggerDevice3(self,event):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto('3',(MainBrain.hostname,common_variables.trigger_network_socket_port))

    def OnLoadCalFile(self,event):
        self.OnLoadCalCore(event,wx.FileDialog,
                       "Select file with calibration data")
    def OnLoadCalDir(self,event):
        self.OnLoadCalCore(event,wx.DirDialog,
                       "Select directory with calibration data")
    def OnLoadCalCore(self,event,klass,msg):
        doit=False
        dlg = klass( self.frame,msg,
                     style = wx.DD_DEFAULT_STYLE,
                     )
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                calib_dir = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False
        if doit:
            self.main_brain.load_calibration(calib_dir)
            cal_status_check = xrc.XRCCTRL(self.cam_preview_panel,
                                       "CAL_STATUS_CHECK")
            cal_status_check.Enable(True)
            cal_status_check.SetValue(True)
            cal_status_check.Enable(False)
            self.OnKalmanParametersChange() # send current Kalman parameters

    def OnClearCal(self,event):
        cal_status_check = xrc.XRCCTRL(self.cam_preview_panel,
                                       "CAL_STATUS_CHECK")
        cal_status_check.Enable(True)
        cal_status_check.SetValue(False)
        cal_status_check.Enable(False)
        self.main_brain.clear_calibration()

    def OnFixedColorRange(self, event):
        if PLOTPANEL:
            self.plotpanel.set_fixed_color_range(event.IsChecked())

    def OnSnapshotColormap(self,event):
        if PLOTPANEL:
            self.plotpanel.set_colormap(event.GetEventObject().GetStringSelection())

    def OnSnapshot(self,event):
        snapshot_cam_choice = xrc.XRCCTRL(self.snapshot_panel,
                                      "snapshot_cam_choice")
        cam_id = snapshot_cam_choice.GetStringSelection()
        if cam_id == '':
            return
        image, show_fps, points, image_coords = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
        if image is None:
            return
        if PLOTPANEL:
            self.plotpanel.set_image(image, image_coords)
            self.plotpanel.set_points(points)
            self.plotpanel.draw()

    def OnSetFps(self,event):
        dlg=wx.TextEntryDialog(self.frame, 'What should the framerate of the cameras be (Hz)?',
                               'Set fps',str(self.main_brain.get_fps()))
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                fps = float(dlg.GetValue())
                self.main_brain.set_fps(fps)
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False

    def OnChangeSaveDataDir(self, event):
        dlg = wx.DirDialog( self.frame, "Change save data directory",
                           style = wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
                           )
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                self.save_data_dir = dlg.GetPath()
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False

    def OnStartSavingData(self, event=None):
        display_save_filename = time.strftime( 'DATA%Y%m%d_%H%M%S.h5' )
        save_filename = os.path.join( self.save_data_dir, display_save_filename )
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

    def OnToggleDebuggingText(self, event=None):
        level = self.main_brain.get_debug_level()
        level = not level
        self.main_brain.set_debug_level(level)

    def OnToggleShowOverallLatency(self, event=None):
        value = self.main_brain.get_show_overall_latency()
        value = not value
        self.main_brain.set_show_overall_latency(value)

    def OnToggleTint(self, event):
        self.cam_image_canvas.set_clipping( event.IsChecked() )

    def OnShowLikelyPointsOnly(self, event):
        self.show_likely_points_only = event.IsChecked()

    def OnToggleDrawPoints(self, event):
        self.cam_image_canvas.set_display_points( event.IsChecked() )

    def OnShowXMLStimulus(self, event):
        if self.main_brain.reconstructor is None:
            dlg = wx.MessageDialog( self.frame, 'No XML stimulus will be shown until calibration is loaded.',
                                   'xml stimulus', wx.OK | wx.ICON_INFORMATION )
            try:
                dlg.ShowModal()
            finally:
                dlg.Destroy()

        doit=False
        dlg = wx.FileDialog( self.frame, "Select .xml stimulus file",
                            style = wx.OPEN,
                            #defaultDir = os.environ.get('HOME',''),
                            wildcard = '*.xml',
                            )
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                open_filename = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
        if doit:
            stim = flydra.a2.xml_stimulus.xml_stimulus_from_filename( open_filename )
            if self.main_brain.reconstructor is not None:
                stim.verify_reconstructor(self.main_brain.reconstructor)
            self.show_xml_stim = stim

    def OnHideXMLStimulus(self, event):
        self.show_xml_stim = None

    def OnSetTimer(self, event):
        dlg=wx.TextEntryDialog(self.frame, 'What interval should the display be updated at (msec)?',
                              'Set display update interval',str(self.update_interval))
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                self.update_interval = int(dlg.GetValue())
                self.timer.Start(self.update_interval)
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False

    def OnSetTimer2(self, event):
        dlg=wx.TextEntryDialog(self.frame, 'What interval should raw frames be grabbed (msec)?',
                              'Set display update interval',str(self.update_interval2))
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                self.update_interval2 = int(dlg.GetValue())
                self.timer2.Start(self.update_interval2)
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False

    def OnSetROI(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        dlg = RES.LoadDialog(self.frame,"ROI_DIALOG") # make frame main panel

        dlg_ok = xrc.XRCCTRL(dlg,"ROI_OK")
        dlg_cam_id = xrc.XRCCTRL(dlg,"ROI_cam_id")
        dlg_cam_id.SetLabel(cam_id)

        lbrt = self.main_brain.get_roi(cam_id)
        width, height = self.main_brain.get_widthheight(cam_id)

        l,b,r,t = lbrt
        xrc.XRCCTRL(dlg,"ROI_LEFT").SetValue(str(l))
        xrc.XRCCTRL(dlg,"ROI_BOTTOM").SetValue(str(b))
        xrc.XRCCTRL(dlg,"ROI_RIGHT").SetValue(str(r))
        xrc.XRCCTRL(dlg,"ROI_TOP").SetValue(str(t))

        def OnROIOK(event):
            dlg.left = int(xrc.XRCCTRL(dlg,"ROI_LEFT").GetValue())
            dlg.right = int(xrc.XRCCTRL(dlg,"ROI_RIGHT").GetValue())
            dlg.bottom = int(xrc.XRCCTRL(dlg,"ROI_BOTTOM").GetValue())
            dlg.top = int(xrc.XRCCTRL(dlg,"ROI_TOP").GetValue())
            dlg.EndModal( wx.ID_OK )
        wx.EVT_BUTTON(dlg_ok, dlg_ok.GetId(),
                   OnROIOK)
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                l,b,r,t = dlg.left,dlg.bottom,dlg.right,dlg.top
                lbrt = l,b,r,t
                if l >= r or b >= t or r >= width or t >= height:
                    raise ValueError("ROI dimensions not possible")
                self.main_brain.send_set_camera_property(cam_id,'roi',lbrt)
                self.cam_image_canvas.set_lbrt(cam_id,lbrt)
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False

    def attach_and_start_main_brain(self,main_brain):
        self.main_brain = main_brain

        if 1:
            ctrl = xrc.XRCCTRL(self.status_panel,
                               "HYPOTHESIS_TEST_MAX_ERR")
            ctrl.SetValue(str(self.main_brain.get_hypothesis_test_max_error()))
            wxvt.Validator(ctrl,
                           ctrl.GetId(),
                           self.OnHypothesisTestMaxError,
                           validate_positive_float)
        if 1:
            fps = self.main_brain.get_fps()
            model_names = flydra.kalman.dynamic_models.get_model_names()

            ctrl = xrc.XRCCTRL(self.status_panel,
                               "kalman_parameters_choice")

            found_rc_default = None
            for i,model_name in enumerate(model_names):
                ctrl.Append(model_name)
                if MainBrain.rc_params['kalman_model'] == model_name:
                    found_rc_default = i
            ctrl.GetParent().GetSizer().Layout()
            if not found_rc_default:
                found_rc_default = 0
                print 'WARNING: could not find rc default for kalman model name'
            else:
                print 'found model name %d: %s'%(i,model_names[i])
            ctrl.SetSelection( found_rc_default )
            self.OnKalmanParametersChange()

        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening()

    def update_wx(self):
        self.statusbar.SetStatusText('%d camera(s)'%len(self.cameras),1)
        self.frame.Layout()

    def OnOpenCamConfig(self, event):
        doit=False
        dlg = wx.FileDialog( self.frame, "Select file from which to open camera config data",
                            style = wx.OPEN,
                            defaultDir = os.environ.get('HOME',''),
                            defaultFile = 'flydra_cameras.cfg',
                            wildcard = '*.cfg',
                            )
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                open_filename = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False
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
                dlg2 = wx.MessageDialog( self.frame, 'Error opening configuration data:\n'\
                                        '%s: %s'%(x.__class__,x),
                                        'Error', wx.OK | wx.ICON_ERROR )
                try:
                    self.pass_all_keystrokes = True
                    dlg2.ShowModal()
                finally:
                    dlg2.Destroy()
                    self.pass_all_keystrokes = False

    def OnSaveCamConfig(self, event):
        all_params = self.main_brain.get_all_params()
        doit=False
        dlg = wx.FileDialog( self.frame, "Select file to save camera config data",
                            style = wx.SAVE | wx.OVERWRITE_PROMPT,
                            #style = wx.DD_DEFAULT_STYLE,
                            defaultDir = os.environ.get('HOME',''),
                            defaultFile = 'flydra_cameras.cfg',
                            wildcard = '*.cfg',
                            )
        try:
            self.pass_all_keystrokes = True
            if dlg.ShowModal() == wx.ID_OK:
                save_filename = dlg.GetPath()
                doit = True
        finally:
            dlg.Destroy()
            self.pass_all_keystrokes = False
        if doit:
            fd = open(save_filename,'wb')
            fd.write(pprint.pformat(all_params))
            fd.close()

    def _on_common_quit(self):
        self.timer.Stop()
        self.timer2.Stop()

        #for t in threading.enumerate():
        #    print t

        self.main_brain.quit()
        del self.main_brain

        #print '-='*20

        #for t in threading.enumerate():
        #    print t

    def OnWindowClose(self, event):
        #print 'in OnWindowClose'
        self._on_common_quit()
        #print 'stopped timers'
        event.Skip()
        #frame = sys._getframe()
        #traceback.print_stack(frame)
        sys.exit(0)

    def OnQuit(self, event):
        #print 'in OnQuit'
        self._on_common_quit()
        self.frame.Destroy()
        #frame = sys._getframe()
        #traceback.print_stack(frame)
        sys.exit(0)

    def OnSynchronizeButton(self,event):
        self.main_brain.do_synchronization()

    def OnUpdateRawImages(self, event):
        DEBUG('5')
        if self.current_page in ['preview','snapshot']:
            for cam_id in self.cameras.keys():
                try:
                    self.main_brain.request_image_async(cam_id)
                except KeyError: # no big deal, camera probably just disconnected
                    pass

    def OnTimer(self, event):
        if not hasattr( self, 'last_timer'):
            self.last_timer = time.time()
        now = time.time()
        if (now-self.last_timer) > 1.0:
            print 'timer took more than 1 sec to be called!'
        self.last_timer = now
        DEBUG('4')
        if not hasattr(self,'main_brain'):
            return # quitting
        self.main_brain.service_pending() # may call OnNewCamera, OnOldCamera, etc
        if not self.main_brain.coord_processor.isAlive():
            dlg = wx.MessageDialog( self.frame, 'Error: the coordinate processor '
                                    'thread died unexpectedly. You should re-start '
                                    'this program.', 'Unexpected error!',
                                     wx.OK | wx.ICON_ERROR )
            try:
                dlg.ShowModal()
            finally:
                dlg.Destroy()
        realtime_data=MainBrain.get_best_realtime_data() # gets global data
        if realtime_data is not None:
            Xs,min_mean_dist=realtime_data
            data3d = Xs[0]
            xrc.XRCCTRL(self.status_panel,'x_pos').SetValue('% 8.1f'%data3d[0])
            xrc.XRCCTRL(self.status_panel,'y_pos').SetValue('% 8.1f'%data3d[1])
            xrc.XRCCTRL(self.status_panel,'z_pos').SetValue('% 8.1f'%data3d[2])
            xrc.XRCCTRL(self.status_panel,'err').SetValue('% 8.1f'%min_mean_dist)
            if min_mean_dist <= 10.0:
                if self.detect_sound is not None and self.status_traits.audio_notification.enabled:
                    now = time.time()
                    if (now - self.last_sound_time) > 1.0:
                        self.detect_sound.Play()
                        self.last_sound_time = now
                if self.current_page == 'preview':
                    r=self.main_brain.reconstructor
                    if r is not None:
                        recon_cam_ids = r.get_cam_ids()
                        for cam_id in self.cameras.keys():
                            if cam_id not in recon_cam_ids:
                                # no reconstructor data for this cam_id -- skip
                                continue
                            pts = []
                            for X in Xs:
                                pt=r.find2d(cam_id,X,
                                            distorted=True)
                                pts.append( pt )
                            #print cam_id, pts
                            #pt_undist,ln=r.find2d(cam_id,data3d,
                            #               Lcoords=line3d,distorted=False)
                            #self.cam_image_canvas.set_red_points(cam_id,([pt],[ln]))
                            self.cam_image_canvas.set_red_points(cam_id,pts)
        else:
            for cam_id in self.cameras.keys():
                if hasattr(self.cam_image_canvas,'set_red_points'):
                    self.cam_image_canvas.set_red_points(cam_id,None)

        if self.current_page == 'preview':
            PT_TUPLE_IDX_AREA = flydra.data_descriptions.PT_TUPLE_IDX_AREA
            PT_TUPLE_IDX_CUR_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_CUR_VAL_IDX
            PT_TUPLE_IDX_MEAN_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_MEAN_VAL_IDX
            PT_TUPLE_IDX_SUMSQF_VAL_IDX = flydra.data_descriptions.PT_TUPLE_IDX_SUMSQF_VAL_IDX

            for cam_id in self.cameras.keys():
                cam = self.cameras[cam_id]
                if not cam.has_key('previewPerCamPanel'):
                    # not added yet
                    continue
                previewPerCamPanel = cam['previewPerCamPanel']
                try:
                    # this may fail with a Key Error if unexpected disconnect:
                    image, show_fps, points, image_coords = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
                    if image is not None:
                        if hasattr(self.cam_image_canvas,'update_image_and_drawings'):
                            # XXX TODO don't redraw image if image hasn't changed, just update point positions

                            # XXX TODO only show points with non-zero probability (or according to a slider-set scale)
                            if self.show_likely_points_only:
                                good_pts = []
                                for pt in points:
                                    if numpy.isnan(pt[0]):
                                        continue
                                    pt_area = pt[PT_TUPLE_IDX_AREA]
                                    cur_val = pt[PT_TUPLE_IDX_CUR_VAL_IDX]
                                    mean_val = pt[PT_TUPLE_IDX_MEAN_VAL_IDX]
                                    sumsqf_val = pt[PT_TUPLE_IDX_SUMSQF_VAL_IDX]
                                    nll = some_rough_negative_log_likelihood(pt_area=pt_area,
                                                                             cur_val=cur_val,
                                                                             mean_val=mean_val,
                                                                             sumsqf_val=sumsqf_val)
                                    if numpy.isfinite(nll):
                                        good_pts.append (pt )
                                points = good_pts
                            if self.show_xml_stim is not None:
                                regenerate = True
                                if hasattr(self, '_cached_xml_stim'):
                                    if self._cached_xml_stim is self.show_xml_stim:
                                        regenerate = False
                                if regenerate:
                                    segs, segcolors = self.show_xml_stim.get_distorted_linesegs( cam_id )
                                    self._cached_linesegs = {}
                                    self._cached_lineseg_colors = {}
                                    self._cached_linesegs[cam_id] = segs
                                    self._cached_lineseg_colors[cam_id] = segcolors
                                    self._cached_xml_stim = self.show_xml_stim
                                if cam_id not in self._cached_linesegs:
                                    segs, segcolors = self.show_xml_stim.get_distorted_linesegs( cam_id )
                                    self._cached_linesegs[cam_id] = segs
                                    self._cached_lineseg_colors[cam_id] = segcolors
                                linesegs, lineseg_colors = self._cached_linesegs[cam_id], self._cached_lineseg_colors[cam_id]
                            else:
                                linesegs = None
                                lineseg_colors = None
                            self.cam_image_canvas.update_image_and_drawings(cam_id,image,
                                                                            points=points,
                                                                            linesegs=linesegs,
                                                                            lineseg_colors=lineseg_colors,
                                                                            sort_add=True)
                    if show_fps is not None:
                        show_fps_label = xrc.XRCCTRL(previewPerCamPanel,'acquired_fps_label') # get container
                        show_fps_label.SetLabel('fps: %.1f'%show_fps)
                except KeyError:
                    pass # may have lost camera since call to service_pending


            if isinstance(event,wx.IdleEvent):
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

    def OnCloseCamera(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera

    def OnSetCameraNSigma(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.send_set_camera_property(cam_id,'n_sigma',value)

    def OnSetCameraThreshold(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = int(value)
            self.main_brain.send_set_camera_property(cam_id,'diff_threshold',value)

    def OnSetCameraClearThreshold(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.send_set_camera_property(cam_id,'clear_threshold',value)

    def OnSetMaxFramerate(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        value = event.GetString()
        if value:
            value = float(value)
            self.main_brain.send_set_camera_property(cam_id,'max_framerate',value)


    def OnSetViewImageChoice(self, event):
        cam_id = self._get_cam_id_for_button(event.GetEventObject())
        widget = event.GetEventObject()
        value = widget.GetStringSelection()
        self.main_brain.send_set_camera_property(cam_id,'visible_image_view',value)

    def OnSetTriggerModeNumber(self,event):
        widget = event.GetEventObject()
        cam_id = self._get_cam_id_for_button(widget)
        value = event.GetString()
        if value:
            value = int(value)
            self.main_brain.send_set_camera_property(cam_id,'trigger_mode',value)

    def OnOldCamera(self, cam_id):
        sys.stdout.flush()
        self.OnRecordRawStop(warn=False)

        try:
            self.cam_image_canvas.delete_image(cam_id)
        except KeyError:
            # camera never sent frame
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
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--server", dest="server", type='string',
                      help="hostname of mainbrain SERVER",
                      default='',
                      metavar="SERVER")
    parser.add_option("--disable-opengl", dest="use_opengl",
                      default=True, action="store_false")
    parser.add_option("--save-profiling-data",
                      default=False, action="store_true",
                      help="save data to profile/debug the Kalman-filter based tracker (WARNING: SLOW)",
                      )
    parser.add_option("--disable-sync-errors", dest='show_sync_errors',
                      default=True, action="store_false",
                      )

    (options, args) = parser.parse_args()

    global use_opengl
    use_opengl = options.use_opengl
    # initialize GUI
    #app = App(redirect=1,filename='flydra_log.txt')
    app = wxMainBrainApp(0)

    # create main_brain server (not started yet)
    main_brain = MainBrain.MainBrain(server=options.server,
                                     save_profiling_data=options.save_profiling_data,
                                     show_sync_errors=options.show_sync_errors,
                                     )

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
    if 0:
        # profile
        import hotshot
        prof = hotshot.Profile("profile.hotshot")
        res = prof.runcall(main)
        prof.close()
    else:
        # don't profile
        main()
