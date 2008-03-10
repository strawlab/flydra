#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import wx
import wx.lib.newevent
import camnode
import camnode_utils
import numpy
import motmot.wxglvideo.simple_overlay as wxglvideo

DisplayImageEvent, EVT_DISPLAYIMAGE = wx.lib.newevent.NewEvent()

class WxApp(wx.App):
    def OnInit(self):
        self.call_often = None
        wx.InitAllImageHandlers()
        self.frame = wx.Frame(None, -1, "camnode wx",size=(640,480))

        # menubar ------------------------------------
        menuBar = wx.MenuBar()
        #   File menu
        filemenu = wx.Menu()

        ID_quit = wx.NewId()
        filemenu.Append(ID_quit, "Quit\tCtrl-Q", "Quit application")
        wx.EVT_MENU(self, ID_quit, self.OnQuit)
        #wx.EVT_CLOSE(self, ID_quit, self.OnQuit)
        # JAB thinks this will allow use of the window-close ('x') button
        # instead of forcing users to file->quit

        menuBar.Append(filemenu, "&File")

        # finish menubar -----------------------------
        self.frame.SetMenuBar(menuBar)

        frame_box = wx.BoxSizer(wx.VERTICAL)

        self.cam_image_canvas = wxglvideo.DynamicImageCanvas(self.frame,-1)
        frame_box.Add(self.cam_image_canvas,1,wx.EXPAND)

        self.frame.SetSizer(frame_box)
        self.frame.Layout()

        self.frame.SetAutoLayout(True)

        self.frame.Show()
        self.SetTopWindow(self.frame)

        wx.EVT_CLOSE(self.frame, self.OnWindowClose)


        ID_Timer2 = wx.NewId()
        self.timer2 = wx.Timer(self, ID_Timer2)
        wx.EVT_TIMER(self, ID_Timer2, self.OnTimer2)
        self.update_interval2=50
        self.timer2.Start(self.update_interval2)

        EVT_DISPLAYIMAGE(self, self.OnDisplayImageEvent )

        return True
    def post_init(self, call_often = None):
        self.call_often = call_often

    def OnWindowClose(self, event):
        event.Skip() # propagate event up the chain...

    def OnQuit(self, dummy_event=None):
        self.frame.Close() # results in call to OnWindowClose()

    def OnTimer2(self,event):
        if self.call_often is not None:
            self.call_often()

    def quit_now(self, exit_value):
        # called from callback thread
        if exit_value != 0:
            # don't know how to make wx exit with exit value otherwise
            sys.exit(exit_value)
        else:
            # send event to app
            event = wx.CloseEvent()
            event.SetEventObject(self)
            wx.PostEvent(self, event)

    def OnDisplayImageEvent(self, event):
        #print 'got display image for %s in wx mainloop'%event.cam_id
        self.cam_image_canvas.update_image_and_drawings(event.cam_id,
                                                        event.buf,
                                                        points=event.pts,
                                                        sort_add=True)
class DisplayCamData(object):
    def __init__(self, wxapp, cam_id=None):
        self._chain = camnode_utils.ChainLink()
        self._wxapp = wxapp
        self._cam_id = cam_id
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            with camnode_utils.use_buffer_from_chain(self._chain) as buf:
                # post images and processed points to wx
                if hasattr(buf,'processed_points'):
                    pts = buf.processed_points
                else:
                    pts = None
                buf_copy = numpy.array( buf.get_buf(), copy=True )
            wx.PostEvent(self._wxapp, DisplayImageEvent(buf=buf_copy, pts=pts, cam_id=self._cam_id))
