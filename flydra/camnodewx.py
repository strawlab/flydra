#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import wx
import camnode
import camnode_utils

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

class DisplayCamData(object):
    def __init__(self):
        self._chain = camnode_utils.ChainLink()
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            buf= self._chain.get_buf()
            camnode.stdout_write('D')
            self._chain.end_buf(buf)

