#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import pygame
import numpy
import camnode
import camnode_utils
import sys,Queue, threading

class SdlApp():
    def __init__(self,call_often=None,num_cameras=None):
        self._queue = Queue.Queue()
        self.call_often = call_often
        self._quit_now = threading.Event()

    def generate_view(self, model, controller ):
        if hasattr(controller, 'trigger_single_frame_start' ):
            raise NotImplementedError('No support for playback controller.')

    def render_frame(self,cam_id, buf):
        """this is called from the processing thread"""
        self._queue.put( (cam_id,buf) )
    def OnQuit(self, event):
        self._quit_now.set()
    def MainLoop(self):
        pygame.init()
        screen_size = 800,600
        pygame.display.set_mode((screen_size))
        cam_id2ypos = {}
        cam_id2ovl = {}
        ovl = None
        while not self._quit_now.isSet():
            try:
                result = self._queue.get(True,0.05) # block, timeout

                if 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                cam_id, image = result
                if cam_id not in cam_id2ypos:
                    cam_ids = cam_id2ypos.keys() + [cam_id]
                    yinc = screen_size[1]//len(cam_ids)
                    cam_ids.sort()
                    for i,c in enumerate(cam_ids):
                        cam_id2ypos[c] = i*yinc
                ypos = cam_id2ypos[cam_id]
                if cam_id not in cam_id2ovl:
                    h,w = image.shape
                    #ovl= pygame.Overlay(pygame.YV12_OVERLAY, (w,h))
                    ovl= pygame.Overlay(pygame.IYUV_OVERLAY, (w,h))
                    ovl.set_location(0,ypos,w,h)
                    cam_id2ovl[cam_id] = ovl
                ovl = cam_id2ovl[cam_id]
                # this should be uint8 MONO8
                y = image.tostring()
                uv_size = len(y)//4
                u = chr(128)*uv_size
                v = u
                ovl.display( (y,u,v) )

            except Queue.Empty, err:
                # no worries if no new frame info
                pass
            self.call_often()
        pygame.quit()

class DisplayCamData(object):
    def __init__(self, sdlapp,
                 cam_id=None,
                 ):
        self._chain = camnode_utils.ChainLink()
        self._sdlapp = sdlapp
        self._cam_id = cam_id
    def get_chain(self):
        return self._chain
    def mainloop(self):
        while 1:
            with camnode_utils.use_buffer_from_chain(self._chain) as chainbuf:
                if chainbuf.quit_now:
                    # XXX TODO: Send done event to GUI.
                    print 'TODO: send quit event to GUI for cam_id %s'%(self._cam_id)
                    break
                # TODO: display pts
                buf_copy = numpy.array( chainbuf.get_buf(), copy=True )
                self._sdlapp.render_frame( self._cam_id, buf_copy )
