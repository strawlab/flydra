"""object oriented, high-level wrapper of the (some of) cairo graphics library

What is a benu? A benu is a purple heron, a species of Egyptian bird
thought to be the phoenix.

"""
from __future__ import division, with_statement

import cairo
import os, warnings
import numpy as np
import contextlib

def numpy2cairo(raw):
    raw=np.asarray(raw)
    if raw.dtype!=np.uint8:
        raise ValueError('only uint8 dtype is supported')
    if raw.ndim==2:
        brga = np.ndarray(
            shape=(raw.shape[0],raw.shape[1],4),
            dtype=np.uint8)
        brga[:,:,0]=raw
        brga[:,:,1]=raw
        brga[:,:,2]=raw
        brga[:,:,3].fill(255)
    elif raw.ndim==3:
        if raw.shape[2]==3:
            brga = np.ndarray(
                shape=(raw.shape[0],raw.shape[1],4),
                dtype=np.uint8)
            brga[:,:,0]=raw[:,:,2]
            brga[:,:,1]=raw[:,:,0]
            brga[:,:,2]=raw[:,:,1]
            brga[:,:,3].fill(255)
        elif raw.shape[2]==4:
            brga = np.ndarray(
                shape=(raw.shape[0],raw.shape[1],4),
                dtype=np.uint8)
            brga[:,:,0]=raw[:,:,2]
            brga[:,:,1]=raw[:,:,0]
            brga[:,:,2]=raw[:,:,1]
            brga[:,:,3]=raw[:,:,3]
        else:
            raise ValueError('if 3d array is given, it must be RGB or RGBA')
    else:
        raise ValueError('only only 2d and 3d arrays are supported')
    in_surface = cairo.ImageSurface.create_for_data(
        brga, cairo.FORMAT_ARGB32,
        raw.shape[1], raw.shape[0],
        raw.shape[1]*4)#, raw.shape[0]*3)
    return in_surface

class Canvas(object):
    """A drawing surface which handles coordinate transforms"""
    def __init__(self,fname,width,height):
        self._output_ext = os.path.splitext(fname)[1].lower()
        if self._output_ext == '.pdf':
            output_surface = cairo.PDFSurface(fname,
                                              width, height)
        elif self._output_ext == '.svg':
            output_surface = cairo.SVGSurface(fname,
                                              width, height)
        elif self._output_ext == '.png':
            output_surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32,width, height)
        else:
            raise ValueError('unknown output extension %s'%self._output_ext)
        self._surf = output_surface
        self._ctx = cairo.Context(self._surf)
        self._fname = fname
    def imshow(self,im,l,b,filter='nearest'):
        """show image im at location (l,b)

        filter can be one of ['best', 'bilinear', 'fast', 'gaussian',
        'good', 'nearest'].

        """
        # Get cairo filter type (e.g. "cario.FILTER_NEAREST")
        cfilter = getattr(cairo,'FILTER_'+filter.upper())

        # Get cairo surface
        in_surface = numpy2cairo(im)

        ctx = self._ctx # shorthand
        #ctx.rectangle(l,b,im.shape[1],im.shape[0])
        ctx.save()
        ctx.set_source_surface(in_surface,l,b)
        ctx.get_source().set_filter(cfilter)
        ctx.paint()
        ctx.restore()

    def plot(self,xarr,yarr,color_rgba=None,close_path=False):
        """line plot of xarr vs. yarr"""
        if color_rgba is None:
            color_rgba = (1,1,1,1)
        if len(xarr)==1:
            warnings.warn('benu plot() currently only plots line segments')
        ctx = self._ctx # shorthand

        ctx.set_source_rgba(*color_rgba)
        ctx.move_to(xarr[0],yarr[0])
        for i in range(1,len(xarr)):
            ctx.line_to(xarr[i],yarr[i])
        if close_path:
            ctx.close_path()
        ctx.stroke()

    def scatter(self,xarr,yarr,color_rgba=None,radius=1.0):
        """scatter plot of xarr vs. yarr"""
        if color_rgba is None:
            color_rgba = (1,1,1,1)
        ctx = self._ctx # shorthand

        ctx.set_source_rgba(*color_rgba)
        for x,y in zip(xarr,yarr):
            #cairo_new_sub_path()
            ctx.new_sub_path()
            ctx.arc(x,y,radius,0,2*np.pi)
        ctx.stroke()

    def save(self):
        """save output to file"""
        if self._output_ext == '.png':
            self._surf.write_to_png(self._fname)
        else:
            self._ctx.show_page()
            self._surf.finish()

    def text(self,text,x,y,color_rgba=None,font_size=10):
        """draw text"""
        if color_rgba is None:
            color_rgba = (0,0,0,1)

        ctx = self._ctx # shorthand

        ctx.set_source_rgba(*color_rgba)
        ctx.select_font_face ("Sans", cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(font_size)

        ctx.move_to(x,y)
        ctx.show_text(text)

    @contextlib.contextmanager
    def set_user_coords(self, device_rect, user_rect,
                        clip=True,
                        transform='orig' ):
        """enter a benu context with a user-defined coordinate system

        **Arguments**

        device_rect : 4 element tuple
            Specifies the coordinates in device space to draw into (l,b,w,h)
        user_rect : 4 element tuple
            Specifies the coordinates in arbitrary user defined space
            mapping into device space (l,b,w,h)

        **Optional keyword arguments**

        clip : boolean
            Whether to limit drawing within the device_rect
        transform : string
            how user_rect is transformed into device_rect. One of 'orig',
            'rot -90', 'rot 180'.
        """
        user_l, user_b, user_w, user_h = user_rect
        user_r = user_l+user_w
        user_t = user_b+user_h
        device_l, device_b, device_w, device_h = device_rect

        if transform=='orig':
            xscale = device_w/user_w
            yscale = device_h/user_h
            matrix = cairo.Matrix(xx=xscale,
                                  yx=0,
                                  xy=0,
                                  yy=yscale,
                                  x0=(device_l-xscale*user_l),
                                  y0=(device_b-yscale*user_b),
                                  )
        elif transform=='rot -90':
            xscale = device_w/user_w
            yscale = device_h/user_h
            matrix = cairo.Matrix(xx=0,
                                  yx=xscale, # XXX these could be flipped
                                  xy=yscale,
                                  yy=0,
                                  x0=(device_l-xscale*user_b),
                                  y0=(device_b-yscale*user_l),
                                  )
        elif transform=='rot 180':
            xscale = device_w/user_w
            yscale = device_h/user_h
            matrix = cairo.Matrix(xx=-xscale,
                                  yx=0,
                                  xy=0,
                                  yy=-yscale,
                                  x0=(device_l+xscale*user_r),
                                  y0=(device_b+yscale*user_t),
                                  )
        else:
            raise ValueError("unknown transform '%s'"%transform)
        orig_matrix = self._ctx.get_matrix()

        self._ctx.set_matrix(matrix)
        if clip:
            self._ctx.save()
            self._ctx.rectangle(user_l,user_b,
                          user_w,user_h)
            self._ctx.clip()
            self._ctx.new_path()
        try:
            yield
        finally:
            if clip:
                self._ctx.restore()
            self._ctx.set_matrix(orig_matrix)

def test_benu():
    c = Canvas('x.png',10,10)
    # this test is broken...
    ## with c.set_user_coords(1,2):
    ##     pass


