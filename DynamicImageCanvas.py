import numarray
import math

from wxPython.wx import *
from wxPython.glcanvas import *
from OpenGL.GL import *

class DynamicImageCanvas(wxGLCanvas):
    def __init__(self, *args, **kw):
        wxGLCanvas.__init__(*(self,)+args, **kw)
        self.init = False
        EVT_ERASE_BACKGROUND(self, self.OnEraseBackground)
        EVT_SIZE(self, self.OnSize)
        EVT_PAINT(self, self.OnPaint)
        EVT_IDLE(self, self.OnDraw)
        self._gl_tex_obj = None

    def __del__(self, *args, **kwargs):
        if self._gl_tex_obj is not None:
            glDeleteTextures( self._gl_tex_obj )
            self._gl_tex_obj = None
        #wxGLCanvas.__del__(*(self,)+args,**kwargs) # doesn't exist
            
    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnSize(self, event):
        size = self.GetClientSize()
        if self.GetContext():
            self.SetCurrent()
            glViewport(0, 0, size.width, size.height)

    def OnPaint(self, event):
        dc = wxPaintDC(self)
        self.SetCurrent()
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()
        
    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,1,0,1,-1,1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable( GL_BLEND )
        glClearColor(0.0, 0.0, 1.0, 0.0) # blue
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
        self.tex_xyfrac = 1,1
        glDisable(GL_DEPTH_TEST);
        glColor4f(1.0,1.0,1.0,1.0)
        
##        image = numarray.zeros( (320,240), numarray.UInt8 )
##        image = image + 20
##        image[20:,20:30] = 255
##        image[20:,60:70] = 254
##        image[20:,80:90] = 0
##        image[20:,100:110] = 1
#        self.create_texture_object(image)
        
    def create_texture_object(self,image):
        
        def next_power_of_2(f):
            return int(math.pow(2.0,math.ceil(math.log(f)/math.log(2.0))))

        height, width = image.shape
        
        width_pow2  = next_power_of_2(width)
        height_pow2  = next_power_of_2(height)
        
        buffer = numarray.zeros( (height_pow2,width_pow2,2), image.typecode() )+128
        buffer[0:height,0:width,0] = image
        
        clipped = numarray.greater(image,254) + numarray.less(image,1)
        mask = numarray.choose(clipped, (255, 0) )
        buffer[0:height,0:width,1] = mask
        
        raw_data = buffer.tostring()

        if self._gl_tex_obj is None:
            self._gl_tex_obj = glGenTextures(1)
            
        self._gl_tex_xy_alloc = width_pow2, height_pow2
        self._gl_tex_xyfrac = width/float(width_pow2),  height/float(height_pow2)
        
        glBindTexture(GL_TEXTURE_2D, self._gl_tex_obj)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, # target
                     0, #mipmap_level
                     GL_LUMINANCE_ALPHA, #internal_format,
                     width_pow2,
                     height_pow2,
                     0, #border,
                     GL_LUMINANCE_ALPHA, #data_format,
                     GL_UNSIGNED_BYTE, #data_type,
                     raw_data);

    def update_texture(self,image):
        if self._gl_tex_obj is None:
            self.create_texture_object(image)
        height, width = image.shape
        max_x, max_y = self._gl_tex_xy_alloc
        if width > max_x or height > max_y:
            self.create_texture_object(image)
        else:
            buffer = numarray.zeros( (height,width,2), image.typecode() )+128
            buffer[:,:,0] = image
            clipped = numarray.greater(image,254) + numarray.less(image,1)
            mask = numarray.choose(clipped, (255, 0) )
            buffer[:,:,1] = mask
            self._gl_tex_xyfrac = width/float(max_x),  height/float(max_y)
            glBindTexture(GL_TEXTURE_2D,self._gl_tex_obj)
            glTexSubImage2D(GL_TEXTURE_2D, #target,
                            0, #mipmap_level,
                            0, #x_offset,
                            0, #y_offset,
                            width,
                            height,
                            GL_LUMINANCE_ALPHA, #data_format,
                            GL_UNSIGNED_BYTE, #data_type,
                            buffer.tostring())

    def OnDraw(self,*dummy_arg):
        if self._gl_tex_obj is None:
            return

        # clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        xx,yy = self._gl_tex_xyfrac

        low = 0.0
        high = 1.0
        
        glBindTexture(GL_TEXTURE_2D,self._gl_tex_obj)
        glBegin(GL_QUADS)
        glTexCoord2f( 0, yy) # texture is flipped upside down to fix OpenGL<->numarray
        glVertex2f( low, low)
        
        glTexCoord2f( xx, yy)
        glVertex2f( high, low)
        
        glTexCoord2f( xx, 0)
        glVertex2f( high, high)
        
        glTexCoord2f( 0, 0)
        glVertex2f( low,high)
        glEnd()
        
        self.SwapBuffers()
        
