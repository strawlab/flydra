import numarray as na
import math

from wxPython.wx import *
from wxPython.glcanvas import *
from OpenGL.GL import *

have_glue = False
try:
    from OpenGL.GLUT import *
    have_glut = True
except:
    pass

class DynamicImageCanvas(wxGLCanvas):
    def __init__(self, *args, **kw):
        wxGLCanvas.__init__(*(self,)+args, **kw)
        self.init = False
        EVT_ERASE_BACKGROUND(self, self.OnEraseBackground)
        EVT_SIZE(self, self.OnSize)
        EVT_PAINT(self, self.OnPaint)
        EVT_IDLE(self, self.OnDraw)
        self._gl_tex_info_dict = {}
        self.do_clipping = True
        self.draw_points = {}

    def set_clipping(self, value):
        self.do_clipping = value

    def get_clipping(self):
        return self.do_clipping

    def delete_image(self,id_val):
        tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight = self._gl_tex_info_dict[id_val]
        glDeleteTextures( tex_id )
        del self._gl_tex_info_dict[id_val]

    def set_draw_points(self,id_val,points):
        self.draw_points[id_val]=points

    def __del__(self, *args, **kwargs):
        for id_val in self._gl_tex_info_dict.keys():
            self.delete_image(id_val)
            
    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW. (inhereted from wxDemo)

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
        glClearColor(0.0, 1.0, 0.0, 0.0) # green
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
        glDisable(GL_DEPTH_TEST);
        glColor4f(1.0,1.0,1.0,1.0)
        glEnable( GL_POINT_SMOOTH )
        glPointSize(5)
        
    def create_texture_object(self,id_val,image):
        def next_power_of_2(f):
            return int(math.pow(2.0,math.ceil(math.log(f)/math.log(2.0))))

        height, width = image.shape
        
        width_pow2  = next_power_of_2(width)
        height_pow2  = next_power_of_2(height)
        
        buffer = na.zeros( (height_pow2,width_pow2,2), image.typecode() )+128
        buffer[0:height,0:width,0] = image

        if self.do_clipping:
            clipped = na.greater(image,254) + na.less(image,1)
            mask = na.choose(clipped, (255, 0) )
            buffer[0:height,0:width,1] = mask
        
        raw_data = buffer.tostring()

        tex_id = glGenTextures(1)

        gl_tex_xy_alloc = width_pow2, height_pow2
        gl_tex_xyfrac = width/float(width_pow2),  height/float(height_pow2)
        widthheight = width, height

        self._gl_tex_info_dict[id_val] = (tex_id, gl_tex_xy_alloc, gl_tex_xyfrac,
                                          widthheight)
        
        glBindTexture(GL_TEXTURE_2D, tex_id)
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

    def update_image(self,id_val,image):
        if id_val not in self._gl_tex_info_dict:
            self.create_texture_object(id_val,image)
            return
        height, width = image.shape
        tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight = self._gl_tex_info_dict[id_val]
        
        max_x, max_y = gl_tex_xy_alloc 
        if width > max_x or height > max_y: 
            self.delete_image(id_val) 
            self.create_texture_object(id_val,image)
        else:
            # XXX allocating new memory...
            if not hasattr(self,'_buffer') or self._buffer.shape != (height,width,2):
                self._buffer = na.zeros( (height,width,2), image.typecode() )

            if self.do_clipping:
                clipped = na.greater(image,254) + na.less(image,1)
                mask = na.choose(clipped, (255, 200) ) # alpha for transparency
                self._buffer[:,:,0] = image
                self._buffer[:,:,1] = mask
                data_format = GL_LUMINANCE_ALPHA
                buffer_string = self._buffer.tostring()
            else:
                data_format = GL_LUMINANCE
                buffer_string = image.tostring()
            
            self._gl_tex_xyfrac = width/float(max_x),  height/float(max_y)
            glBindTexture(GL_TEXTURE_2D,tex_id)
            glTexSubImage2D(GL_TEXTURE_2D, #target,
                            0, #mipmap_level,
                            0, #x_offset,
                            0, #y_offset,
                            width,
                            height,
                            data_format,
                            GL_UNSIGNED_BYTE, #data_type,
                            buffer_string)

    def OnDraw(self,*dummy_arg):
        N = len(self._gl_tex_info_dict)
        if N == 0:
            glClearColor(0.0, 0.0, 0.0, 0.0) # black
            glColor4f(0.0,1.0,0.0,1.0) # green
            glClear(GL_COLOR_BUFFER_BIT)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            glRasterPos3f(.02,.02,0)
            for char in 'no image sources':
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,ord(char))
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glClearColor(0.0, 1.0, 0.0, 0.0) # green
            glColor4f(1.0,1.0,1.0,1.0) # white
        else:
            glClear(GL_COLOR_BUFFER_BIT)
            ids = self._gl_tex_info_dict.keys()
            ids.sort()
            x_border_pixels = 1
            y_border_pixels = 1
            size = self.GetClientSize()

            x_border = x_border_pixels/float(size[0])
            y_border = y_border_pixels/float(size[1])
            hx = x_border*0.5
            hy = y_border*0.5
            x_borders = x_border*(N+1)
            y_borders = y_border*(N+1)
            for i in range(N):
                bottom = y_border
                top = 1.0-y_border
                left = (1.0-2*hx)*i/float(N)+hx+hx
                right = (1.0-2*hx)*(i+1)/float(N)-hx+hx

                tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight = self._gl_tex_info_dict[ids[i]]

                xx,yy = gl_tex_xyfrac

                glBindTexture(GL_TEXTURE_2D,tex_id)
                glBegin(GL_QUADS)
                glTexCoord2f( 0, yy) # texture is flipped upside down to fix OpenGL<->na
                glVertex2f( left, bottom)

                glTexCoord2f( xx, yy)
                glVertex2f( right, bottom)

                glTexCoord2f( xx, 0)
                glVertex2f( right, top)

                glTexCoord2f( 0, 0)
                glVertex2f( left,top)
                glEnd()

                # draw points if needed
                xg=right-left
                xo=left
                yg=top-bottom
                yo=bottom
                draw_points = self.draw_points.get(ids[i],[])
                pts_mode = False
                width = float(widthheight[0])
                height = float(widthheight[1])
                for pt in draw_points:
                    x = pt[0]/width*xg+xo
                    y = (height-pt[1])/height*yg+yo
                    if not pts_mode:
                        glDisable(GL_TEXTURE_2D)
                        glColor4f(0.0,1.0,0.0,1.0)                        
                        glBegin(GL_POINTS)
                        pts_mode = True
                    glVertex2f(x,y)
                if pts_mode:
                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)                        
                    glEnable(GL_TEXTURE_2D)

                for pt in draw_points:
                    ox0 = pt[0]
                    oy0 = pt[1]

                    angle_radians = pt[2]
                    r = 20.0
                    odx = r*math.cos( angle_radians )
                    ody = r*math.sin( angle_radians )

                    x0 = (ox0-odx)/width*xg+xo
                    x1 = (ox0+odx)/width*xg+xo
                    
                    y0 = (height-oy0-ody)/height*yg+yo
                    y1 = (height-oy0+ody)/height*yg+yo

                    glDisable(GL_TEXTURE_2D)
                    glColor4f(0.0,1.0,0.0,1.0)                        
                    glBegin(GL_LINES)
                    glVertex2f(x0,y0)
                    glVertex2f(x1,y1)
                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)                        
                    glEnable(GL_TEXTURE_2D)
                    
        self.SwapBuffers()
