# $Id$

import numarray as nx
#import Numeric as nx
import math

import imops

from wxPython.wx import *
from wxPython.glcanvas import *
from OpenGL.GL import *
from common_variables import MINIMUM_ECCENTRICITY

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
        self.rotate_180 = False
        self.flip_LR = False
        self.lbrt = {}
        self.draw_points = {}
        self.reconstructed_points = {}
        self.do_draw_points = True
        self.x_border_pixels = 1
        self.y_border_pixels = 1

    def set_clipping(self, value):
        self.do_clipping = value

    def set_rotate_180(self, value):
        self.rotate_180 = value

    def set_flip_LR(self, value):
        self.flip_LR = value

    def get_clipping(self):
        return self.do_clipping

    def set_display_points(self, value):
        self.do_draw_points = value

    def get_display_points(self):
        return self.do_draw_points

    def delete_image(self,id_val):
        tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight, internal_format, data_format = self._gl_tex_info_dict[id_val]
        glDeleteTextures( tex_id )
        del self._gl_tex_info_dict[id_val]

    def set_draw_points(self,id_val,points):
        self.draw_points[id_val]=points

    def set_reconstructed_points(self,id_val,points):
        self.reconstructed_points[id_val]=points

    def set_lbrt(self,id_val,lbrt):
        self.lbrt[id_val]=lbrt

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

        height, width = image.shape[:2]
        if len(image.shape) == 3:
            assert image.shape[2] == 3 # only support for RGB now...
            has_color = True
        else:
            has_color = False
        
        width_pow2  = next_power_of_2(width)
        height_pow2  = next_power_of_2(height)

        if not has_color:
            buffer = nx.zeros( (height_pow2,width_pow2,2), image.typecode() )+128
            buffer[0:height,0:width,0] = image

            if self.do_clipping:
                clipped = nx.greater(image,254) + nx.less(image,1)
                mask = nx.choose(clipped, (255, 0) )
                buffer[0:height,0:width,1] = mask
                
            internal_format = GL_LUMINANCE_ALPHA
            data_format = GL_LUMINANCE_ALPHA
        else:
            buffer = nx.zeros( (height_pow2,width_pow2,image.shape[2]), image.typecode() )
            buffer[0:height, 0:width, :] = image
            internal_format = GL_RGB
            data_format = GL_RGB

        raw_data = buffer.tostring()

        tex_id = glGenTextures(1)

        gl_tex_xy_alloc = width_pow2, height_pow2
        gl_tex_xyfrac = width/float(width_pow2),  height/float(height_pow2)
        widthheight = width, height

        self._gl_tex_info_dict[id_val] = (tex_id, gl_tex_xy_alloc, gl_tex_xyfrac,
                                          widthheight, internal_format, data_format)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, # target
                     0, #mipmap_level
                     internal_format,
                     width_pow2,
                     height_pow2,
                     0, #border,
                     data_format,
                     GL_UNSIGNED_BYTE, #data_type,
                     raw_data)

    def update_image(self,id_val,image,format='MONO8',xoffset=0,yoffset=0):
        if format == 'YUV411':
            image = imops.yuv411_to_rgb8( image )
        elif format == 'YUV422':
            image = imops.yuv422_to_rgb8( image )
        elif format == 'MONO8':
            pass
        else:
            raise ValueError("Unknown format '%s'"%(format,))
        
        if id_val not in self._gl_tex_info_dict:
            self.create_texture_object(id_val,image)
            return
        height, width = image.shape[:2]
        tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight, internal_format, data_format = self._gl_tex_info_dict[id_val]
        
        max_x, max_y = gl_tex_xy_alloc 
        if width > max_x or height > max_y: 
            self.delete_image(id_val) 
            self.create_texture_object(id_val,image)
        else:
            if len(image.shape) == 3:
                # color image
                buffer_string = image.tostring()
            else:
                # XXX allocating new memory...
                if not hasattr(self,'_buffer') or self._buffer.shape != (height,width,2):
                    self._buffer = nx.zeros( (height,width,2), image.typecode() )

                if self.do_clipping:
                    clipped = nx.greater(image,254).astype(nx.UInt8) + nx.less(image,1).astype(nx.UInt8)
                    mask = nx.choose(clipped, (255, 200) ).astype(nx.UInt8) # alpha for transparency
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
                            xoffset, #x_offset,
                            yoffset, #y_offset,
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
##            for char in 'no image sources':
##                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,ord(char))
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glClearColor(0.0, 1.0, 0.0, 0.0) # green
            glColor4f(1.0,1.0,1.0,1.0) # white
        else:
            glClear(GL_COLOR_BUFFER_BIT)
            ids = self._gl_tex_info_dict.keys()
            ids.sort()
            size = self.GetClientSize()

            x_border = self.x_border_pixels/float(size[0])
            y_border = self.y_border_pixels/float(size[1])
            hx = x_border*0.5
            hy = y_border*0.5
            x_borders = x_border*(N+1)
            y_borders = y_border*(N+1)
            for i in range(N):
                bottom = y_border
                top = 1.0-y_border
                left = (1.0-2*hx)*i/float(N)+hx+hx
                right = (1.0-2*hx)*(i+1)/float(N)-hx+hx

                tex_id, gl_tex_xy_alloc, gl_tex_xyfrac, widthheight, internal_format, data_format = self._gl_tex_info_dict[ids[i]]

                xx,yy = gl_tex_xyfrac

                glBindTexture(GL_TEXTURE_2D,tex_id)
                if self.flip_LR:
                    left,right=right,left
                if not self.rotate_180:
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
                else:
                    glBegin(GL_QUADS)
                    glTexCoord2f( xx, 0)
                    glVertex2f( left, bottom)

                    glTexCoord2f( 0, 0)
                    glVertex2f( right, bottom)

                    glTexCoord2f( 0, yy)
                    glVertex2f( right, top)

                    glTexCoord2f( xx, yy)
                    glVertex2f( left,top)
                    glEnd()
                    

                if self.do_draw_points:
                    # draw points if needed
                    xg=right-left
                    xo=left
                    yg=top-bottom
                    yo=bottom
                    
                    draw_points = self.draw_points.get(ids[i],[])
                    
                    glDisable(GL_TEXTURE_2D)
                    glColor4f(0.0,1.0,0.0,1.0) # green point
                    glBegin(GL_POINTS)

                    width = float(widthheight[0])
                    height = float(widthheight[1])
                    
                    for pt in draw_points:
                        if not pt[9]: # found_anything is false
                            continue
                        x = pt[0]/width*xg+xo
                        y = (height-pt[1])/height*yg+yo
                        glVertex2f(x,y)
                        
                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)                        
                    glEnable(GL_TEXTURE_2D)

                    for draw_point in draw_points:
                        found_anything = draw_point[9]
                        if not found_anything:
                            continue
                        
                        ox0,oy0,area,slope,eccentricity = draw_point[:5]
                        
                        if eccentricity <= MINIMUM_ECCENTRICITY:
                            # don't draw green lines -- not much orientation info
                            continue
                        
                        xmin = 0
                        ymin = 0
                        xmax = width-1
                        ymax = height-1
                        
                        # ax+by+c=0
                        a=slope
                        b=-1
                        c=oy0-a*ox0
                        
                        x1=xmin
                        y1=-(c+a*x1)/b
                        if y1 < ymin:
                            y1 = ymin
                            x1 = -(c+b*y1)/a
                        elif y1 > ymax:
                            y1 = ymax
                            x1 = -(c+b*y1)/a

                        x2=xmax
                        y2=-(c+a*x2)/b
                        if y2 < ymin:
                            y2 = ymin
                            x2 = -(c+b*y2)/a
                        elif y2 > ymax:
                            y2 = ymax
                            x2 = -(c+b*y2)/a                

                        x1 = x1/width*xg+xo
                        x2 = x2/width*xg+xo

                        y1 = (height-y1)/height*yg+yo
                        y2 = (height-y2)/height*yg+yo

                        glDisable(GL_TEXTURE_2D)
                        glColor4f(0.0,1.0,0.0,1.0) # green line
                        glBegin(GL_LINES)
                        glVertex2f(x1,y1)
                        glVertex2f(x2,y2)
                        glEnd()
                        glColor4f(1.0,1.0,1.0,1.0)                        
                        glEnable(GL_TEXTURE_2D)

                    # reconstructed points
                    draw_points, draw_lines = self.reconstructed_points.get(ids[i],([],[]))

                    # draw points
                    glDisable(GL_TEXTURE_2D)
                    glColor4f(1.0,0.0,0.0,0.5) # red points
                    glBegin(GL_POINTS)

                    for pt in draw_points:
                        if pt[0] < 0 or pt[0] >= width or pt[1] < 0 or pt[1] >= height:
                            continue
                        x = pt[0]/width*xg+xo
                        y = (height-pt[1])/height*yg+yo
                        glVertex2f(x,y)
                        
                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)                        
                    glEnable(GL_TEXTURE_2D)

                    # draw lines

                    glDisable(GL_TEXTURE_2D)
                    glColor4f(1.0,0.0,0.0,0.5) # red lines
                    glBegin(GL_LINES)

                    for L in draw_lines:
                        pt=[0,0]
                        a,b,c=L
                        # ax+by+c=0
                        # y = -(c+ax)/b
                        
                        # at x=0:
                        pt[0] = 0
                        pt[1] = -(c+a*pt[0])/b
                        
                        x = pt[0]/width*xg+xo
                        y = (height-pt[1])/height*yg+yo
                        glVertex2f(x,y)

                        # at x=width-1:
                        pt[0] = width-1
                        pt[1] = -(c+a*pt[0])/b
                        
                        x = pt[0]/width*xg+xo
                        y = (height-pt[1])/height*yg+yo
                        glVertex2f(x,y)

                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)                        
                    glEnable(GL_TEXTURE_2D)

                if ids[i] in self.lbrt:
                    # draw ROI
                    l,b,r,t = self.lbrt[ids[i]]
                    if not self.rotate_180:
                        l = l/width*xg+xo
                        r = r/width*xg+xo
                        b = (height-b)/height*yg+yo
                        t = (height-t)/height*yg+yo
                    else:
                        l = (width-l)/width*xg+xo
                        r = (width-r)/width*xg+xo
                        b = b/height*yg+yo
                        t = t/height*yg+yo                        

                    glDisable(GL_TEXTURE_2D)
                    glColor4f(0.0,1.0,0.0,1.0)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glBegin(GL_QUADS)
                    glVertex2f(l,b)
                    glVertex2f(l,t)
                    glVertex2f(r,t)
                    glVertex2f(r,b)
                    glEnd()
                    glColor4f(1.0,1.0,1.0,1.0)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glEnable(GL_TEXTURE_2D)
        
        self.SwapBuffers()
