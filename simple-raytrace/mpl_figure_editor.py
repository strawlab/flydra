import wx

import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from enthought.traits.api import Any, Instance, ListInstance
from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.wx.basic_editor_factory import BasicEditorFactory

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Event
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, InstanceEditor, ListEditor

import numpy as np

class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor

class CanBeDirty(HasTraits):
    dirty = traits.Event

class Drawable(HasTraits):
    axes = Instance(Axes)
    viewer = Instance(CanBeDirty)

class Coord(HasTraits):
    y = Float(0)
    z = Float(0)

    traits_view = View(Item('y'),
                       Item('z'))

    def __repr__(self):
        return 'Coord(%s,%s)'%(repr(self.y),repr(self.z))

    def __sub__(self,other):
        return Coord( y=self.y-other.y,
                      z=self.z-other.z)

    def __add__(self,other):
        return Coord( y=self.y+other.y,
                      z=self.z+other.z)

    def rotate(self,theta):
        y = self.y*np.cos(theta) + -self.z*np.sin(theta)
        z = self.y*np.sin(theta) +  self.z*np.cos(theta)
        return Coord( y=y, z=z)

def test_coord():
    for start,angle,stop in [ [(1,0), np.pi, (-1,0)],
                              [(0,1), np.pi, (0,-1)],
                              [(1,2), np.pi/2, (-2,1)],
                              ]:
        a = Coord(y=start[0],z=start[1])
        b=a.rotate(angle)
        bv = np.array([b.y,b.z])
        assert np.allclose(bv,[stop[0],stop[1]])

def solve_r(a,b,c,d):
    """given points at (a,b) and (c,d), find r such that (r,0) is the
    reflection point"""

    def err_func(r):
        1

## def hack_calc_reflection_points(a,b,mirrors):
##     assert len(mirrors)==2
##     m1,m2 = mirrors

##     def err_func(psi):
##         m1n = m1.theta + np.pi/2
##         a1 = 

class MirrorLink(Drawable):
    a = Coord()
    b = Coord()
    line = Instance(mlines.Line2D)
    mirrors = traits.List(HasTraits)

    def __init__(self,*args,**kwargs):
        super(MirrorLink,self).__init__(*args,**kwargs)
        self.a.on_trait_change(self.update, '+')
        self.b.on_trait_change(self.update, '+')
        for mirror in self.mirrors:
            mirror.on_trait_change(self.update, 'origin,theta_degrees')
        self._reflection_dist = {}
        self.update()

    @on_trait_change('mirror')
    def new_mirror(self):
        for mirror in self.mirrors:
            mirror.on_trait_change(self.update, 'origin,theta_degrees')
        self.update()

    def get_reflection_dist(self,what):
        return self._reflection_dist[what]

    def update(self):
        if self.line is not None:
            if self.line._remove_method is not None:
                self.line._remove_method(self.line)

        ys = []
        zs = []

        self._reflection_dist.clear()

        if len(self.mirrors)==0:
            return
        elif len(self.mirrors)==1:
            mirror = self.mirrors[0]
            m,r = mirror.calc_reflection_point(self.a,self.b)
            self._reflection_dist[mirror] = r
            print 'm',m
            print 'r',r
            ys.append(m.y)
            zs.append(m.z)
        else:
            ms,rs = hack_calc_reflection_points(self.a,self.b,self.mirrors)
            for m in ms:
                ys.append( m.y )
                zs.append( m.z )
            for r,mirror in zip(rs,self.mirrors):
                self._reflection_dist[mirror] = r

        if self.axes is not None:
            ys.insert(0,self.a.y)
            ys.append(self.b.y)
            zs.insert(0,self.a.z)
            zs.append(self.b.z)
            self.line, = self.axes.plot( ys,zs, 'g:')

        for mirror in self.mirrors:
            mirror.update() # re-draw (kinda hacky)

        if self.viewer is not None:
            self.viewer.dirty = True

class Mirror(Drawable):
    name = traits.String()
    origin = Instance(Coord)
    theta_degrees = Range(0.0,180.0)
    theta = traits.Property(depends_on='theta_degrees')

    origin_patch = Instance(mpatches.Patch)
    mirror_patch = Instance(mpatches.Patch)
    links = traits.List(MirrorLink)

    traits_view = View(Item('name',style='readonly'),
                       Item('origin',
                            editor=InstanceEditor(),
                            style='custom'),
                       Item('theta_degrees'),
                       resizable=True)

    def __init__(self,*args,**kwargs):
        super(Mirror,self).__init__(*args,**kwargs)

    def _get_theta(self):
        D2R = np.pi/180.0
        return self.theta_degrees*D2R

    def calc_reflection_point(self,a,b):
        Ah = a-self.origin
        Bh = b-self.origin
        A = Ah.rotate(-self.theta)
        B = Bh.rotate(-self.theta)

        a,b=A.y,A.z
        c,d=B.y,B.z
        r=(d*a+b*c)/(d+b)
        #r=solve_r(a,b,c,d)
        C = Coord(y=r,z=0)
        Ch = C.rotate(self.theta)
        Ch2 = Ch+self.origin
        return Ch2,r

    @on_trait_change('origin')
    def set_origin(self):
        self.origin.on_trait_change(self.update, '+')

    @on_trait_change('theta_degrees')
    def update(self):
        if self.origin_patch is not None:
            if self.origin_patch._remove_method is not None:
                self.origin_patch._remove_method(self.origin_patch)

        if self.mirror_patch is not None:
            if self.mirror_patch._remove_method is not None:
                self.mirror_patch._remove_method(self.mirror_patch)

        self.origin_patch = mpatches.CirclePolygon(
            (self.origin.y,self.origin.z),
            radius=1.0 )

        if len(self.links)<2:
            return

        rs = np.array([link.get_reflection_dist(self) for link in self.links])
        print 'rs',rs
        r0 = np.min(rs)
        r1 = np.max(rs)
        print 'r0,r1',r0,r1

        c0 = Coord(y=r0,z=0)
        c1 = Coord(y=r1,z=0)
        c0 = c0.rotate(self.theta)
        c1 = c1.rotate(self.theta)
        c0 = c0+self.origin
        c1 = c1+self.origin
        self.mirror_patch = mpatches.PathPatch(mpath.Path([(c0.y,c0.z),(c1.y,c1.z)]))

        self.origin_patch = mpatches.CirclePolygon(
            (self.origin.y,self.origin.z),
            radius=1.0 )

        if self.axes is not None:
            if self.origin_patch is not None:
                self.axes.add_patch(self.origin_patch)
            if self.mirror_patch is not None:
                self.axes.add_patch(self.mirror_patch)

        if self.viewer is not None:
            self.viewer.dirty = True

class Projector(Drawable):
    center = Coord(y=-80, z=5)
    patch = Instance(mpatches.Patch)

    traits_view = View(Item('center',
                            editor=InstanceEditor(),
                            style='custom'),
                       resizable=True)

    def __init__(self,*args,**kwargs):
        super(Projector,self).__init__(*args,**kwargs)
        self.center.on_trait_change(self.update, '+')

    def get_location(self,name):
        if name=='center':
            return self.center
        else:
            raise KeyError('unknown location %s'%name)

    @on_trait_change('center')
    def update(self):
        if self.patch is not None:
            if self.patch._remove_method is not None:
                self.patch._remove_method(self.patch)

        self.patch = mpatches.CirclePolygon( (self.center.y,self.center.z),
                                             radius=5.0 )
        if self.axes is not None:
            self.axes.add_patch(self.patch)

        if self.viewer is not None:
            self.viewer.dirty = True

class WindTunnel(Drawable):
    line = Instance(mlines.Line2D)
    w = Float(30)
    h = Float(30)
    center = Coord()

    floor_a = Coord()
    floor_b = Coord()
    yminus_a = Coord()
    yminus_b = Coord()

    moved = traits.Event

    traits_view = View(Item('center',
                            editor=InstanceEditor(),
                            style='custom'),
                       Item('w'),
                       Item('h'),
                       resizable=True)

    def __init__(self,*args,**kwargs):
        super(WindTunnel,self).__init__(*args,**kwargs)
        self.center.on_trait_change(self.update, '+')
        self.moved=True # fire event once

    def draw_orig(self):
        if self.axes is not None:
            if self.center is None:
                return
            if self.w is None:
                return
            if self.h is None:
                return
            self.line, = self.axes.plot( [self.center.y-self.w/2.0,
                                          self.center.y-self.w/2.0,
                                          self.center.y+self.w/2.0,
                                          self.center.y+self.w/2.0,],
                                         [self.center.z+self.h,
                                          self.center.z,
                                          self.center.z,
                                          self.center.z+self.h],
                                         'k', lw=2 )

    def _moved_fired(self):
        print self, 'on_coord_change'

        self.floor_a.y = self.center.y - self.w/2.0
        self.floor_a.z = self.center.z

        self.floor_b.y = self.center.y + self.w/2.0
        self.floor_b.z = self.center.z

        self.yminus_a.y = self.center.y - self.w/2.0
        self.yminus_a.z = self.center.z

        self.yminus_b.y = self.center.y - self.w/2.0
        self.yminus_b.z = self.center.z + self.h

    @on_trait_change('w,h,center,axes,viewer')
    def update(self,arg1,arg2,arg3,arg4):
        print '+ changed',arg1,arg2,arg3,arg4
        if arg1 in [self.center,self.w,self.h]:
            print 'moved true'
            self.moved=True
        if self.line is None:
            self.draw_orig()
        if self.line is None:
            # not ready yet
            return

        xdata = [self.center.y-self.w/2.0,
                 self.center.y-self.w/2.0,
                 self.center.y+self.w/2.0,
                 self.center.y+self.w/2.0,]
        ydata = [self.center.z+self.h,
                 self.center.z,
                 self.center.z,
                 self.center.z+self.h]

        self.line.set_xdata(xdata)
        self.line.set_ydata(ydata)

        if self.viewer is not None:
            self.viewer.dirty = True

class MyApp(CanBeDirty):

    figure = Instance(Figure, ())
    wind_tunnel = Instance(WindTunnel)
    projector = Instance(Projector)
    mirrors = traits.List(Mirror)

    traits_view = View(Item('figure', editor=MPLFigureEditor(),
                            show_label=False),
                       Item('wind_tunnel', style='custom'),
                       Item('projector', style='custom'),
                       Item('mirrors', editor=ListEditor(use_notebook=True),
                            style='custom'),
                       width=800,
                       height=800,
                       resizable=True)

    def __init__(self):
        super(MyApp, self).__init__()
        self.axes = self.figure.add_subplot(111,aspect='equal')
        #self.axes.grid(True)

        self.wind_tunnel= WindTunnel(axes=self.axes,viewer=self)

        self.projector= Projector(axes=self.axes,viewer=self)
        self.projector.update() # draw

        if 1:
            # floor mirror
            mirror = Mirror(
                origin=Coord(y=0,z=-50,),
                theta_degrees=45,
                name='floor',
                axes=self.axes,
                viewer=self)
            self.mirrors.append(mirror)
            link = MirrorLink(a=self.projector.center,
                              b=self.wind_tunnel.floor_a,
                              mirrors=[mirror],
                              axes=self.axes,
                              viewer=self)
            mirror.links.append( link )

            link = MirrorLink(a=self.projector.center,
                              b=self.wind_tunnel.floor_b,
                              mirrors=[mirror],
                              axes=self.axes,
                              viewer=self)
            mirror.links.append( link )

            mirror.update() #draw

        if 0:
            # ym mirrors
            top_mirror = Mirror(
                origin=Coord(y=-50,z=0,),
                theta_degrees=135,
                name='yminus top',
                axes=self.axes,
                viewer=self)
            self.mirrors.append(top_mirror)

            floor_mirror = Mirror(
                origin=Coord(y=-50,z=-50,),
                theta_degrees=45,
                name='yminus floor',
                axes=self.axes,
                viewer=self)
            self.mirrors.append(floor_mirror)

            # links
            link1 = MirrorLink(a=self.projector.center,
                               b=self.wind_tunnel.yminus_a,
                               mirrors=[top_mirror,floor_mirror],
                               axes=self.axes,
                               viewer=self)
            top_mirror.links.append(link1)
            floor_mirror.links.append(link1)

            link2 = MirrorLink(a=self.projector.center,
                               b=self.wind_tunnel.yminus_b,
                               mirrors=[top_mirror,floor_mirror],
                               axes=self.axes,
                               viewer=self)
            top_mirror.links.append(link2)
            floor_mirror.links.append(link2)

            # draw
            top_mirror.update()
            floor_mirror.update()


        self.axes.set_xlim(-150,100)
        self.axes.set_ylim(-150,50)

    def update_plot(self):
        if self.figure.canvas is None:
            # not yet initialized
            return
        self.figure.canvas.draw()

    def _dirty_fired(self):
        self.update_plot()

if __name__ == "__main__":
    MyApp().configure_traits()
