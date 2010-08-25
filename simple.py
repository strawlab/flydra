import numpy as np
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, ListEditor, Label, \
     RangeEditor

from enthought.chaco.api import DataView, ArrayDataSource, ScatterPlot, \
     LinePlot, LinearMapper, ArrayPlotData, Plot, gray
from enthought.enable.component_editor import ComponentEditor

import mpl_figure_editor
import warnings, collections

xmin,xmax=-3,3
ymin,ymax=-3,3
FAR = 6
D2R = np.pi/180.

def unwrap(angle):
    return angle % (2*np.pi)

def traits2mpl_color(color):
    if hasattr(color,'red'):
        # traits.Color?
        if color.red is None:
            warnings.warn('undefined color')
            mplcolor = (0,0,0)
        else:
            mplcolor = (color.red/255.0, color.green/255.0, color.blue/255.0)
    else:
        # numpy array?
        mplcolor = np.array(color)[:3]/255.0
    return mplcolor

def sgn(x):
    if x<0:
        return -1
    return 1

def intersect_line_circle( x1=0.0, y1=0.0,
                           x2=1.0, y2=0.0,
                           r=1.0):
    # return [x,y] where x is a list of intersection points and y is, too
    # See http://mathworld.wolfram.com/Circle-LineIntersection.html
    dx = x2-x1
    dy = y2-y1
    dr = np.sqrt( dx**2 + dy**2 )
    D = x1*y2 - x2*y1

    x=[]; y=[]
    dis = r**2 * dr**2 - D**2
    if dis < 0:
        return x,y

    sqr = np.sqrt( dis )
    for switch in (1,-1):
        xi = ( D * dy + switch * sgn(dy) * dx * sqr)/ dr**2
        yi = (-D * dx + switch * abs(dy) * sqr) / dr**2
        x.append(xi)
        y.append(yi)
    return x,y

class TwoVec(traits.HasTraits):
    x = traits.Float
    y = traits.Float
    traits_view = View( Group( Item('x',
                                    editor=RangeEditor(low=xmin,high=xmax,
                                                       is_float=True)),
                               Item('y',
                                    editor=RangeEditor(low=ymin,high=ymax,
                                                       is_float=True)),
                               )
                        )

    def __add__(self,other):
        return TwoVec( x=self.x+other.x,
                       y=self.y+other.y )

    def __mul__(self,other):
        return TwoVec( x=self.x*other,
                       y=self.y*other )

    def length(self):
        return np.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        l = self.length()
        li = 1.0/l
        return self*li

    def dist(self,other):
        dirvec = self + other*-1
        return dirvec.length()

class OpticalElement(traits.HasTraits):
    _changed = traits.Event
    def updated(self):
        self._changed = True

class RayEmitter(OpticalElement):
    pos = traits.Instance(TwoVec)
    direction = traits.Float
    accum_dist = traits.Float(0)
    num_bounces = traits.Int(0)
    ray_color = traits.Color('black')

    traits_view = View( Group( Item('pos',
                                    style='custom'),
                               Item('direction',
                                    editor=RangeEditor(low=0,high=2*np.pi)),
                               ))

    @traits.on_trait_change('pos')
    def new_attr(self):
        self.on_trait_change(self.updated,'pos.anytrait')
        self.updated()

    @traits.on_trait_change('direction')
    def my_set_attr(self):
        self.updated()

    def is_along_direction(self,ipt):
        dirvec = (ipt + self.pos*-1)
        dirvec = dirvec.normalize()

        dx = np.cos(self.direction)
        dy = np.sin(self.direction)
        sumvec = dirvec + TwoVec(x=dx,y=dy)
        return sumvec.length() > 1.5

    def dist(self,elem):
        """calculate the distance between self (origin) and elem.

        A result of infinity signifies there is no intersection.
        """
        if isinstance(elem,FlatMirror):
            x1 = self.pos.x
            y1 = self.pos.y

            x2 = self.pos.x+np.cos(self.direction)
            y2 = self.pos.y+np.sin(self.direction)

            x3 = elem.end1.x
            y3 = elem.end1.y

            x4 = elem.end2.x
            y4 = elem.end2.y

            # http://en.wikipedia.org/wiki/Line-line_intersection
            x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
                ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/ \
                ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))

            ipt = TwoVec(x=x, y=y)

            if not self.is_along_direction(ipt):
                # intersection not in ray direction
                return np.inf

            mirror_vec = elem.end2 + elem.end1*-1 # vec from end1 to end2
            mirror_size = mirror_vec.length()

            e1_dist = elem.end1.dist( ipt )
            e2_dist = elem.end2.dist( ipt )
            if not (e1_dist < mirror_size and e2_dist < mirror_size):
                return np.inf

            dist = self.pos.dist(ipt)
            return dist

        elif isinstance(elem,CircularScreen):
            dx = np.cos(self.direction)
            dy = np.sin(self.direction)
            # center coords so circle is at origin
            cx = elem.pos.x
            cy = elem.pos.y
            x,y=intersect_line_circle( x1=(self.pos.x-cx),    y1=(self.pos.y-cy),
                                       x2=(self.pos.x-cx)+dx, y2=(self.pos.y-cy)+dy,
                                       r=elem.radius )
            dist = np.inf
            if len(x):
                for (xi,yi) in zip(x,y):
                    ipt = TwoVec(x=xi+cx, y=yi+cy)

                    if not self.is_along_direction(ipt):
                        # intersection not in ray direction
                        continue

                    this_dist = self.pos.dist( ipt )
                    if this_dist < dist:
                        dist = this_dist

            return dist
        elif isinstance(elem,Projector):
            return np.inf
        else:
            raise ValueError('unknown element %s'%elem)

    def draw(self,other_elements=None):
        if other_elements is None:
            other_elements = []
        dists = [ self.dist(elem) for elem in other_elements ]
        idxs = np.argsort( dists )

        i = idxs[0]
        if dists[i] < FAR:
            hit = True
        else:
            hit = False

        if hit:
            r = dists[i]
        else:
            r = FAR

        accum_dist = r + self.accum_dist

        result = []
        v = TwoVec( x=r*np.cos(self.direction),
                    y=r*np.sin(self.direction))
        a = self.pos
        b = self.pos + v

        elem = other_elements[i]
        if isinstance(elem,FlatMirror):
            mirror_vec = elem.end2 + elem.end1*-1 # vec from end1 to end2
            mirror_angle = np.arctan2( mirror_vec.y, mirror_vec.x )
            mirror_normal = mirror_angle #+ np.pi/2.0
            incidence = self.direction - mirror_normal
            reflect = mirror_normal - incidence

            if 0:
                R2D = 180/np.pi
                print
                print 'self.direction',self.direction*R2D
                print 'mirror_angle',mirror_angle*R2D
                print 'mirror_normal',mirror_normal*R2D
                print 'incidence',incidence*R2D
                print 'reflect',reflect*R2D

            if hit:
                ray = RayEmitter( pos=b,
                                  direction=reflect,
                                  accum_dist=accum_dist,
                                  num_bounces=self.num_bounces+1,
                                  ray_color=self.ray_color,
                                  )
                others2 = [e for e in other_elements if e is not elem ]
                subresult = ray.draw(other_elements=others2)
                result.extend( subresult )


        drawable = {'type':'line',
                    'x':[a.x,b.x],
                    'y':[a.y,b.y],
                    'ray_color':self.ray_color,
                    }
        if isinstance(elem,CircularScreen) and hit:
            drawable['target'] = {'instance':elem,
                                  'data':{
                'ray_length':accum_dist,
                'intercept':b,
                'num_bounces':self.num_bounces,
                'ray_color':self.ray_color,
                },
                                  }
        result.append( drawable )

        return result

def test_RayEmitter():
    re = RayEmitter( pos=TwoVec(x=0,y=0), direction=0, accum_dist=0)
    circ = CircularScreen( pos = TwoVec(x=1,y=0),radius = 1.1, angle=0)
    actual = re.dist(circ)
    expected = 2.1
    assert abs(actual-expected)<1e-6

def test_intersect_line_circle():
    x,y = intersect_line_circle( x1=-1, y1=0.0,
                                 x2=0.0, y2=0.0,
                                 r=1.1)
    expected = 0
    for actual in y:
        assert abs(actual-expected) < 1e-6

class Projector(OpticalElement):
    axis = traits.Instance(RayEmitter)
    width = traits.Float(1.0) # in radians, centered on axis
    n_rays = traits.Int(20)
    ray_color = traits.Color('black')

    traits_view = View( Group( Label('projector'),
                               Item('axis',
                                    style='custom'),
                               Item('width',
                                    editor=RangeEditor(low=0,high=2*np.pi)),
                               Item('n_rays',
                                    editor=RangeEditor(low=0,high=1000)),
                               )
                        )

    @traits.on_trait_change('axis')
    def new_attr(self):
        self.on_trait_change(self.updated,'axis.anytrait')
        self.updated()

    @traits.on_trait_change('width,n_rays')
    def my_set_attr(self):
        self.updated()

    def draw(self,*args,**kws):
        result = []
        for angle in np.linspace(-self.width/2.0, self.width/2.0,self.n_rays):
            d = unwrap(self.axis.direction + angle)
            re = RayEmitter( pos=self.axis.pos,
                             direction=d,
                             ray_color=self.ray_color )
            result.extend(re.draw(*args,**kws))
        return result

class FlatMirror(OpticalElement):
    end1 = traits.Instance(TwoVec)
    end2 = traits.Instance(TwoVec)

    traits_view = View( Group( Label('mirror'),
                               Item('end1',
                                    style='custom'),
                               Item('end2',
                                    style='custom'),
                               )
                        )

    @traits.on_trait_change('end1,end2')
    def new_attr(self):
        self.on_trait_change(self.updated,'end1.anytrait')
        self.on_trait_change(self.updated,'end2.anytrait')
        self.updated()

    def draw(self,*args,**kws):
        drawable = {'type':'line',
                    'x':[self.end1.x,self.end2.x],
                    'y':[self.end1.y,self.end2.y]}
        return [drawable]

class CircularScreen(OpticalElement):
    pos = traits.Instance(TwoVec)
    radius = traits.Float
    angle = traits.Float # in radians

    traits_view = View( Group( Label('circular screen'),
                               Item('pos',
                                    style='custom'),
                               Item('radius',
                                    editor=RangeEditor(low=0,high=10,
                                                       is_float=True)),
                               Item('angle',
                                    editor=RangeEditor(low=0,high=2*np.pi)),
                               )
                        )

    @traits.on_trait_change('pos')
    def new_attr(self):
        self.on_trait_change(self.updated,'pos.anytrait')
        self.updated()

    @traits.on_trait_change('radius,angle')
    def my_set_attr(self):
        self.updated()

    def draw(self,*args,**kws):
        n_sides = 40
        theta = (np.linspace(0,2*np.pi,n_sides) + self.angle).tolist()
        theta.append( theta[0] )
        theta = np.array(theta)
        x = self.pos.x+self.radius*np.cos(theta)
        y = self.pos.y+self.radius*np.sin(theta)
        drawable = {'type':'line','x':x,'y':y}
        return [drawable]

    def calc_quality( self, data ):
        intercept = np.array([(d.x,d.y) for d in data['intercept']])
        me = np.array( (self.pos.x,self.pos.y) )
        rel = intercept - me
        iangle = np.arctan2( rel[:,1], rel[:,0] )
        idist = np.sqrt(rel[:,1]**2 + rel[:,0]**2)
        shift_iangle = unwrap(iangle-self.angle)
        ray_length = np.array(data['ray_length'])

        result = {'angle':shift_iangle,
                  }
        for key in [k for k in data.keys() if k != 'intercept']:
            result[key] = np.array(data[key])
        return result

class MyApp(traits.HasTraits):
    elements = traits.List(traits.Instance(OpticalElement))
    target = traits.Instance(CircularScreen)
    mplfig = traits.Instance(mpl_figure_editor.Figure, ())
    quality_metric = traits.Trait( 'distance', 'intensity', 'perceived' )

    traits_view = View( Group( Item('elements',
                                    editor=ListEditor(style='custom'),
                                    height=500,
                                    width=500,
                                    ),
                               Item('quality_metric'),
                               Item('mplfig',
                                    editor=mpl_figure_editor.MPLFigureEditor(),
                                    height=300,
                                    width=500,
                                    show_label=False),
                               ),
                        title='arena layout',
                        resizable = True,
                        )
    def __init__(self,*args,**kws):
        super(MyApp,self).__init__(*args,**kws)
        #self.update_plot_view()
        self.topview_ax = self.mplfig.add_subplot(1,2,1,aspect='equal')
        self.qual_ax = self.mplfig.add_subplot(1,2,2,polar=True)
        self.topview_ax.set_xlim(xmin,xmax)
        self.topview_ax.set_ylim(ymin,ymax)

        self.topview_ax.set_xlabel('x (m)')
        self.topview_ax.set_ylabel('y (m)')

        self.qual_ax.set_title( 'quality' )

        self.mpl_lines = []
        self.update_plot_view()

    def _parse_drawables_for_target(self,drawables):
        data = None
        for drawable in drawables:
            info = drawable.get('target',None)
            if info is None: continue
            if info['instance'] is not self.target:
                continue
            if data is not None:
                for key,val in info['data'].iteritems():
                    data[key].append(val)
            else:
                data = {}
                for key,val in info['data'].iteritems():
                    data[key] = [val]
        if data is None:
            return None
        return self.target.calc_quality( data )

    def do_raytracing(self):
        drawables = []
        #assert len(self.elements)
        for i,element in enumerate(self.elements):
            others=[self.elements[j] for j in range(len(self.elements)) if j!=i]

            # get the elements to drawthemselves (i.e. do the raytracing)
            drawables.extend( element.draw( other_elements = others))
        target_info = self._parse_drawables_for_target(drawables)
        return {'drawables':drawables,'target_info':target_info}

    @traits.on_trait_change('elements._changed,quality_metric')
    def update_plot_view(self):
        if self.mplfig.canvas is None:
            # not yet initialized
            return

        for line in self.mpl_lines:
            if line._remove_method is not None:
                line._remove_method(line)

        raytracing_result = self.do_raytracing()

        self.mpl_lines = []
        for drawable in raytracing_result['drawables']:
            assert drawable['type']=='line'
            color = drawable.get('ray_color',traits.Color('black'))
            mpl_color = traits2mpl_color(color)
            mpl_line, = self.topview_ax.plot( drawable['x'],drawable['y'],
                                              '-',
                                              color=mpl_color)
            self.mpl_lines.append(mpl_line)

        info = raytracing_result['target_info']
        if info is not None:
            if self.quality_metric=='distance':
                r = info['ray_length']
            elif self.quality_metric=='intensity':
                r = 1.0/info['ray_length']**2
            elif self.quality_metric=='perceived':
                r = np.log(1.0/info['ray_length']**2)
                r -= np.min(r) # make all positive
            colors = info['ray_color']
            for (angle, ri, color) in self._sort_color( info['angle'],r,colors):
                mpl_color = traits2mpl_color(color)
                mpl_line, = self.qual_ax.plot( angle, ri, '+', color=mpl_color )
                self.mpl_lines.append(mpl_line)

        self.topview_ax.set_xlim(xmin,xmax)
        self.topview_ax.set_ylim(ymin,ymax)

        self.mplfig.canvas.draw()

    def _sort_color( self,a,b,colors):
        assert len(a)==len(colors)
        assert len(b)==len(colors)
        a_by_color = collections.defaultdict(list)
        b_by_color = collections.defaultdict(list)
        # break into dict by colors
        for i in range(len(colors)):
            color = tuple(colors[i])
            a_by_color[color].append(a[i])
            b_by_color[color].append(b[i])

        # reassemble dict into list
        results = []
        for k in a_by_color.keys():
            a = a_by_color[k]
            b = b_by_color[k]
            color = k
            results.append( (a,b,color) )
        return results

def build_arena_posts():
    m2ft=100.0/2.54/12.0
    ft2m=1.0/m2ft
    spacing = 4*ft2m
    radius = 0.02
    posts = []
    x0 = -spacing/2.0
    y0 = -spacing/2.0
    for x in [x0,x0+spacing]:
        for y in [y0,y0+spacing]:
            ## if x==x0 and y==y0:
            ##     continue
            if x<0 and y>0:
                continue
            post = CircularScreen( pos = TwoVec(x=x,y=y),
                                   radius = radius )
            posts.append(post)
    return posts

def make_projector_groups(n,r=1.5,mirrors=True):
    delta = 2*np.pi/n
    elements=[]

    colors = ['red','green','blue']
    for i in range(n):
        this_els=[]
        theta0 = unwrap(delta*i + 20*D2R)
        x = r*np.cos(theta0)
        y = r*np.sin(theta0)
        p = Projector( axis=RayEmitter( pos=TwoVec(x=x,y=y),
                                        direction=unwrap(np.pi+theta0)),
                       width=0.65,
                       ray_color = colors[i],
                       n_rays=30,
                       )
        this_els.append(p)

        odeg = 70
        for sign in (-1,+1):
            theta_offset = 180 + sign*odeg
            thetam = unwrap(theta0 + theta_offset*D2R)

            mirror_r = 1.0
            x0 = mirror_r * np.cos(thetam)
            y0 = mirror_r * np.sin(thetam)

            mirror_angle = thetam + 90*D2R - sign*30*D2R
            mirror_width = 0.5
            mdx = mirror_width*np.cos(mirror_angle)
            mdy = mirror_width*np.sin(mirror_angle)

            end1 = TwoVec(x=x0+mdx, y=y0+mdy)
            end2 = TwoVec(x=x0-mdx, y=y0-mdy)

            mirror = FlatMirror( end1 = end1, end2=end2)
            if mirrors:
                this_els.append(mirror)

        elements.extend(this_els)


    return elements

if __name__=='__main__':

    elements = make_projector_groups(3,r=1.73,mirrors=False)
    main_screen = CircularScreen( pos = TwoVec(x=0,y=0),
                                  radius = 0.5,
                                  angle = 0.0,
                                  )
    elements.append(main_screen)
    elements.extend( build_arena_posts() )
    setup = MyApp(elements=elements,target=main_screen)

    if 1:
        setup.do_raytracing()

    if 1:
        setup.configure_traits()

