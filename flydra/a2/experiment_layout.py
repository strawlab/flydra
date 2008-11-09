import conditions_draw
import numpy
from numpy import array
from enthought.tvtk.api import tvtk
import warnings

def cylindrical_arena(info=None):
    assert numpy.allclose(info['axis'],numpy.array([0,0,1])), "only vertical areas supported at the moment"

    N = 128
    theta = numpy.linspace(0,2*numpy.pi,N)
    r = info['diameter']/2.0
    xs = r*numpy.cos( theta ) + info['origin'][0]
    ys = r*numpy.sin( theta ) + info['origin'][1]

    z_levels = numpy.linspace(info['origin'][2],info['origin'][2]+info['height'],5)

    verts = []
    vi = 0 # vert idx
    lines = []

    for z in z_levels:
        zs = z*numpy.ones_like(xs)
        v = numpy.array([xs,ys,zs]).T
        for i in range(N):
            verts.append( v[i] )

        for i in range(N-1):
            lines.append( [i+vi,i+1+vi] )
        lines.append( [vi+N-1,vi] )

        vi += (N)
    pd = tvtk.PolyData()
    pd.points = verts
    pd.lines = lines
    pt = tvtk.TubeFilter(radius=0.001,input=pd,
                         number_of_sides=4,
                         vary_radius='vary_radius_off',
                         )
    m = tvtk.PolyDataMapper(input=pt.output)
    a = tvtk.Actor(mapper=m)
    a.property.color = .9, .9, .9
    a.property.specular = 0.3
    return [a]

def cylindrical_post(info=None):
    verts=info['verts']
    diameter=info['diameter']

    radius = diameter/2.0
    actors = []
    verts = numpy.asarray(verts)
    pd = tvtk.PolyData()

    np = len(verts) - 1
    lines = numpy.zeros((np, 2), numpy.int64)
    lines[:,0] = numpy.arange(0, np-0.5, 1, numpy.int64)
    lines[:,1] = numpy.arange(1, np+0.5, 1, numpy.int64)

    pd.points = verts
    pd.lines = lines

    pt = tvtk.TubeFilter(radius=radius,input=pd,
                         number_of_sides=20,
                         vary_radius='vary_radius_off',
                         )
    m = tvtk.PolyDataMapper(input=pt.output)
    a = tvtk.Actor(mapper=m)
    a.property.color = 0,0,0
    a.property.specular = 0.3
    actors.append(a)
    return actors

def cubic_arena(info=None,hack_postmultiply=None):
    tube_radius = info['tube_diameter']/2.0

    def make_2_vert_tube(a,b):
        pd = tvtk.PolyData()
        pd.points = [ a,b ]
        pd.lines = [ [0,1] ]
        pt = tvtk.TubeFilter(radius=tube_radius,
                             input=pd,
                             number_of_sides=20,
                             vary_radius='vary_radius_off',
                             )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0,0,0
        a.property.specular = 0.3
        return a

    v=info['verts4x4'] # arranged in 2 rectangles of 4 verts
    if hack_postmultiply is not None:
        warnings.warn('Using postmultiplication hack')
        vnew=[]
        for vi in v:
            vi2 = numpy.array((vi[0], vi[1], vi[2], 1.0))
            vi = numpy.dot( hack_postmultiply, vi2 )
            vnew.append(vi)
        v = vnew

    actors = []

    # bottom rect
    actors.append( make_2_vert_tube( v[0], v[1] ))
    actors.append( make_2_vert_tube( v[1], v[2] ))
    actors.append( make_2_vert_tube( v[2], v[3] ))
    actors.append( make_2_vert_tube( v[3], v[0] ))

    # top rect
    actors.append( make_2_vert_tube( v[4], v[5] ))
    actors.append( make_2_vert_tube( v[5], v[6] ))
    actors.append( make_2_vert_tube( v[6], v[7] ))
    actors.append( make_2_vert_tube( v[7], v[4] ))

    # verticals
    actors.append( make_2_vert_tube( v[0], v[4] ))
    actors.append( make_2_vert_tube( v[1], v[5] ))
    actors.append( make_2_vert_tube( v[2], v[6] ))
    actors.append( make_2_vert_tube( v[3], v[7] ))

    return actors

def get_mayavi_cubic_arena_source(engine,info=None):
    from enthought.mayavi.sources.api import VTKDataSource
    from enthought.mayavi.modules.surface import Surface

    v=info['verts4x4'] # arranged in 2 rectangles of 4 verts

    points = v
    lines = [ [0,1], [1,2], [2,3], [3,0],
              [4,5], [5,6], [6,7], [7,4],
              [0,4], [1,5], [2,6], [3,7] ]
    #polys = numpy.arange(0, len(points), 1, 'l')
    #polys = numpy.reshape(polys, (len(points), 1))
    pd = tvtk.PolyData(points=points, #polys=polys,
                       lines=lines)

    e=engine
    e.add_source( VTKDataSource(data=pd,name='cubic arena') )

    if 0:
        filter = tvtk.TubeFilter(number_of_sides=6)
        filter.radius = radius
        f = FilterBase(filter=filter, name='TubeFilter')
        mayavi.add_filter(f)
    if 1:
        s = Surface()
        e.add_module(s)
    if 0:
        v = Vectors()
        v.glyph.scale_mode = 'data_scaling_off'
        #v.glyph.color_mode = 'color_by_scalar'
        #v.glyph.color = 'black'
        v.glyph.glyph_source = tvtk.SphereSource(radius=options.radius*5.0)
        #v.glyph.glyph_source = tvtk.GlyphSource2D()
        e.add_module(v)
