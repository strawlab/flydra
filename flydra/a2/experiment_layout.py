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

## def get_tvtk_actors_for_file(filename=None,force_stimulus=False):

##     # XXX TODO Implement some kind of reconstructor hash to ensure
##     # these values are matched to the reconstructor used to create
##     # them.

##     actors = []
##     if filename=='DATA20080525_194631.kalmanized.h5':
##         import warnings
##         warnings.warn('using mama20080501, even though setup/calibration has changed')
##         instance = conditions_draw.mama20080501()
##         actors.extend( instance.get_tvtk_actors() )
##         actors.extend(get_posts([[array([ 0.15307339,  0.52554792,  0.05171393]),
##                                   array([ 0.16880691,  0.53390287,  0.30010557])]]))
##     elif filename=='DATA20080528_201023.kalmanized.h5':
##         import warnings
##         warnings.warn('using mama20080501, even though setup/calibration has changed')
##         instance = conditions_draw.mama20080501()
##         actors.extend( instance.get_tvtk_actors() )
##         actors.extend(get_posts([[array([ 0.10265471,  0.47653907,  0.28677366]),
##                                   array([ 0.10198015,  0.4799127 ,  0.02159713])]]))
##     elif filename=='DATA20080528_204034.kalmanized.h5':
##         import warnings
##         warnings.warn('using mama20080501, even though setup/calibration has changed')
##         instance = conditions_draw.mama20080501()
##         actors.extend( instance.get_tvtk_actors() )
##         actors.extend(get_posts([[array([ 0.10087657,  0.4751555 ,  0.28746428]),
##                                   array([ 0.09823772,  0.47664041,  0.0220485 ])]]))

##     else:
##         if force_stimulus:
##             raise KeyError('no stimulus defined for file filename')
##     return actors
