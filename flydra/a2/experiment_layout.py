import conditions_draw
import numpy
from numpy import array
from enthought.tvtk.api import tvtk

def get_posts(all_verts):
    actors = []
    for verts in all_verts:

        verts = numpy.asarray(verts)
        pd = tvtk.PolyData()

        np = len(verts) - 1
        lines = numpy.zeros((np, 2), numpy.int64)
        lines[:,0] = numpy.arange(0, np-0.5, 1, numpy.int64)
        lines[:,1] = numpy.arange(1, np+0.5, 1, numpy.int64)

        pd.points = verts
        pd.lines = lines

        pt = tvtk.TubeFilter(radius=0.006,input=pd,
                             number_of_sides=20,
                             vary_radius='vary_radius_off',
                             )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0,0,0
        a.property.specular = 0.3
        actors.append(a)
    return actors

def get_tvtk_actors_for_file(filename=None):
    actors = []
    if filename=='DATA20080525_194631.kalmanized.h5':
        import warnings
        warnings.warn('using mama20080501, even though setup/calibration has changed')
        instance = conditions_draw.mama20080501()
        actors.extend( instance.get_tvtk_actors() )
        actors.extend(get_posts([[array([ 0.15307339,  0.52554792,  0.05171393]),
                                  array([ 0.16880691,  0.53390287,  0.30010557])]]))
    return actors
