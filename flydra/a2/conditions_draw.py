from __future__ import division
import math
import numpy
from enthought.tvtk.api import tvtk
import conditions
import stimulus_positions

def mamarama(filename=None):
    actors = []

    if 1:
        # mamarama
        z = 0
        N = 64
        radius = 1.0
        center = numpy.array([ 1.0,0,0])
        
        verts = []
        vi = 0 # vert idx
        lines = []
        theta = numpy.linspace(0,2*math.pi,N,endpoint=False)
        X = radius*numpy.cos(theta) + center[0]
        Y = radius*numpy.sin(theta) + center[1]
        height = .762
        for z in numpy.linspace(0,height,4):
            Z = numpy.zeros(theta.shape) + center[2] + z
            v = numpy.array([X,Y,Z]).T
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
        actors.append(a)
    return actors

def draw_default_stim(filename=None):
    ### draw floor
    actors = []
    if 1:
        floorz = 0.0
        
        x0 = 0.007
        x1 = 1.007
        y0 = .065
        y1 = .365
        #z0 = -.028
        z0 = floorz-.06
        
        inc = 0.05
        if 1:
            nx = int(math.ceil((x1-x0)/inc))
            ny = int(math.ceil((y1-y0)/inc))
            eps = 1e-10
            x1 = x0+nx*inc+eps
            y1 = y0+ny*inc+eps
        
        segs = []
        for x in numpy.r_[x0:x1:inc]:
            seg =[(x,y0,z0),
                  (x,y1,z0)]
            segs.append(seg)
        for y in numpy.r_[y0:y1:inc]:
            seg =[(x0,y,z0),
                  (x1,y,z0)]
            segs.append(seg)
            
        if 1:
            verts = []
            for seg in segs:
                verts.extend(seg)
            verts = numpy.asarray(verts)

            pd = tvtk.PolyData()

            np = len(verts)/2
            lines = numpy.zeros((np, 2), numpy.int64)
            lines[:,0] = 2*numpy.arange(np,dtype=numpy.int64)
            lines[:,1] = lines[:,0]+1

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
            actors.append(a)

    ###################
    stim = None
    try:
        condition, stim = conditions.get_condition_stimname_from_filename(filename)
        print 'Data from condition "%s",with stimulus'%(condition,),stimname
    except KeyError, err:
        print 'Unknown condition and stimname'
        
    if stim is None:
        return actors
    
    all_verts = stimulus_positions.stim_positions[stim]

    for verts in all_verts:

        verts = numpy.asarray(verts)
        floorz = min(floorz, verts[:,2].min() )
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
