#!/usr/bin/env python
from Numeric import *
from LinearAlgebra import *
##import numarray as na
##from numarray import *
##from numarray.linear_algebra import *
import os
from OpenGLContext.quaternion import Quaternion

class Halfspaces:
    def __init__(self, coeff_offs, dim=None, n=None):
        """

        coeff_offs has n rows, where n is the number of halfspaces and
        dim+1 columns, where dim is the dimensionality of the space.
        The first dim columns are the coefficients of the boundary
        plane and the final column is the offset.

        """
        self.co = asarray(coeff_offs)
        assert len(self.co.shape)==2
        if dim is not None:
            assert dim+1 == self.co.shape[1]
        if n is not None:
            assert n == self.co.shape[0]

    def copy(self):
        return Halfspaces( self.co.copy() )

    def get_dim(self):
        return self.co.shape[1]-1
    dim = property(get_dim, None, None, 'dimension of enclosing space')
    
    def get_n(self):
        return self.co.shape[0]
    n = property(get_n, None, None, 'number of halfspaces')

    def qhull(self):
        s = '\n'.join( [' '.join(map(str,co)) for co in self.co] )
        return '%d # halfspaces\n%d\n%s\n'%(self.dim+1,self.n,s)

    def __getitem__(self,index):
        return self.co[index,:]

    def append(self, co):
        assert len(co) == self.dim+1
        toappend = asarray(co)
        toappend.shape=(1,self.dim+1)
        self.co = concatenate( (self.co,toappend), axis=0 )
        
    def get_intersection_vertices(self, other, interior_point):
        assert isinstance(other,Halfspaces)
        assert isinstance(interior_point,Vertices)
        assert interior_point.n == 1

        all_hs = self.copy()
        for i in range(other.n):
            all_hs.append(other[i])
        
        cmd = 'qhalf Fp -'
        #print cmd
        #print interior_point.qhull()
        #print all_hs.qhull()
        #print
        stdin, stdout, stderr = os.popen3(cmd)
        stdin.write( interior_point.qhull() )
        stdin.write( all_hs.qhull() )
        stdin.close()

        lines = stdout.read().splitlines()
        #print 'lines',lines
        if not len(lines):
            return
        dim = int(lines[0])
        n = int(lines[1])
        v = [ map(float,line.split()) for line in lines[2:]]
        v=transpose(array(v))
        #print 'stderr:',stderr.read()
        return Vertices(v)

class Vertices(object):
    def __init__(self, verts, dim=None, n=None):
        """

        verts has dim rows, where dim is the dimensionality of the
        enclosing space, and n columns where n is the number of
        vertices.

        """
        self.v = asarray(verts)
        
        assert len(self.v.shape)==2
        if dim is not None:
            assert dim == self.v.shape[0]
        if n is not None:
            assert n == self.v.shape[1]

    def copy(self):
        return Vertices( self.v.copy() )

    def get_dim(self):
        return self.v.shape[0]
    dim = property(get_dim, None, None, 'dimension of enclosing space')
    
    def get_n(self):
        return self.v.shape[1]
    n = property(get_n, None, None, 'number of vertices')
    
    def qhull(self):
        s = '\n'.join( [' '.join(map(str,v)) for v in transpose(self.v)] )
        return '%d # vertices\n%d\n%s\n'%(self.dim, self.n, s)

    def __getitem__(self,index):
        return self.v[:,index]

    def append(self, vert):
        assert len(vert) == self.dim
        toappend = asarray(vert)
        toappend.shape=(self.dim,1)
        self.v = concatenate( (self.v,toappend), axis=1 )

    def get_halfspaces(self):
        cmd = 'qconvex n -'
        stdin, stdout, stderr = os.popen3(cmd)
        #print 'cmd:',cmd
        stdin.write( self.qhull() )
        stdin.close()

        lines = stdout.read().splitlines()
        if not len(lines):
            return
        #print lines
        dim = int(lines[0])
        n = int(lines[1])
        h = [ map(float,line.split()) for line in lines[2:]]
        #print 'stderr:',stderr.read()
        return Halfspaces(h)

    def intersect(self,other,interior_point):
        assert isinstance(other,Vertices)
        my_hs = self.get_halfspaces()
        other_hs = other.get_halfspaces()
        vi = my_hs.get_intersection_vertices( other_hs, interior_point )
        return vi

class Point(Vertices):
    def __init__(self, origin=None):
        v=asarray(origin)
        assert len(v.shape)==1
        v.shape=(v.shape[0],1)
        Vertices.__init__(self,v)

class Cube(Vertices):
    def __init__(self, scale = 0.5, origin=None):
        if origin is None:
            origin = array((0.0,0.0,0.0))
            origin.shape = 3,1
        else:
            origin = asarray(origin)
            if len(origin.shape) == 1:
                assert origin.shape[0] == 3
                origin.shape = 3,1
            else:
                assert origin.shape == 3,1
        v = []
        for x in -scale,scale:
            for y in -scale,scale:
                for z in -scale,scale:
                    v.append((x,y,z))
        v=transpose(array(v))
        v=v+origin
        Vertices.__init__(self,v)
        
class Arena:
    def __init__(self, l=500, w=500, h=300):
        # for now, model is rectangular solid open on top
        self.l=l
        self.w=w
        self.h=h
        
        lowv = Vertices( zeros( (3,0), type=Float64 ) )
        lowz=0.0
        highv = Vertices( zeros( (3,0), type=Float64 ) )
        highz=self.h
        
        for x in 0.0, self.l:
            for y in 0.0, self.w:
                for v,z in zip((lowv,highv), (lowz,highz)):
                    v.append(( x,y,z))

        allv = lowv.copy()
        for i in range(highv.n):
            allv.append(v[i])

        self.high_verts = highv # used for occlusion
        self.halfspaces = allv.get_halfspaces()
        
    def get_internal_halfspaces(self):
        return self.halfspaces

    def get_cam_halfspaces(self, cam_vert):
        v = self.high_verts.copy()
        v.append(cam_vert)
        tmp = v.get_halfspaces()
        # remove "lid"
        lid = array( (0,0,-1) )
        result = Halfspaces( zeros( (0,4), type=Float64 ) )
        for i in range(tmp.n):
            hs = tmp[i]
            if not allclose(hs[:3],lid):
                result.append(hs)
        return result

class System:
    def __init__(self, arena=None, K=None):
        if arena is None:
            arena = Arena()
        if K is None:
            K = array([[  5.38229998e+00,   2.55301390e-04,  -1.46283860e+00],
                       [  1.46367293e-18,  -5.38249777e+00,  -1.08254240e+00],
                       [ -8.89384595e-20,   4.76985428e-20,  -4.42411291e-03]])

        self.arena = arena
        self.K = K
        
        # setup initial frustum
        Ki = inverse(self.K)

        # in image plane coordinates
        minx, maxx = 0.0, 655.0
        miny, maxy = 0.0, 490.0
        minw, maxw = 1.0, 10.0
        
        self.start_frustum = Vertices( zeros( (3,0), type=Float64 ) )
        for xi in minx, maxx:
            for yi in miny, maxy:
                # xi = ui/wi, yi = vi/wi
                for wi in minw, maxw:
                    ui = xi*wi
                    vi = yi*wi

                    x = array([[ui],[vi],[wi]])
                    # get world coords, not rotated or translated yet
                    V = matrixmultiply(Ki,x)
                    self.start_frustum.append( (V.flat[0], V.flat[1], V.flat[2]) )

    def get_badness(self, parameters):
        n_cams = len(parameters)/7
        for i in range(n_cams):
            C_    = parameters[i*7:i*7+3]   # cam origin
            cam_q = parameters[i*7+3:i*7+7] # cam quaternion
            
            R=Quaternion(cam_q).matrix()[:3,:3]
            t = matrixmultiply( -R, C_ )

            Xcam=self.start_frustum.copy() # world coords, not rotated or translated yet
            X = matrixmultiply(inverse(R),Xcam-t) # frustum in world coords
