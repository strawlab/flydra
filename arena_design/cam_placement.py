#!/usr/bin/env python
import tempfile, sets, os, sys
from Numeric import *
from LinearAlgebra import *
##import numarray as na
##from numarray import *
##from numarray.linear_algebra import *
import RandomArray
from OpenGLContext.quaternion import Quaternion
import gts

def combination2_w_index( alist ):
    """make all unique 2-combinations of elements in list, with index"""
    results = []
    index_dict = {}
    for i in range(len(alist)):
        for j in range(i+1,len(alist)):
            a = alist[i]
            b = alist[j]
            pair = a,b
            results.append( pair )
            index_dict.setdefault( a, [] ).append(pair)
            index_dict.setdefault( b, [] ).append(pair)
    for k in index_dict.keys():
        index_dict[k] = list(sets.Set(index_dict[k])) # clear redundant tuples
    return results, index_dict

class CamPair:
    def __init__(self,cam1,cam2):
        print 'intersecting',cam1,cam2
        print 'line',sys._getframe().f_lineno
        self.cam1 = cam1
        self.cam2 = cam2
        
        qh = self.cam1.get_visible_space_qhull()
        if qh is not None:
            qh.cycle()
            c1v = qh.to_gts()
        else:
            c1v = None

        qh = self.cam2.get_visible_space_qhull()
        if qh is not None:
            qh.cycle()
            c2v = qh.to_gts()
        else:
            c2v = None

        if 0:
            print 'line',sys._getframe().f_lineno
            c1v = self.cam1.get_visible_space_gts()
            c2v = self.cam2.get_visible_space_gts()

            print 'line',sys._getframe().f_lineno
            qh = self.cam1.get_visible_space_qhull()
            print 'line',sys._getframe().f_lineno
            qh.save('c1v.qhull')
            qh.save('c1v.oogl')
            qh.save('c1v.gts')
            print 'line',sys._getframe().f_lineno
            qh.cycle()
            gts = qh.to_gts()
            print 'line',sys._getframe().f_lineno
            gts.save('cam1_qh.gts')

            qh = self.cam2.get_visible_space_qhull()
            qh.cycle()
            gts = qh.to_gts()
            gts.save('cam2_qh.gts')

            print 'line',sys._getframe().f_lineno
##        c1v.save('cam1.stl')
##        c2v.save('cam2.stl')
##        c1v.save('cam1.gts')
##        c2v.save('cam2.gts')
        print 'line',sys._getframe().f_lineno
        if (c1v is None) or (c2v is None):
            self.gts = None
        else:
            try:
                self.gts = c1v & c2v
            except gts.GTSIntersectionNotClosedError:
                self.gts = None
        print 'line',sys._getframe().f_lineno

        if self.gts is not None:
            self.gts.save(str(self)+'.stl')
            self.gts.save(str(self)+'.gts')
        else:
            fd=file(str(self)+'.stl',mode='w') # clear file
            fd=file(str(self)+'.gts',mode='w') # clear file
        
    def __repr__(self):
        return repr(self.cam1)+'I'+repr(self.cam2)

    def __or__(self,other):
        if self.get_gts() is None:
            return other.get_gts()
        if other.get_gts() is None:
            return self.get_gts()
        try:
            result = self.get_gts() | other.get_gts()
        except gts.GTSIntersectionNotClosedError:
            result = None
        return result

    def get_gts(self):
        return self.gts

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
        stdin, stdout, stderr = os.popen3(cmd)
        stdin.write( interior_point.qhull() )
        stdin.write( all_hs.qhull() )
        stdin.close()

        lines = stdout.read().splitlines()
        if not len(lines):
            return
        dim = int(lines[0])
        n = int(lines[1])
        vv = [ map(float,line.split()) for line in lines[2:]]
        final_v=[]
        def distance(v,fv):
            # no sqrt needed -- this isn't used for much
            return (v[0]-fv[0])**2 + (v[1]-fv[1])**2 + (v[2]-fv[2])**2
        epsilon = 1e-6
        for v in vv:
            tooclose=False
            for fv in final_v:
                if distance(v,fv) < epsilon:
                    tooclose=True
                    break
            if not tooclose:
                final_v.append(v)
        v=transpose(array(final_v))
        return Vertices(v)

class Vertices(object):
    def __init__(self, verts, dim=None, n=None):
        """

        verts has dim rows, where dim is the dimensionality of the
        enclosing space, and n columns where n is the number of
        vertices.

        """
        if len(verts) == 0:
            self.v = zeros((3,0))
        else:
            self.v = asarray(verts)

        assert len(self.v.shape)==2
        if dim is not None:
            assert dim == self.v.shape[0]
        if n is not None:
            assert n == self.v.shape[1]

    def as_array(self):
        return self.v

    def copy(self):
        return Vertices( self.v.copy() )

    def get_dim(self):
        return self.v.shape[0]
    dim = property(get_dim, None, None, 'dimension of enclosing space')
    
    def get_n(self):
        return self.v.shape[1]
    n = property(get_n, None, None, 'number of vertices')
    
    def qhull(self):
        def f2s(f):
            return '%f'%f
        s = '\n'.join( [' '.join(map(f2s,v)) for v in transpose(self.v)] )
        return '%d # vertices\n%d\n%s\n'%(self.dim, self.n, s)

    def __getitem__(self,index):
        return self.v[:,index]

    def append(self, vert, check=False):
        assert len(vert) == self.dim
        toappend = asarray(vert)
        toappend.shape=(self.dim,1)
        self.v = concatenate( (self.v,toappend), axis=1 )

    def get_halfspaces_old(self):
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

    def get_halfspaces(self):
        stdout, stderr = self._raw_qhull('qconvex n')

        lines = stdout.splitlines()
        if not len(lines):
            return
        #print lines
        dim = int(lines[0])
        n = int(lines[1])
        h = [ map(float,line.split()) for line in lines[2:]]
        #print 'stderr:',stderr.read()
        return Halfspaces(h)

    def save(self,save_file,fmt=None):
        if type(save_file) == file:
            fd = save_file
        else:
            fd=open(save_file,'w')
            if fmt is None:
                if save_file[-6:]=='.qhull':
                    fmt='qhull'
                elif save_file[-5:]=='.oogl':
                    fmt='oogl'
                elif save_file[-4:]=='.gts':
                    fmt='gts'
        if fmt == 'qhull': 
            stdout,stderr = self._raw_qhull('qconvex p')
            fd.write(stdout)
        elif fmt == 'oogl': # oogl is for geomview
            stdout,stderr = self._raw_qhull('qconvex G')
            fd.write(stdout)
        elif fmt == 'gts':
            # get triangulated points and face indices
            stdout,stderr = self._raw_qhull('qconvex Qt p i')
            lines = stdout.splitlines()
            
            line_no = 0
            dim = int(lines[line_no])
            if dim != 3:
                raise NotImplementedError('can only handle 3D')

            line_no += 1
            n_verts = int(lines[line_no])
            
            verts = []
            for i in range(n_verts):
                line_no += 1
                vert = map(float,lines[line_no].split())
                verts.append(vert)
                
            line_no += 1
            n_faces = int(lines[line_no])
            
            faces = []
            for i in range(n_faces):
                line_no += 1

                face = map(int,lines[line_no].split())
                faces.append(face)

            # parsed data, now make .gts format

            vert_faces = faces
            all_edges_idx = {} # keeps track of edge index via vert indices
            edges = [] # vert indices
            faces = []
            for vert_face in vert_faces:
                this_edge_face = []
                # get vertex-indexed edges (from face data)
                vert_edges = [] 
                for i in range(len(vert_face)-1):
                    vert_edges.append( (vert_face[i], vert_face[i+1]) )
                vert_edges.append( (vert_face[len(vert_face)-1], vert_face[0]) )

                # generate edge-indexed faces
                #   find all unique edges
                for vert1, vert2 in vert_edges:
                    found = False
                    if vert1 in all_edges_idx:
                        if vert2 in all_edges_idx[vert1]:
                            idx = all_edges_idx[vert1][vert2]
                            found = True
                            
                    if not found:
                        if vert2 in all_edges_idx:
                            if vert1 in all_edges_idx[vert2]:
                                idx = all_edges_idx[vert2][vert1]
                                found = True
                                
                    if not found:
                        idx = len(edges)
                        edges.append( (vert1, vert2) )
                        all_edges_idx.setdefault(vert1,{})[vert2]=idx
                        
                    this_edge_face.append(idx)
                faces.append(this_edge_face)
                
            # now save data
            fd.write('%d %d %d\n'%(len(verts),len(edges),len(faces)))
            for vert in verts:
                fd.write( ' '.join(map(str,vert)) )
                fd.write( '\n' )
            for edge in edges:
                edge = asarray(edge)+1 # deal with 1-based indexing
                fd.write( ' '.join(map(str,edge)) )
                fd.write( '\n' )
            for face in faces:
                face = asarray(face)+1 # deal with 1-based indexing
                face = face[::-1] # flip normals
                fd.write( ' '.join(map(str,face)) )
                fd.write( '\n' )
        else:
            raise ValueError('unknown save format "%s"'%fmt)

    def cycle(self):
        # cycle through qhull XXX this is a big hack... why needed?
        fd=tempfile.TemporaryFile()
        self.save(fd,fmt='qhull')
        fd.seek(0)
        self.v=read_qhull_vertices(fd).v
            
    def _raw_qhull(self,cmd):
        cmd = cmd + ' -'
        stdin, stdout, stderr = os.popen3(cmd)
        stdin.write( self.qhull() )
        stdin.close()

        stdout = stdout.read()
        stderr = stderr.read()
        return stdout, stderr

    def intersect(self,other,interior_point):
        assert isinstance(other,Vertices)
        my_hs = self.get_halfspaces()
        other_hs = other.get_halfspaces()
        vi = my_hs.get_intersection_vertices( other_hs, interior_point )
        return vi

    def to_gts(self):
        fd=tempfile.TemporaryFile()
        self.save(fd,fmt='gts')
        fd.seek(0)
        return gts.surface_read(fd)

def gts2qhull(gtso):
    qh_verts = Vertices([])
    for v in gtso.get_vertices():
        qh_verts.append(v)
    return qh_verts
        
def read_qhull_vertices(readfile):
    if type(readfile) == file:
        fd = readfile
    else:
        fd = file(readfile,mode='b')
    lines = fd.read().splitlines()
    dim = int(lines[0])
    n = int(lines[1])
    line_no = 1
    result = Vertices([])
    for i in range(n):
        line_no += 1
        vert = map(float,lines[line_no].split())
        result.append(vert)
    return result

class Point(Vertices):
    def __init__(self, origin=None):
        v=array(origin)
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
        
        lowv = Vertices( zeros( (3,0), Float64 ) )
        lowz=0.0
        highv = Vertices( zeros( (3,0), Float64 ) )
        highz=self.h
        
        for x in 0.0, self.l:
            for y in 0.0, self.w:
                for v,z in zip((lowv,highv), (lowz,highz)):
                    v.append(( x,y,z))

        allv = lowv.copy()
        for i in range(highv.n):
            allv.append(v[i])

        self.high_verts = highv # used for occlusion
        self.verts = allv
        self.halfspaces = allv.get_halfspaces()

    def get_verts(self):
        return self.verts

    def get_gts_surface(self):
        return self.verts.to_gts()
        
    def get_internal_halfspaces(self):
        return self.halfspaces

    def get_cam_halfspaces(self, cam_center):
        v = self.high_verts.copy()
        v.append(cam_center)
        v.save('hat.gts')
        tmp = v.get_halfspaces()
        # remove "lid"
        lid = array( (0,0,-1) )
        result = Halfspaces( zeros( (0,4), Float64 ) )
        for i in range(tmp.n):
            hs = tmp[i]
            if not allclose(hs[:3],lid):
                result.append(hs)
        return result

    def get_visible_point(self):
        return Vertices( array([[self.l/2.0],[self.w/2.0],[self.h/2.0]]) )

class Cam:
    def __init__(self,name,center,frustum_verts):
        self.name = name
        self.center = center
        self.frustum_verts = frustum_verts

    def get_frustum_verts(self):
        return Vertices(self.frustum_verts)
    
    def set_visible_space_gts(self,vv):
        self.vv = vv
        if self.vv is None:
            self.qh_verts = None
        else:
            self.qh_verts = gts2qhull(self.vv)

    def get_visible_space_gts(self):
        return self.vv

    def get_visible_space_qhull(self):
        return self.qh_verts

    def __repr__(self):
        return "Cam"+self.name

class System:
    def __init__(self, arena=None, K=None):
        if arena is None:
            arena = Arena()
        self.arena = arena
        
        if K is None:
            K = array([[  5.38229998e+00,   2.55301390e-04,  -1.46283860e+00],
                       [  1.46367293e-18,  -5.38249777e+00,  -1.08254240e+00],
                       [ -8.89384595e-20,   4.76985428e-20,  -4.42411291e-03]])

        self.K = K
        
        # setup initial frustum
        Ki = inverse(self.K)

        # in image plane coordinates
        minx, maxx = 0.0, 655.0
        miny, maxy = 0.0, 490.0
        minw, maxw = 1.0, 2
        
        self.start_frustum = Vertices( zeros( (3,0), Float64 ) )
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
        cams = []
        for i in range(n_cams):
            print 'cam',i
            C_    = parameters[i*7:i*7+3]   # cam center
            cam_q = parameters[i*7+3:i*7+7] # cam quaternion
            
            R=Quaternion(cam_q).matrix()[:3,:3]
            t = matrixmultiply( -R, C_ )
            t.shape=3,1 # column vector

            Xcam=self.start_frustum.copy() # world coords, not rotated or translated yet
            X = matrixmultiply(inverse(R),Xcam.as_array()-t) # frustum in world coords

            cams.append( Cam(str(i+1),C_,X) )

            # 0. get camera frustum
            fname = 'frustum_cam%02d.gts'%i
            frustum = cams[i].get_frustum_verts().to_gts()
            print ' frustum.volume',frustum.volume
            # 1. get total area of interest: determine non-occluded arena per cam
            DEBUG=1
            if DEBUG:
                self.arena.get_gts_surface().save('arena.stl')
                frustum.save('frustum%d.stl'%(i+1,))
                cams[i].get_frustum_verts().save('frustum%d.qhull'%(i+1,))
            
            cam_to_arena_hs = self.arena.get_cam_halfspaces( cams[i].center )
            non_occluded_verts = cam_to_arena_hs.get_intersection_vertices(
                self.arena.get_internal_halfspaces(),
                self.arena.get_visible_point())
            if non_occluded_verts is None:
                if DEBUG: print 'all occluded'
                continue # no part of the arena is visible
            
##            if DEBUG: 
##                non_occluded_verts.save('non_occluded_DEBUG')
            non_occluded = non_occluded_verts.to_gts()
            print ' non_occluded.volume',non_occluded.volume
            
            # 2.   intersect with camera's frustum
            # 2.a. easy way -- is non_occluded completely within frustum?
            one_not_in = False
            for pt in non_occluded.get_vertices():
                one_not_in = not frustum.contains(pt)
                if one_not_in:
                    break
            if one_not_in:
                # 2.b. perform intersection using GTS
                try:
                    visible_area = frustum & non_occluded
                except gts.GTSIntersectionNotClosedError,x:
                    visible_area = None
            else:
                visible_area = non_occluded
                
            cams[i].set_visible_space_gts(visible_area)
            
        # generate 2-tuples of camera combinations of overlapping FOVs: cam_pairs
        print 'line',sys._getframe().f_lineno
        cam_pairs, cam_pair_idx = combination2_w_index(cams)
        print 'line',sys._getframe().f_lineno
        cam_pairs = [ CamPair(*cam_pair) for cam_pair in cam_pairs ]
        print 'line',sys._getframe().f_lineno
        print cam_pairs
        print 'line',sys._getframe().f_lineno

        unions = []
        for cam_pair in cam_pairs:
            print cam_pair
            cam_pair_gts = cam_pair.get_gts()
            if cam_pair_gts is None:
                continue
            print 'line',sys._getframe().f_lineno
            found_union = False
            for i,joined_cam_pairs_gts in enumerate(unions):
                print 'line',sys._getframe().f_lineno
                
                merge1 = gts2qhull(cam_pair_gts)
                merge1.cycle()
                merge1.cycle()
                merge1=merge1.to_gts()
                merge1.save('merge1.gts')
                merge1.save('merge1.stl')
                
                merge2 = gts2qhull(joined_cam_pairs_gts)
                merge2.cycle()
                merge2.cycle()
                merge2=merge2.to_gts()
                merge2.save('merge2.gts')
                merge2.save('merge2.stl')
                
                print 'IN'
                merged_gts = cam_pair_gts | joined_cam_pairs_gts
                print 'OUT'
                if merged_gts is not None:
                    print 'line',sys._getframe().f_lineno
                    merged_gts.save('merged.stl')
                    print 'line',sys._getframe().f_lineno
                    unions[i] = merged_gts
                    found_union = True
                    break
            if not found_union:
                unions.append( cam_pair_gts )
        # total volume
        total_volume = 0
        for joined_cam_pairs in unions:
            total_volume += joined_cam_pairs.volume
            
        return -total_volume

if __name__=='__main__':
    s=System()
    n_cams = 4
    params = []
    for i in range(n_cams):
        cam_params = zeros((7,),'d')
        rnd_pos = RandomArray.normal(0.0, 200.0,(3,))
        start_pos = array([250.0,250,2000])
        cam_params[:3] = start_pos+rnd_pos
        cam_params[3:] = 1,0,0,1
        params.extend(cam_params)
    s.get_badness(params)
