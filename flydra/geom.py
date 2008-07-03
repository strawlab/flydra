import math
import numpy
import scipy.optimize

__all__=['ThreeTuple',
         'PlueckerLine',
         'line_from_points',
         'LineSegment',
         'Plane']

# see http://web.mit.edu/thouis/pluecker.txt

def cross(vec1,vec2):
    return ( vec1[1]*vec2[2] - vec1[2]*vec2[1],
             vec1[2]*vec2[0] - vec1[0]*vec2[2],
             vec1[0]*vec2[1] - vec1[1]*vec2[0] )

class ThreeTuple:
    def __init__(self,vals):
        if isinstance(vals,ThreeTuple):
            self.vals = numpy.array(vals.vals,copy=True)
            return
        self.vals = numpy.asarray(vals)
        if self.vals.shape != (3,):
            raise ValueError('shape must be (3,)')
    def __eq__(self,other):
        return (self.vals[0]==other.vals[0] and
                self.vals[1]==other.vals[1] and
                self.vals[2]==other.vals[2])
    def __repr__(self):
        return 'ThreeTuple((%s,%s,%s))'%tuple(map(repr,self.vals))
    def __sub__(self,other):
        return ThreeTuple(self.vals-other.vals)
    def __add__(self,other):
        return ThreeTuple(self.vals+other.vals)
    def __mul__(self,other):
        return ThreeTuple(self.vals*other)
    def __rmul__(self,other):
        return ThreeTuple(self.vals*other)
    def __neg__(self):
        return ThreeTuple(-self.vals)
    def cross(self,other):
        return ThreeTuple(cross(self.vals,other.vals))
    def dot(self,other):
        return numpy.dot(self.vals,other.vals)
    def __getitem__(self,i):
        return self.vals[i]
    def dist_from(self,other):
        return math.sqrt(numpy.sum((other.vals-self.vals)**2)) # L2 norm

class Homogeneous3D:
    def __init__(self,xyz,w):
        self.vals = numpy.array([xyz[0], xyz[1], xyz[2], w])
    def to_3tup(self):
        return ThreeTuple( self.vals[:3]/self.vals[3] )

class PlueckerLine:
    def __init__(self, u, v):
        if not isinstance(u,ThreeTuple):
            raise TypeError('u must be ThreeTuple')
        if not isinstance(v,ThreeTuple):
            raise TypeError('v must be ThreeTuple')
        self.u = u
        self.v = v
    def __eq__(self,other):
        return ((self.u == other.u) and (self.v == other.v))
    def to_hz(self):
        return (self.v[2], -self.v[1], self.u[0], self.v[0], -self.u[1], self.u[2])
    def __repr__(self):
        return 'PlueckerLine(%s,%s)'%(repr(self.u),repr(self.v))
    def get_my_point_closest_to_line(self,other):
        """find point on line closest to other line"""

        class ErrFMaker:
            def __init__(self, line, other):
                self.other = other
                self.direction = line.u
                self.pt0 = line.closest()
            def get_point_by_mu( self, mu ):
                return self.pt0 + mu*self.direction
            def errf( self, mu_vec ):
                mu = mu_vec[0]
                pt = self.get_point_by_mu(mu)
                rel_line = self.other.translate( -pt )
                return rel_line.dist2()

        # XXX TODO. The implementation could be improved (i.e. sped up).
        import warnings
        warnings.warn('slow/lazy way to find closest point to line')
        initial_mu = 0.0
        efm = ErrFMaker(self,other)
        final_mu, = scipy.optimize.fmin( efm.errf, [initial_mu], disp=0 )
        pt = efm.get_point_by_mu( final_mu )
        return pt

    def dist2(self):
        """return minimum squared distance from origin"""
        return self.v.dot(self.v) / self.u.dot(self.u)
    def closest(self):
        """return point on line closest to origin"""
        VxU = self.v.cross(self.u)
        UdotU = self.u.dot(self.u)
        h = Homogeneous3D(VxU,UdotU)
        return h.to_3tup()
    def direction(self):
        return self.u
    def intersect(self,plane):
        if not isinstance(plane,Plane):
            raise NotImplementedError('only Plane intersections implemented')
        N = plane.N
        n = plane.n

        VxN = self.v.cross(N)
        Un = self.u*n

        U_N = self.u.dot(N)
        pt = (VxN-Un)*(1.0/U_N)
        return pt
    def translate(self,threetuple):
        if not isinstance(threetuple,ThreeTuple):
            raise ValueError('expected ThreeTuple instance, got %s'%repr(threetuple))
        on_line = self.closest()
        on_new_line_a = on_line+threetuple
        on_new_line_b = on_new_line_a + self.u
        return line_from_points(on_new_line_a,on_new_line_b)

def line_from_points(p,q):
    """create PlueckerLine instance given 2 distinct points

    example2:

    >>> p1 = ThreeTuple((2,3,7))
    >>> p2 = ThreeTuple((2,1,0))
    >>> L = line_from_points(p1,p2)
    >>> print L
    PlueckerLine(ThreeTuple(0,2,7),ThreeTuple(-7,14,-4))

    >>> q1 = ThreeTuple((0,2,7))
    >>> q2 = ThreeTuple((0,2,0))
    >>> L2 = line_from_points(q1,q2)
    >>> print L2.dist2()
    4
    >>> print L2.closest()
    ThreeTuple(0,2,0)
    """

    if not isinstance(p,ThreeTuple):
        raise ValueError('must be ThreeTuple')
    if not isinstance(q,ThreeTuple):
        raise ValueError('must be ThreeTuple')
    u = p-q # line direction
    v = p.cross(q)
    return PlueckerLine(u,v)

def line_from_HZline(P):
    """line from Hartley & Zisserman Pluecker coordinates"""
    u = ThreeTuple( (P[2], -P[4], P[5] ) )
    v = ThreeTuple( (P[3], -P[1], P[0] ) )
    return PlueckerLine(u,v)

class LineSegment:
    """part of a line between 2 endpoints

    >>> seg = LineSegment(ThreeTuple((0,0,0)),ThreeTuple((0,0,10)))

    >>> point = ThreeTuple((1,0,5))
    >>> print seg.get_distance_from_point(point)
    1.0

    >>> point = ThreeTuple((0,0,-1))
    >>> print seg.get_distance_from_point(point)
    1.0

    >>> point = ThreeTuple((2,0,0))
    >>> print seg.get_distance_from_point(point)
    2.0

    """

    def __init__(self,p,q):
        """create LineSegment instance given endpoints"""
        self.p = p
        self.q = q
        self.length = p.dist_from(q)
    def __repr__(self):
        return 'LineSegment(%s,%s)'%(repr(self.p),repr(self.q))
    def get_closest_point(self,r):
        if not isinstance(r,ThreeTuple):
            raise ValueError('r must be ThreeTuple')

        # create line such that r is at origin
        ps = self.p-r
        qs = self.q-r
        L = line_from_points(ps,qs)
        closest = L.closest() # find point on line closest to origin
        pc = ps.dist_from(closest)
        qc = qs.dist_from(closest)

        # There are 2 cases.
        # If closest point is between endpoints:
        if pc < self.length and qc < self.length:
            return closest+r

        # closest point is closer to one endpoint
        if pc < qc:
            # closest to self.p
            return self.p
        else:
            # closest to self.q
            return self.q

    def get_distance_from_point(self,r):
        return self.get_closest_point(r).dist_from(r)

class Plane:
    def __init__(self, normal_vec, dist_from_origin):
        if not isinstance(normal_vec,ThreeTuple):
            raise ValueError('must be ThreeTuple')
        self.N = normal_vec
        self.n = float(dist_from_origin)

def _test():
    import doctest
    doctest.testmod()

if __name__=='__main__':
    _test()
