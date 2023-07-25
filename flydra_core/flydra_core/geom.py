from __future__ import print_function
import math
import numpy
import numpy as np
import scipy.optimize
import warnings

__all__ = ["ThreeTuple", "PlueckerLine", "line_from_points", "LineSegment", "Plane"]

# see http://web.mit.edu/thouis/pluecker.txt


def cross(vec1, vec2):
    return (
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    )


class ThreeTuple:
    """A tuple of 3 points.

    Parameters
    ----------
    vals : {sequence, ThreeTuple instance}
      The tuple of 3 points
    """

    def __init__(self, vals):
        if isinstance(vals, ThreeTuple):
            self.vals = numpy.array(vals.vals, copy=True)
            return
        self.vals = numpy.asarray(vals)
        if self.vals.shape != (3,):
            raise ValueError("shape must be (3,)")

    def __eq__(self, other):
        return (
            self.vals[0] == other.vals[0]
            and self.vals[1] == other.vals[1]
            and self.vals[2] == other.vals[2]
        )

    def __repr__(self):
        return "ThreeTuple((%s,%s,%s))" % tuple(map(repr, self.vals))

    def __sub__(self, other):
        return ThreeTuple(self.vals - other.vals)

    def __add__(self, other):
        return ThreeTuple(self.vals + other.vals)

    def __mul__(self, other):
        return ThreeTuple(self.vals * other)

    def __rmul__(self, other):
        return ThreeTuple(self.vals * other)

    def __neg__(self):
        return ThreeTuple(-self.vals)

    def __abs__(self):
        return np.sqrt(np.sum(self.vals ** 2))

    def cross(self, other):
        """cross product

        Parameters
        ----------
        other : ThreeTuple instance
          The other point to do the cross product with.

        Returns
        -------
        result : ThreeTuple instance
          The cross product result

        Examples
        --------
        >>> x = ThreeTuple((1,0,0))
        >>> y = ThreeTuple((0,1,0))
        >>> x.cross(y)
        ThreeTuple((0,0,1))
        """
        return ThreeTuple(cross(self.vals, other.vals))

    def dot(self, other):
        """dot product

        Parameters
        ----------
        other : ThreeTuple instance
          The other point to do the dot product with.

        Returns
        -------
        result : scalar
          The dot product result

        Examples
        --------
        >>> x = ThreeTuple((1,0,0))
        >>> y = ThreeTuple((0,1,0))
        >>> x.dot(y)
        0
        """
        return numpy.dot(self.vals, other.vals)

    def __getitem__(self, i):
        return self.vals[i]

    def dist_from(self, other):
        """get distance from other point

        Parameters
        ----------
        other : ThreeTuple instance
          The other point to find the distance from

        Returns
        -------
        result : scalar
          The distance to the other point

        Examples
        --------
        >>> x = ThreeTuple((1,0,0))
        >>> y = ThreeTuple((0,1,0))
        >>> x.dist_from(y) == np.sqrt(2)
        True

        """
        return math.sqrt(numpy.sum((other.vals - self.vals) ** 2))  # L2 norm


class Homogeneous3D:
    def __init__(self, xyz, w):
        self.vals = numpy.array([xyz[0], xyz[1], xyz[2], w])

    def to_3tup(self):
        return ThreeTuple(self.vals[:3] / self.vals[3])


class PlueckerLine:
    """a line in 3D space

    Parameters
    ----------
    u : ThreeTuple instance
       direction of line
    v : ThreeTuple instance
       cross product of 2 points on line

    """

    def __init__(self, u, v):
        if not isinstance(u, ThreeTuple):
            raise TypeError("u must be ThreeTuple")
        if not isinstance(v, ThreeTuple):
            raise TypeError("v must be ThreeTuple")
        self.u = u
        self.v = v

    def __eq__(self, other):
        return (self.u == other.u) and (self.v == other.v)

    def to_hz(self):
        return (self.v[2], -self.v[1], self.u[0], self.v[0], -self.u[1], self.u[2])

    def __repr__(self):
        return "PlueckerLine(%s,%s)" % (repr(self.u), repr(self.v))

    def get_my_point_closest_to_line(self, other):
        """find point on line closest to other line

        Parameters
        ----------
        other : PlueckerLine instance
          The line to find closest point relative to

        Returns
        -------
        pt : ThreeTuple instance
          The point closest to other line

        Examples
        --------

        >>> # A line along +y going through (1,0,0)
        >>> a = ThreeTuple((1,0,0))
        >>> b = ThreeTuple((1,1,0))
        >>> line = line_from_points(a,b)

        >>> # A line along +z going through (0,0,0)
        >>> O = ThreeTuple((0,0,0))
        >>> z = ThreeTuple((0,0,1))
        >>> zaxis = line_from_points(z,O)

        >>> # The closest point between them:
        >>> line.get_my_point_closest_to_line( zaxis )
        ThreeTuple((1.0,0.0,0.0))

        """

        class ErrFMaker:
            def __init__(self, line, other):
                self.other = other
                self.direction = line.u
                self.pt0 = line.closest()

            def get_point_by_mu(self, mu):
                return self.pt0 + mu * self.direction

            def errf(self, mu_vec):
                mu = mu_vec[0]
                pt = self.get_point_by_mu(mu)
                rel_line = self.other.translate(-pt)
                return rel_line.dist2()

        # XXX TODO. The implementation could be improved (i.e. sped up).
        # should do something like is done for mahalanobis case.
        warnings.warn("slow/lazy way to find closest point to line")
        initial_mu = 0.0
        efm = ErrFMaker(self, other)
        (final_mu,) = scipy.optimize.fmin(efm.errf, [initial_mu], disp=0)
        pt = efm.get_point_by_mu(final_mu)
        return pt

    def dist2(self):
        """return minimum squared distance from origin"""
        return self.v.dot(self.v) / self.u.dot(self.u)

    def closest(self):
        """return point on line closest to origin
        Examples
        --------
        >>> a = ThreeTuple((1.0, 0.0, 0.0))
        >>> b = ThreeTuple((1.0, 1.0, 0.0))
        >>> line = line_from_points(a,b)
        >>> line.closest()
        ThreeTuple((1.0,0.0,-0.0))
        """
        VxU = self.v.cross(self.u)
        UdotU = self.u.dot(self.u)
        h = Homogeneous3D(VxU, UdotU)
        return h.to_3tup()

    def direction(self):
        return self.u

    def intersect(self, plane):
        if not isinstance(plane, Plane):
            raise NotImplementedError("only Plane intersections implemented")
        N = plane.N
        n = plane.n

        VxN = self.v.cross(N)
        Un = self.u * n

        U_N = self.u.dot(N)
        pt = (VxN - Un) * (1.0 / U_N)
        return pt

    def translate(self, threetuple):
        if not isinstance(threetuple, ThreeTuple):
            raise ValueError("expected ThreeTuple instance, got %s" % repr(threetuple))
        on_line = self.closest()
        on_new_line_a = on_line + threetuple
        on_new_line_b = on_new_line_a + self.u
        return line_from_points(on_new_line_a, on_new_line_b)


def line_from_points(p, q):
    """create PlueckerLine instance given 2 distinct points

    example2:

    >>> p1 = ThreeTuple((2.0, 3.0, 7.0))
    >>> p2 = ThreeTuple((2.0, 1.0, 0.0))
    >>> L = line_from_points(p1,p2)
    >>> print(L)
    PlueckerLine(ThreeTuple((0.0,2.0,7.0)),ThreeTuple((-7.0,14.0,-4.0)))

    >>> q1 = ThreeTuple((0.0, 2.0, 7.0))
    >>> q2 = ThreeTuple((0.0, 2.0, 0.0))
    >>> L2 = line_from_points(q1,q2)
    >>> print(L2.dist2())
    4.0
    >>> print(L2.closest())
    ThreeTuple((0.0,2.0,-0.0))
    """

    if not isinstance(p, ThreeTuple):
        raise ValueError("must be ThreeTuple")
    if not isinstance(q, ThreeTuple):
        raise ValueError("must be ThreeTuple")
    u = p - q  # line direction
    v = p.cross(q)
    return PlueckerLine(u, v)


def line_from_HZline(P):
    """line from Hartley & Zisserman Pluecker coordinates"""
    u = ThreeTuple((P[2], -P[4], P[5]))
    v = ThreeTuple((P[3], -P[1], P[0]))
    return PlueckerLine(u, v)


class LineSegment:
    """part of a line between 2 endpoints

    >>> seg = LineSegment(ThreeTuple((0,0,0)),ThreeTuple((0,0,10)))

    >>> point = ThreeTuple((1,0,5))
    >>> print(seg.get_distance_from_point(point))
    1.0

    >>> point = ThreeTuple((0,0,-1))
    >>> print(seg.get_distance_from_point(point))
    1.0

    >>> point = ThreeTuple((2,0,0))
    >>> print(seg.get_distance_from_point(point))
    2.0

    """

    def __init__(self, p, q):
        """create LineSegment instance given endpoints"""
        self.p = p
        self.q = q
        self.length = p.dist_from(q)

    def __repr__(self):
        return "LineSegment(%s,%s)" % (repr(self.p), repr(self.q))

    def get_closest_point(self, r):
        if not isinstance(r, ThreeTuple):
            raise ValueError("r must be ThreeTuple")

        # create line such that r is at origin
        ps = self.p - r
        qs = self.q - r
        L = line_from_points(ps, qs)
        closest = L.closest()  # find point on line closest to origin
        pc = ps.dist_from(closest)
        qc = qs.dist_from(closest)

        # There are 2 cases.
        # If closest point is between endpoints:
        if pc < self.length and qc < self.length:
            return closest + r

        # closest point is closer to one endpoint
        if pc < qc:
            # closest to self.p
            return self.p
        else:
            # closest to self.q
            return self.q

    def get_distance_from_point(self, r):
        return self.get_closest_point(r).dist_from(r)


class Plane:
    def __init__(self, normal_vec, dist_from_origin):
        if not isinstance(normal_vec, ThreeTuple):
            raise ValueError("must be ThreeTuple")
        self.N = normal_vec
        self.n = float(dist_from_origin)

        if self.n < 0:
            # make distance always positive
            self.n = -self.n
            self.N = -self.N

    def __repr__(self):
        return "Plane(%s,%s)" % (repr(self.N), repr(self.n))

    def is_close(self, other, eps=1e-15):
        assert isinstance(other, Plane)

        # compare distance from origin
        if abs(self.n - other.n) > eps:
            return False

        near_origin = False
        if abs(self.n) < eps:
            near_origin = True

        v1 = self.N.vals
        v2 = other.N.vals

        # normalize
        n1 = v1 / np.sqrt(np.sum(v1 ** 2))
        n2 = v2 / np.sqrt(np.sum(v2 ** 2))

        costheta = np.dot(n1, n2)
        if near_origin:
            costheta = abs(costheta)

        return abs(costheta - 1.0) < eps


class GeometryException(Exception):
    pass


class NotCoplanarError(GeometryException):
    pass


class ColinearError(GeometryException):
    pass


def points_to_plane(*args, **kwds):
    if len(args) < 3:
        raise ValueError("must input at least 3 points")

    eps = kwds.get("eps", 1e-16)

    X = []
    for A in args:

        assert isinstance(A, ThreeTuple)
        A = np.asarray(A.vals)

        # make homogeneous
        A = np.concatenate((A, [1]))
        X.append(A)

    # eqn 3.3 of Hartley Zisserman
    X = np.array(X)

    u, d, vt = np.linalg.svd(X)  # ,full_matrices=True)

    if np.any(d[:3] < eps):
        raise ColinearError("points not in general position")

    if not np.all(d[3:] < eps):
        raise NotCoplanarError("points not co-planar (errors=%s)" % (d[3:],))

    if 0:
        print("X")
        print(X)
        print("u", u.shape)
        print("u")
        print(u)
        print("d", d.shape)
        print("d", d)
        print("vt", vt.shape)
        print("vt")
        print(vt)
        print()

    n = vt[3, :3]
    mag = np.sqrt(np.sum(n ** 2))
    norm = n / mag
    dist = vt[3, 3] / mag
    p = Plane(ThreeTuple(norm), dist)
    return p


def test_plane():
    p = Plane(ThreeTuple((1, 0, 0)), 1)
    p2 = Plane(ThreeTuple((-1, 0, 0)), -1)
    assert p.is_close(p2)

    eps = 1e-16
    # ensure that distance is always positive
    assert abs(p2.n - 1) < eps


def test_points_to_plane():

    A = ThreeTuple((1, 0, 0))
    B = ThreeTuple((0, 1, 0))
    C = ThreeTuple((0, 0, 0))
    p = points_to_plane(A, B, C)

    assert Plane(ThreeTuple((0, 0, 1)), 0).is_close(p)

    A = ThreeTuple((1, 0, 1))
    B = ThreeTuple((0, 1, 1))
    C = ThreeTuple((0, 0, 1))
    p = points_to_plane(A, B, C)
    assert Plane(ThreeTuple((0, 0, 1)), -1).is_close(p)

    A = ThreeTuple((1, 0, -1))
    B = ThreeTuple((0, 1, -1))
    C = ThreeTuple((0, 0, -1))
    p = points_to_plane(A, B, C)
    assert Plane(ThreeTuple((0, 0, 1)), 1).is_close(p)

    A = ThreeTuple((1, -1, 0))
    B = ThreeTuple((0, -1, 1))
    C = ThreeTuple((0, -1, 0))
    p = points_to_plane(A, B, C)
    assert Plane(ThreeTuple((0, 1, 0)), 1).is_close(p)

    A = ThreeTuple((1, -2, 0))
    B = ThreeTuple((0, -2, 1))
    C = ThreeTuple((0, -2, 0))
    p = points_to_plane(A, B, C)
    assert Plane(ThreeTuple((0, 1, 0)), 2).is_close(p)

    # test ability to do 4 points
    A = ThreeTuple((1, 0, 0))
    B = ThreeTuple((0, 1, 0))
    C = ThreeTuple((0, 0, 0))
    D = ThreeTuple((1, 1, 0))
    p = points_to_plane(A, B, C, D)

    assert Plane(ThreeTuple((0, 0, 1)), 0).is_close(p)

    # test ability to detect 4 non-coplanar points
    A = ThreeTuple((1, 0, 0))
    B = ThreeTuple((0, 1, 0))
    C = ThreeTuple((0, 0, 0))
    D = ThreeTuple((1, 1, 1))
    try:
        p = points_to_plane(A, B, C, D)
    except NotCoplanarError:
        pass
    else:
        raise RuntimeError("failed to detect NotCoplanarError")

    # test ability to detect 3 co-linear points
    A = ThreeTuple((1, 0, 0))
    C = ThreeTuple((2, 0, 0))
    B = ThreeTuple((3, 0, 0))
    try:
        p = points_to_plane(A, B, C)
    except ColinearError:
        pass
    else:
        raise RuntimeError("failed to detect ColinearError")


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
