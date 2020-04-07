from __future__ import print_function
import numpy as nx

svd = nx.linalg.svd


class Point:
    def __init__(self, x0, x1, x2, x3=None):
        if x3 is None:
            x3 = 1.0
        self.data = nx.array([x0, x1, x2, x3], nx.Float)

    def __repr__(self):
        return "Point" + repr(tuple(self.data))

    def is_on_plane(self, plane, eps=1e-14):
        return abs(nx.dot(plane.data, self.data)) < eps


def PointFromIntersectingPlanes(*planes):
    return planes2point(*planes)


def PointFromIntersectingPlaneAndLine(plane, line):
    line = LinePlueckerMatrix(line)
    X = nx.matrixmultiply(line.data, nx.reshape(plane.data, (4, 1)))
    X.shape = (4,)
    return Point(*X)


class Plane:
    def __init__(self, p0, p1, p2, p3):
        self.data = nx.array([p0, p1, p2, p3], nx.Float)

    def __repr__(self):
        return "Plane" + repr(tuple(self.data))


class Line:
    # base class for representations of line
    def create_point(self):
        """return a Point on the line"""
        # see if line intersects plane x==1
        planex1 = Plane(1, 0, 0, -1)
        # xxx need to make sure plane is not parallel to line
        return PointFromIntersectingPlaneAndLine(planex1, self)


def LineFromIntersectingPlanes(*planes):
    return planes2line(*planes)


class LinePlueckerCoords(Line):
    def __init__(self, p0, p1, p2, p3, p4, p5):
        self.data = nx.array([p0, p1, p2, p3, p4, p5], nx.Float)

    def __repr__(self):
        return "LinePlueckerCoords" + repr(tuple(self.data))


class LinePlueckerMatrix(Line):
    def __init__(self, *args):
        """create a line

        arguments to initialize:
        line_coords  -- instance of LinePlueckerCoords
        """
        if isinstance(args[0], LinePlueckerMatrix):
            if len(args) != 1:
                raise TypeError("expected one argument")
            self.data = args[0].data[:]
        elif isinstance(args[0], LinePlueckerCoords):
            if len(args) != 1:
                raise TypeError("expected one argument")
            L_i = nx.array([0, 0, 0, 1, 3, 2])
            L_j = nx.array([1, 2, 3, 2, 1, 3])

            Lcoords = args[0].data
            Lmatrix = nx.zeros((4, 4), nx.Float)
            for c, (i, j) in enumerate(zip(L_i, L_j)):
                Lmatrix[i, j] = Lcoords[c]
                Lmatrix[j, i] = -Lcoords[c]
            # numarray version of above:
            # Lmatrix[L_i,L_j]=Lcoords
            # Lmatrix[L_j,L_i]=-Lcoords
            self.data = Lmatrix
        elif len(args) == 4:
            self.data = nx.array(args)
            if self.data.shape != (4, 4):
                raise ValueError("expected 4x4 matrix")
        else:
            raise TypeError("cannot convert arguments to LinePlueckerMatrix")

    def __repr__(self):
        return "LinePlueckerMatrix(*%s)" % (repr(self.data),)


### hidden


def planes2line(*planes):
    """find line formed by intersection of planes"""
    P = nx.array([plane.data for plane in planes])
    u, d, vt = svd(P, full_matrices=True)
    # "two columns of V corresponding to the two largest singular
    # values span the best rank 2 approximation to A and may be
    # used to define the line of intersection of the planes"
    # (Hartley & Zisserman, p. 323)
    P = vt[0, :]  # P,Q are planes (take row because this is transpose(V))
    Q = vt[1, :]

    # directly to Pluecker line coordinates
    Lcoords = (
        -(P[3] * Q[2]) + P[2] * Q[3],
        P[3] * Q[1] - P[1] * Q[3],
        -(P[2] * Q[1]) + P[1] * Q[2],
        -(P[3] * Q[0]) + P[0] * Q[3],
        -(P[2] * Q[0]) + P[0] * Q[2],
        -(P[1] * Q[0]) + P[0] * Q[1],
    )
    return LinePlueckerCoords(*Lcoords)


def planes2point(*planes):
    """find point formed by intersection of planes"""
    A = nx.array([plane.data for plane in planes])
    u, d, vt = svd(A, full_matrices=True)
    Pt = vt[3, :]
    return Point(*Pt)


def is_point_on_plane(plane, point, eps=1e-14):
    return abs(nx.dot(plane.data, point.data)) < eps


### end of hidden


def _test():

    X1 = Plane(2.0, 0.0, 0.0, 1.0)
    X2 = Plane(10.0, 0.0, 0.0, 1.0)
    X3 = Plane(0.0, 20.0, 0.0, 1.0)

    pt1 = PointFromIntersectingPlanes(X1, X2, X3)

    line1 = LineFromIntersectingPlanes(X1, X2)

    print(pt1, "in", X1, "?=", pt1.is_on_plane(X1))
    print(line1)

    pt2 = PointFromIntersectingPlaneAndLine(X3, line1)
    print(pt2)

    print(pt2, "in", X1, "?=", pt1.is_on_plane(X1))
    print(pt2, "in", X2, "?=", pt1.is_on_plane(X2))
    print(pt2, "in", X3, "?=", pt1.is_on_plane(X3))

    print()

    # example 3.4
    lmat = nx.zeros((4, 4), nx.Float)
    lmat[0, 3] = -1
    lmat[3, 0] = 1
    xaxis = LinePlueckerMatrix(*lmat)
    print(xaxis)
    planex1 = Plane(1, 0, 0, -1)
    print(planex1)
    print(PointFromIntersectingPlaneAndLine(planex1, xaxis))


if __name__ == "__main__":
    _test()
