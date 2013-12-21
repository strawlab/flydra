#emacs, this is -*-Python-*- mode
import numpy
cimport _fastgeom

def line_fit_2d(line_2d, mu, sigma_inv):
    """find the point closest to mu on the line according to Mahalanois distance"""
    cdef double x0, x1, S00, S01, S10, S11, a,b,c,d

    a,c,b,d = line_2d # parametric representation of line passing through (a,c) in direction (b,d)

    S00 = sigma_inv[0,0]
    S01 = sigma_inv[0,1]
    S10 = sigma_inv[1,0]
    S11 = sigma_inv[1,1]

    x0 = mu[0]
    x1 = mu[1]

    # this was generated with sympy (see mahal.history)
    s = 1.0/(-2*S01*b*d - 2*S10*b*d - 2*S00*b**2 - 2*S11*d**2)*(S01*a*d + S01*b*c + S10*a*d + S10*b*c - S01*b*x1 - S01*d*x0 - S10*b*x1 - S10*d*x0 - 2*S00*b*x0 - 2*S11*d*x1 + 2*S00*a*b + 2*S11*c*d)

    loc2d = (a+s*b,c+s*d)
    return loc2d

def line_fit_3d(_fastgeom.PlueckerLine line_pluecker, mu, sigma_inv):
    """find the point closest to mu on the line according to Mahalanobis distance"""

    # The idea -- Because the Mahalanobis distance function is convex,
    # find the inflection point (where the derivative is zero).

    cdef double x0, x1, x2, S00, S01, S02, S10, S11, S12, S20, S21, S22, a,b,c,d,e,f

    cdef _fastgeom.ThreeTuple lu, lv

    lu = line_pluecker.u
    lv = line_pluecker.v

    a,b,c = lu.a, lu.b, lu.c
    d,e,f = lv.a, lv.b, lv.c

    # slice into potentially bigger arrays
    x0,x1,x2 = mu[:3]

    S00,S01,S02 = sigma_inv[0,:3]
    S10,S11,S12 = sigma_inv[1,:3]
    S20,S21,S22 = sigma_inv[2,:3]

    # this was generated with sympy (see mahal.history)
    s = 1.0/(-2*S01*d*e - 2*S02*d*f - 2*S10*d*e - 2*S12*e*f - 2*S20*d*f - 2*S21*e*f - 2*S00*d**2 - 2*S11*e**2 - 2*S22*f**2)*(S01*a*e + S01*b*d + S02*a*f + S02*c*d + S10*a*e + S10*b*d + S12*b*f + S12*c*e + S20*a*f + S20*c*d + S21*b*f + S21*c*e - S01*d*x1 - S01*e*x0 - S02*d*x2 - S02*f*x0 - S10*d*x1 - S10*e*x0 - S12*e*x2 - S12*f*x1 - S20*d*x2 - S20*f*x0 - S21*e*x2 - S21*f*x1 - 2*S00*d*x0 - 2*S11*e*x1 - 2*S22*f*x2 + 2*S00*a*d + 2*S11*b*e + 2*S22*c*f)

    loc3d = (a+s*d, b+s*e, c+s*f)
    return loc3d

def dist2(x,mu,sigma_inv,least_dimensions_fixup=True):
    """return the squared Mahalanobis distance"""
    if least_dimensions_fixup:
        ndim = min(len(x),len(mu))
        x = numpy.asarray(x[:ndim])
        mu = numpy.asarray(mu[:ndim])
        sigma_inv = numpy.asarray(sigma_inv[:ndim,:ndim])
    sub = x-mu
    return numpy.dot( sub.T, numpy.dot( sigma_inv, sub ) )
