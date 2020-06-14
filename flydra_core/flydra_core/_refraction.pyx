# -*- coding: utf-8 -*-
#cython: language_level=2
cimport _Roots3And4
import _Roots3And4

"""
This code calculates rays according to [Fermat's principle of least
time](http://en.wikipedia.org/wiki/Fermat's_principle). Light
traveling from point 1 to point 2 (or vice-versa) takes the path of
least time.

 1--
    \---                      medium 1
        \----
             \---
                 \----
                      \---
                          \----
                               \---
 ====================================0====== interface
                                      \
                                       \
                                        \
      medium 2                           \
                                          \
                                           2


n1: refractive index of medium 1
n2: refractive index of medium 2
z1: height of point 1 (always positive)
z2: depth  of point 2 (always positive)
h1: horizontal distance between points 1,0 (always positive)
h2: horizontal distance between points 2,0 (always positive)
h:  horizontal distance between points 1,2 (always positive)

duration = n1*sqrt( h1*h1 + z1*z1 ) + n2*sqrt(z2*z2 + h2*h2)
ddur_dh1 = diff(duration,h1)
#print solve(ddur_dh1,dh1) # fails due to quartic polynomial
"""

cpdef double find_fastest_path_fermat(double n1,double n2,double z1,double h,double z2,double eps) except *:
    cdef double result
    cdef double a,b,c,d,e

    if z2==0.0:
        return h

    # see refraction_demo.py for factorization
    a = (n1**2 - n2**2)
    b = (-2*h*n1**2 + 2*h*n2**2)
    c = (h**2*n1**2 - h**2*n2**2 + n1**2*z2**2 - n2**2*z1**2)
    d = 2*h*n2**2*z1**2
    e = -h**2*n2**2*z1**2

    # According to
    #
    #  Glaeser, G., & Schröcker, H. (2000). Reflections on
    #  Refractions. JOURNAL FOR GEOMETRY AND GRAPHICS VOLUME, 4(1),
    #  1–18.
    #
    # we can choose the real root less than h. I (ADS) also found that
    # it should be non-negative.

    result = _Roots3And4.real_nonnegative_root_less_than(a,b,c,d,e,h, eps)
    return result
