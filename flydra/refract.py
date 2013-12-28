# -*- coding: utf-8 -*-
from __future__ import division
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
z1: height of point 1
z2: depth  of point 2
h1: horizontal distance between points 1,0
h2: horizontal distance between points 2,0
h:  horizontal distance between points 1,2

duration = n1*sqrt( h1*h1 + z1*z1 ) + n2*sqrt(z2*z2 + h2*h2)
ddur_dh1 = diff(duration,h1)
#print solve(ddur_dh1,dh1) # fails due to quartic polynomial
"""

def fermat1(n1,n2,z1,h,z2):
    # see refraction_demo.py for factorization
    a = (n1**2 - n2**2)
    b = (-2*h*n1**2 + 2*h*n2**2)
    c = (h**2*n1**2 - h**2*n2**2 + n1**2*z2**2 - n2**2*z1**2)
    d = 2*h*n2**2*z1**2
    e = -h**2*n2**2*z1**2

    roots = _Roots3And4.roots(a,b,c,d,e)

    # According to
    #
    #  Glaeser, G., & Schröcker, H. (2000). Reflections on
    #  Refractions. JOURNAL FOR GEOMETRY AND GRAPHICS VOLUME, 4(1),
    #  1–18.
    #
    # the following is correct:

    valid = []
    for r in roots:
        if abs(r.imag) == 0:
            r2 = r.real
            if r2 <= h:
                valid.append(r2)
    assert len(valid)==1
    result = valid[0]
    return result
