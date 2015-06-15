"""

 1--
 ^  \---                      medium 1
 |       \----
 |            \---
 z1               \----
 |                     \---
 |                         \----
 V                              \---
 ====================================0====== interface
 ^                                    \
 |                                     \
 z2                                     \
 |      medium 2                         \
 v                                        \
 -                                         2

 |<--------------h1----------------->|
 |<--------------------h------------------>|
                                     |<-h2>|

h1 is the horizontal distance from point 1 to point 0.
h is the horitzontal distance from point 1 to point 2.
h2 is the horizontal distance from point 0 to point 2.

n1 is refractive index of medium 1
n2 is refractive index of medium 2

z1 is height of point 1
z2 is depth of point 2
"""
from sympy import *
from sympy.solvers.solvers import unrad # tested with sympy==0.7.3

h1 = Symbol('h1', real=True, positive=True)
h = Symbol('h', real=True, positive=True)
h2 = h-h1

n1 = Symbol('n1', real=True, positive=True)
n2 = Symbol('n2', real=True, positive=True)

z1 = Symbol('z1', real=True, positive=True)
z2 = Symbol('z2', real=True, positive=True)

duration = n1*sqrt( h1*h1 + z1*z1 ) + n2*sqrt(z2*z2 + h2*h2)

ddur_dh1 = diff(duration,h1)
eq, cov, dens = unrad(ddur_dh1)
poly = Poly(eq,h1)
print poly
