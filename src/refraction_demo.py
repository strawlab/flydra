from sympy import *
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

h1 = Symbol('h1', real=True, positive=True)
h = Symbol('h', real=True, positive=True)
h2 = h-h1

n1 = Symbol('n1', real=True, positive=True)
n2 = Symbol('n2', real=True, positive=True)

z1 = Symbol('z1', real=True, positive=True)
z2 = Symbol('z2', real=True, positive=True)

duration = n1*sqrt( h1*h1 + z1*z1 ) + n2*sqrt(z2*z2 + h2*h2)

# Minimize the duration by calculating the derivative with respect to
# h1 and solve for when it is equal to zero.
ddur_dh1 = diff(duration,h1)
result = solve(ddur_dh1,h1)
print 'result', result

# I had to modify sympy 0.7.3 to print this before raising a PolynomialError:
# Poly((n1**2 - n2**2)*h1**4 +
#       (-2*h*n1**2 + 2*h*n2**2)*h1**3 +
#       (h**2*n1**2 - h**2*n2**2 + n1**2*z2**2 - n2**2*z1**2)*h1**2 +
#       2*h*n2**2*z1**2*h1
#       - h**2*n2**2*z1**2, h1, domain='ZZ[z1,z2,h,n1,n2]')
