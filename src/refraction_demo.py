from sympy import *

h1 = Symbol('h1', real=True, positive=True)
h = Symbol('h', real=True, positive=True)
h2 = h-h1

n1 = Symbol('n1', real=True, positive=True)
n2 = Symbol('n2', real=True, positive=True)

z1 = Symbol('z1', real=True, positive=True)
z2 = Symbol('z2', real=True, positive=True)

duration = n1*sqrt( h1*h1 + z1*z1 ) + n2*sqrt(z2*z2 + h2*h2)
ddur_dh1 = diff(duration,h1)
print ddur_dh1

expr = h1**2*n1**2/(h1**2+z1**2) - h2**2*n2**2/(h2**2+z2**2)
print expr

#print solve(expr,h1)
print expand(expr)

print
expr = h1**2*n1**2*(h2**2+z2**2) - h2**2*n2**2*(h1**2+z1**2)
print expr
print expand(expr)

print
print expr.factor(h1)
