"""show how the math in _pmat_jacobian.pyx was derived"""
import sympy

# Let (u,v) be coorindates on the image plane.  (u,v) = (r/t, s/t)
# where (r,s,t) = P (x,y,z,w) with (r,s,t) being 2D homogeneous
# coords, (x,y,z,w) are 3D homogeneous coords, and P is the 3x4 camera calibration matrix.

# write out equations for u and v
x,y,z,w = sympy.var('x y z w')
r=sympy.sympify('(P00*x + P01*y + P02*z + P03*w)')
s=sympy.sympify('(P10*x + P11*y + P12*z + P13*w)')
t=sympy.sympify('(P20*x + P21*y + P22*z + P23*w)')

u=r/t
v=s/t

# now take partial derivatives
pu_x = sympy.diff(u,x)
pu_y = sympy.diff(u,y)
pu_z = sympy.diff(u,z)
pu_w = sympy.diff(u,w)

pv_x = sympy.diff(v,x)
pv_y = sympy.diff(v,y)
pv_z = sympy.diff(v,z)
pv_w = sympy.diff(v,w)

print 'pu_x=',pu_x
print 'pu_y=',pu_y
print 'pu_z=',pu_z
print 'pu_w=',pu_w
print
print 'pv_x=',pv_x
print 'pv_y=',pv_y
print 'pv_z=',pv_z
print 'pv_w=',pv_w
