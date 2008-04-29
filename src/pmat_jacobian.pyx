#emacs, this is -*-Python-*- mode
import numpy
"""
import sympy

# Let (u,v) be coorindates on the image plane.  (u,v) = (r/t, s/t)
# where (r,s,t) = P (x,y,z,w) with (r,s,t) being 2D homogeneous
# coords, (x,y,z,w) are 3D homogeneous coords, and P is the 3x4 camera calibration matrix.

# write out equations for u and v
u=sympy.sympify('(P00*x + P01*y + P02*z + P03*w)/(P20*x + P21*y + P22*z + P23*w)')
v=sympy.sympify('(P10*x + P11*y + P12*z + P13*w)/(P20*x + P21*y + P22*z + P23*w)')

# now take partial derivatives
pu_x = sympy.diff(u,x)
pu_y = sympy.diff(u,y)
pu_z = sympy.diff(u,z)
pu_w = sympy.diff(u,w)

pv_x = sympy.diff(v,x)
pv_y = sympy.diff(v,y)
pv_z = sympy.diff(v,z)
pv_w = sympy.diff(v,w)

"""

cdef class PinholeCameraModelWithJacobian:
    """Represent the 3D projection of a camera.

    Camera projection is a non-linear process where the 3D coordinate
    X = (x,y,z) is transformed to the 2D coordinate Y = (r,s) on the
    image plane by the function h:

                          Y = h(X)

    This class implements this function and allows one to evaluate the
    jacobian at various X, as well.

    """
    def __init__(self,P):
        self.pmat = P
        self.P00, self.P01, self.P02, self.P03 = P[0,:]
        self.P10, self.P11, self.P12, self.P13 = P[1,:]
        self.P20, self.P21, self.P22, self.P23 = P[2,:]
    cdef void evaluate_jacobian_at_(self,double x,double y,double z,double w,
                                    double *ux, double *uy, double *uz, double *uw,
                                    double *vx, double *vy, double *vz, double *vw):
        """This is implemented for 3D homogeneous vector (x,y,z,w)"""

        #sympy.diff(u,x)
        ux[0]=-self.P20*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P00*x + self.P01*y + self.P02*z + self.P03*w) + self.P00/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(u,y)
        uy[0]=-self.P21*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P00*x + self.P01*y + self.P02*z + self.P03*w) + self.P01/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(u,z)
        uz[0]=-self.P22*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P00*x + self.P01*y + self.P02*z + self.P03*w) + self.P02/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(u,w)
        uw[0]=-self.P23*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P00*x + self.P01*y + self.P02*z + self.P03*w) + self.P03/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)

        #sympy.diff(v,x)
        vx[0]=-self.P20*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P10*x + self.P11*y + self.P12*z + self.P13*w) + self.P10/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(v,y)
        vy[0]=-self.P21*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P10*x + self.P11*y + self.P12*z + self.P13*w) + self.P11/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(v,z)
        vz[0]=-self.P22*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P10*x + self.P11*y + self.P12*z + self.P13*w) + self.P12/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
        #sympy.diff(v,w)
        vw[0]=-self.P23*(self.P20*x + self.P21*y + self.P22*z + self.P23*w)**(-2)*(self.P10*x + self.P11*y + self.P12*z + self.P13*w) + self.P13/(self.P20*x + self.P21*y + self.P22*z + self.P23*w)
#        return ux, uy, uz, uw, vx, vy, vz, vw
    def evaluate_jacobian_from_homogeneous(self,X):
        cdef double ux, uy, uz, uw
        cdef double vx, vy, vz, vw
        cdef double w

        x,y,z,w=X
        self.evaluate_jacobian_at_(x,y,z,w, &ux, &uy, &uz, &uw,
                                   &vx, &vy, &vz, &vw)
        J=numpy.array([[ux,uy,uz,uw],
                       [vx,vy,vz,vw]],
                      dtype=numpy.float64)
        return J
    def evaluate_jacobian_at(self,X):
        cdef double ux, uy, uz, uw
        cdef double vx, vy, vz, vw
        cdef double w

        x,y,z=X[:3]
        w=1.0
        self.evaluate_jacobian_at_(x,y,z,w, &ux, &uy, &uz, &uw,
                                   &vx, &vy, &vz, &vw)
        J=numpy.array([[ux,uy,uz],
                       [vx,vy,vz]],
                      dtype=numpy.float64)
        return J
    def __call__(self, X):
        """evaluate the non-linear function

        Y = h(X)
        """
        predicted3d_homogeneous = numpy.ones((4,1))
        predicted3d_homogeneous[:3,0] = X[:3]
        uvw = numpy.dot( self.pmat, predicted3d_homogeneous )
        Y = (uvw / uvw[2])[:2,0]
        return Y
