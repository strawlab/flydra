#emacs, this is -*-Python-*- mode
#cython: language_level=2

"""calculate jacobian matrices for pinhole camera model

see pinhole_jacobian_demo.py for the derivation of the math in this module.
"""
import numpy
import numpy as np
cimport numpy as np

def make_PinholeCameraModelWithJacobian(*args,**kw):
    return PinholeCameraModelWithJacobian(*args,**kw)

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

    def __reduce__(self):
        """this allows PinholeCameraModelWithJacobian to be pickled"""
        args = (self.pmat,)
        return (make_PinholeCameraModelWithJacobian, args)

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
    cpdef evaluate_jacobian_at(self, np.ndarray[np.double_t, ndim=1] X): # tested on Cython 0.19.2
        cdef double ux, uy, uz, uw
        cdef double vx, vy, vz, vw
        cdef double w

        if len(X)==4:
            assert X[3]==1.0
        else:
            assert len(X)==3
        x,y,z=X[:3]
        w=1.0
        self.evaluate_jacobian_at_(x,y,z,w, &ux, &uy, &uz, &uw,
                                   &vx, &vy, &vz, &vw)
        J=numpy.array([[ux,uy,uz],
                       [vx,vy,vz]],
                      dtype=numpy.float64)
        return J
    cpdef evaluate(self, np.ndarray[np.double_t, ndim=1] X):
        """evaluate the non-linear function

        Y = h(X)
        """

        if len(X)==4:
            assert X[3]==1.0
        else:
            assert len(X)==3

        predicted3d_homogeneous = numpy.ones((4,1))
        predicted3d_homogeneous[:3,0] = X[:3]
        uvw = numpy.dot( self.pmat, predicted3d_homogeneous )
        Y = (uvw / uvw[2])[:2,0]
        return Y
