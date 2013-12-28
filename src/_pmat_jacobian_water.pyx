#emacs, this is -*-Python-*- mode
import numpy as np
cimport numpy as np
import flydra.water as water
import flydra.reconstruct as reconstruct
import flydra.refract as refract
from math cimport atan2, sqrt, cos, sin
import _pmat_jacobian

def make_PinholeCameraWaterModelWithJacobian(*args,**kw):
    return PinholeCameraWaterModelWithJacobian(*args,**kw)

cdef class PinholeCameraWaterModelWithJacobian(_pmat_jacobian.PinholeCameraModelWithJacobian):
    def __init__(self,P,wateri):
        self.pmat = P
        self.wateri = wateri
        assert isinstance(wateri, water.WaterInterface)
        self.n1 = wateri.n1
        self.n2 = wateri.n2

        C = reconstruct.pmat2cam_center(self.pmat)
        C = C[:,0]
        self.camx, self.camy, self.camz = C

        self.shift = np.zeros((3,))
        self.shift[0] = self.camx
        self.shift[1] = self.camy

        self.delta=0.001

        self.dx = np.array( (self.delta, 0,          0) )
        self.dy = np.array( (0,          self.delta, 0) )
        self.dz = np.array( (0,                      0,     self.delta) )

        self.pinhole = _pmat_jacobian.PinholeCameraModelWithJacobian(self.pmat)

    def __reduce__(self):
        """this allows PinholeCameraWaterModelWithJacobian to be pickled"""
        args = (self.pmat,self.wateri)
        return (make_PinholeCameraWaterModelWithJacobian, args)

    cpdef evaluate_jacobian_at(self, np.ndarray[np.double_t, ndim=1] X):
        # evaluate the Jacobian numerically
        F = self.evaluate(X)

        Fx = self.evaluate(X + self.dx)
        Fy = self.evaluate(X + self.dy)
        Fz = self.evaluate(X + self.dz)

        dF_dx = (Fx-F)/self.delta
        dF_dy = (Fy-F)/self.delta
        dF_dz = (Fz-F)/self.delta

        result = np.array([dF_dx,
                           dF_dy,
                           dF_dz]).T

        return result

    cpdef evaluate(self, np.ndarray[np.double_t, ndim=1] X):
        """evaluate the non-linear function

        Y = h(X)
        """
        cdef double depth, r, theta, r0
        cdef double s1, s0

        cdef epsilon = 1e-16
        cdef scale=1.0
        cdef np.ndarray[np.double_t, ndim=1] surfX

        assert len(X)==3

        pt_shifted = X-self.shift
        s0,s1,depth = pt_shifted
        depth = -depth # convert from -z to z

        theta = atan2( s1, s0 )
        r = sqrt( s0*s0 + s1*s1 )
        r0 = refract.fermat1( self.n1, self.n2, self.camz,
                              r, depth)

        t0 = r0*cos(theta)
        t1 = r0*sin(theta)

        surfX_shifted = np.array((t0,t1,0),dtype=np.float)
        surfX = surfX_shifted+self.shift
        result = self.pinhole.evaluate(surfX)
        return result
