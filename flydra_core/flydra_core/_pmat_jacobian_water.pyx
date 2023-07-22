#emacs, this is -*-Python-*- mode
#cython: language_level=2
import numpy as np
cimport numpy as np

import flydra_core.water as water
cimport _refraction
import _refraction
from libc.math cimport atan2, sqrt, cos, sin
import _pmat_jacobian

def make_PinholeCameraWaterModelWithJacobian(*args,**kw):
    return PinholeCameraWaterModelWithJacobian(*args,**kw)

def pmat2cam_center(P):
    """

    See Hartley & Zisserman (2003) p. 163
    """
    assert P.shape == (3, 4)
    determinant = np.linalg.det

    # camera center
    X = determinant([P[:, 1], P[:, 2], P[:, 3]])
    Y = -determinant([P[:, 0], P[:, 2], P[:, 3]])
    Z = determinant([P[:, 0], P[:, 1], P[:, 3]])
    T = -determinant([P[:, 0], P[:, 1], P[:, 2]])

    C_ = np.transpose(np.array([[X / T, Y / T, Z / T]]))
    return C_

cdef class PinholeCameraWaterModelWithJacobian(_pmat_jacobian.PinholeCameraModelWithJacobian):
    def __init__(self,P,wateri,roots3and4_eps):
        self.pmat = P
        self.wateri = wateri
        assert isinstance(wateri, water.WaterInterface)
        self.n1 = wateri.n1
        self.n2 = wateri.n2
        self.roots3and4_eps = roots3and4_eps

        C = pmat2cam_center(self.pmat)
        C = C[:,0]
        self.camx, self.camy, self.camz = C

        self.shift = np.zeros((3,))
        self.shift[0] = self.camx
        self.shift[1] = self.camy

        self.delta=0.001

        self.dx = np.array( (self.delta, 0,          0) )
        self.dy = np.array( (0,          self.delta, 0) )

        self.pinhole = _pmat_jacobian.PinholeCameraModelWithJacobian(self.pmat)

    def __reduce__(self):
        """this allows PinholeCameraWaterModelWithJacobian to be pickled"""
        args = (self.pmat,self.wateri)
        return (make_PinholeCameraWaterModelWithJacobian, args)

    cpdef evaluate_jacobian_at(self, np.ndarray[np.double_t, ndim=1] X):
        cdef int flip_z
        cdef np.ndarray[np.double_t, ndim=1] eval_z
        cdef np.ndarray[np.double_t, ndim=1] dz
        cdef double z_delta

        # evaluate the Jacobian numerically
        F = self.evaluate(X)

        Fx = self.evaluate(X + self.dx)
        Fy = self.evaluate(X + self.dy)
        z_delta = min( abs(X[2]), self.delta ) # do not allow z>0
        dz = np.array( (0, 0, z_delta) )

        eval_z = X+dz
        if eval_z[2] > 0:
            flip_z = False
        else:
            flip_z = True
            eval_z = X-dz
        Fz = self.evaluate(eval_z)

        dF_dx = (Fx-F)/self.delta
        dF_dy = (Fy-F)/self.delta
        dF_dz = (Fz-F)/self.delta
        if flip_z:
            dF_dz = -dF_dz

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

        if depth < 0:
            # do not use refraction model if point above water
            return self.pinhole.evaluate( X )

        theta = atan2( s1, s0 )
        r = sqrt( s0*s0 + s1*s1 )
        r0 = _refraction.find_fastest_path_fermat( self.n1, self.n2, self.camz,
                                                   r, depth, self.roots3and4_eps )

        t0 = r0*cos(theta)
        t1 = r0*sin(theta)

        surfX_shifted = np.array((t0,t1,0),dtype=np.float64)
        surfX = surfX_shifted+self.shift
        result = self.pinhole.evaluate(surfX)
        return result
