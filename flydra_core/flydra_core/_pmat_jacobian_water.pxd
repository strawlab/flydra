#emacs, this is -*-Python-*- mode
#cython: language_level=2
cimport numpy as np
cimport _pmat_jacobian

cdef class PinholeCameraWaterModelWithJacobian(_pmat_jacobian.PinholeCameraModelWithJacobian):
    cdef object wateri
    cdef double camx, camy, camz
    cdef double n1, n2
    cdef object shift
    cdef _pmat_jacobian.PinholeCameraModelWithJacobian pinhole
    cdef double delta
    cdef double roots3and4_eps
    cdef object dx,dy

    cpdef evaluate_jacobian_at(self, np.ndarray[np.double_t, ndim=1] X)
    cpdef evaluate(self, np.ndarray[np.double_t, ndim=1] X)
