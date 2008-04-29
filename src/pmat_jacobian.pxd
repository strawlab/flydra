#emacs, this is -*-Python-*- mode
cdef class PinholeCameraModelWithJacobian:
    cdef double P00, P01, P02, P03
    cdef double P10, P11, P12, P13
    cdef double P20, P21, P22, P23
    cdef object pmat
    cdef void evaluate_jacobian_at_(self,double x,double y,double z,double w,
                                    double *ux, double *uy, double *uz, double *uw,
                                    double *vx, double *vy, double *vz, double *vw)

