#cython: language_level=2
cimport _fastgeom

cpdef _fastgeom.ThreeTuple line_fit_3d(_fastgeom.PlueckerLine line_pluecker, _fastgeom.ThreeTuple mu, sigma_inv)
cpdef double dist2(_fastgeom.ThreeTuple x,_fastgeom.ThreeTuple mu,sigma_inv)
