cimport _fastgeom

cdef _fastgeom.ThreeTuple _line_fit_3d(_fastgeom.PlueckerLine line_pluecker, _fastgeom.ThreeTuple mu, sigma_inv)
cdef double _dist2(_fastgeom.ThreeTuple x,_fastgeom.ThreeTuple mu,sigma_inv)
