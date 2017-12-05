# emacs, this is -*-Python-*- mode

cdef class ThreeTuple:
    cdef readonly double a
    cdef readonly double b
    cdef readonly double c

cdef class PlueckerLine:
    cdef readonly ThreeTuple u
    cdef readonly ThreeTuple v
