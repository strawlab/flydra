#emacs, this is -*-Python-*- mode

# This code shamelessly stolen from PyTables, copyright (c) Francesc
# Alted

# Structs and functions from numarray
cdef extern from "stdlib.h":
    ctypedef int size_t
    void *memcpy(void*,void*,size_t)
    void *malloc(size_t size)
    void free(void* ptr)
    int abs(int i)
    double round( double f)
    double ceil( double f)
    float ceilf( float f)    
