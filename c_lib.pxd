#emacs, this is -*-Python-*- mode

# Structs and functions from numarray
cdef extern from "stdlib.h":
    ctypedef int size_t
    void *memcpy(void*,void*,size_t)
    void *malloc(size_t size)
    void free(void* ptr)
    void exit(int status)
    int printf(char *format)
    
cdef extern from "math.h":
    int abs(int i)
    double round( double f)
    double ceil( double f)
    float ceilf( float f)
    double atan2( double y, double x)
    double sqrt(double)
    int isnan(double x)
    int isinf(double x)
