#emacs, this is -*-Python-*- mode

cdef extern from "stdlib.h":
    ctypedef int size_t
    void *memcpy(void*,void*,size_t)
    void *memset(void*,int,size_t)
    void *malloc(size_t size)
    void free(void* ptr)
    void exit(int status)
    int printf(char *format, ...)
    ctypedef int FILE

cdef extern from "math.h":
    int abs(int i)
    double round( double f)
    double ceil( double f)
    float ceilf( float f)
    double atan2( double y, double x)
    double sqrt(double)
    int isnan(double x)
    int isinf(double x)
    double sin(double x)
    float sinf(float x)

cdef extern from "ads_wrap_system.h":
    int close(int fd)
    size_t write( int fd, void* buf, size_t count)
    
cdef extern from "stdio.h":
    size_t fwrite( void* ptr, size_t size, size_t nmemb, FILE* stream)
    
