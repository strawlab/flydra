#emacs, this is -*-Python-*- mode
import Numeric as na

cimport ipp
cimport c_fit_params

# Done in file where this file is included:
#cimport c_numeric
#c_numeric.import_array()

cdef extern from "stdlib.h":
    ctypedef int size_t
    void *memcpy(void*,void*,size_t)

cdef extern from "Python.h":
    int PyObject_AsReadBuffer( object obj, void **buffer, int *len) except -1
    object PyError_Occurred()

cdef void CHK( int errval ) except *:
    if errval != 0:
        raise RuntimeError("c_fit_params error %d"%errval)

def fit_params(A, index_x=None, index_y=None, centroid_search_radius=50):
    cdef double x0,y0,slope # return values

    cdef int width, height
    cdef ipp.IppiSize sz
    cdef int im_step, im1_step
    cdef ipp.Ipp32f *im
    cdef ipp.Ipp8u *im1
    
    cdef char *buf_ptr
    cdef int buflen
    
    cdef int i
    cdef c_numeric.PyArrayObject* pyarray
    
    #assert A.type() == na.UInt8
    height,width = A.shape
    
    sz.width = width
    sz.height = height

    if index_x is None:
        index_x = int(width/2)
    if index_y is None:
        index_y = int(height/2)
        
    # allocate memory for IPP
    im1=ipp.ippiMalloc_8u_C1( width, height, &im1_step )
    if im1==NULL:
        raise MemoryError("Error allocating memory by IPP")
    im=ipp.ippiMalloc_32f_C1( width, height, &im_step )
    if im==NULL:
        raise MemoryError("Error allocating memory by IPP")
    CHK( c_fit_params.init_moment_state() )

    # convert Python image to ipp
    
    # using numarray:
    #PyObject_AsReadBuffer(A._data,<void**>&buf_ptr,&buflen)
    #for i from 0 <= i < height:
    #    memcpy(im1+im1_step*i,buf_ptr+width*i,width)

    # using Numeric:
    pyarray=<c_numeric.PyArrayObject*>c_numeric.PyArray_ContiguousFromObject( A,c_numeric.PyArray_UBYTE,2,2 )
    for i from 0 <= i < height:
        memcpy(im1+im1_step*i,<void*>(pyarray.data+pyarray.strides[0]*i),width)
        
    CHK(
        ipp.ippiConvert_8u32f_C1R(im1, im1_step,
                                  im, im_step, sz))

    # call into the C function
    CHK( c_fit_params.fit_params( &x0, &y0, &slope,
         index_x, index_y, centroid_search_radius,
         width, height, im, im_step ) )
                    
    # free memory
    CHK( c_fit_params.free_moment_state() )
    ipp.ippiFree(im)
    ipp.ippiFree(im1)
    
    return x0, y0, slope
