#emacs, this is -*-Python-*- mode

import numarray as na

# Pyrex stuff:
cimport ipp

cdef extern from "stdlib.h":
    ctypedef int size_t
    void *memcpy(void*,void*,size_t)

cdef extern from "Python.h":
    int PyObject_AsReadBuffer( object obj, void **buffer, int *len) except -1

cdef void CHK(ipp.IppStatus status) except *:
    if (status < ipp.ippStsNoErr):
        raise IPPError(IppStatus2str(status))
    elif (status > ipp.ippStsNoErr):
        warnings.warn(IppStatus2str(status))



cdef void _fit_params( float *x0, float *y0, float *orientation,
                       int index_x, int index_y, int centroid_search_radius,
                       int width, int height, ipp.Ipp32f *im, int im_step,
                       ipp.IppiMomentState_64f *pState ):
    
    # This function does the computation.
    #
    # This function could just as well be in C,
    # but I prefer Pyrex.

    cdef int left, right, bottom, top
    cdef ipp.IppiSize roi_sz
    cdef ipp.Ipp32f *roi_start
    cdef ipp.IppiPoint roi_offset
    cdef ipp.Ipp64f Mu00, Mu10, Mu01
    
    left   = index_x - centroid_search_radius
    right  = index_x + centroid_search_radius
    bottom = index_y - centroid_search_radius
    top    = index_y + centroid_search_radius

    if left < 0:
        left = 0
    if right >= width:
        right = width-1
    if bottom < 0:
        bottom = 0
    if top >= height:
        top = height-1

    roi_sz.width = right-left+1
    roi_sz.height = top-bottom+1

    roi_start = im + (im_step/4)*bottom + left
    CHK( ipp.ippiMoments64f_32f_C1R( roi_start, im_step, roi_sz, pState))

    roi_offset.x = left
    roi_offset.y = bottom
    CHK( ipp.ippiGetSpatialMoment_64f( pState, 0, 0, 0, roi_offset, &Mu00 ))
    if Mu00 == 0.0:
        x0[0]=-1 # *x0=-1 (Pyrex has no * operator)
        y0[0]=-1
    else:
        CHK( ipp.ippiGetSpatialMoment_64f( pState, 1, 0, 0, roi_offset, &Mu10 ))
        CHK( ipp.ippiGetSpatialMoment_64f( pState, 0, 1, 0, roi_offset, &Mu01 ))

        x0[0]=Mu10/Mu00
        y0[0]=Mu01/Mu00

    # dummy assignment as placeholder for orientation computation
    orientation[0]=1.0

def fit_params(A, index_x=None, index_y=None, centroid_search_radius=10):
    """find 'center of gravity' and orientation in image"""

    # This function is a bridge between Python and C, getting the data
    # from a numarray and converting it to an IPP image.
    
    cdef int width, height
    cdef ipp.IppiSize sz
    cdef int im_step, im1_step
    cdef ipp.Ipp32f *im
    cdef ipp.Ipp8u *im1
    cdef ipp.IppiMomentState_64f *pState
    
    cdef char *buf_ptr
    cdef int buflen
    
    cdef int i
    
    cdef float x0, y0 # centroid
    cdef float orientation

    assert A.type() == na.UInt8
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
    CHK(
        ipp.ippiMomentInitAlloc_64f(&pState, ipp.ippAlgHintFast))

    # convert Python image to ipp
    PyObject_AsReadBuffer(A._data,<void**>&buf_ptr,&buflen)
    for i from 0 <= i < height:
        memcpy(im1+im1_step*i,buf_ptr+width*i,width)
        
    CHK(
        ipp.ippiConvert_8u32f_C1R(im1, im1_step,
                                  im, im_step, sz))
               
    # call into the C function
    _fit_params( &x0, &y0, &orientation,
                 index_x, index_y, centroid_search_radius,
                 width, height, im, im_step, pState )
                    

    # free memory
    CHK(
        ipp.ippiMomentFree_64f(pState))
    ipp.ippiFree(im)
    ipp.ippiFree(im1)
    
    return x0, y0, orientation
