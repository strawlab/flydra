#emacs, this is -*-Python-*- mode
# $Id$

import numarray as nx

cimport c_numarray
c_numarray.import_libnumarray()

cimport c_lib

cimport ipp

class IPPError(Exception):
    pass

cdef void CHK( ipp.IppStatus errval ) except *:
    if errval != 0:
        raise IPPError( ipp.ippCoreGetStatusString( errval ) )

ctypedef class MomentState:
    cdef ipp.IppiMomentState_64f *pState
    def __init__(self):
        CHK( ipp.ippiMomentInitAlloc_64f( &self.pState, ipp.ippAlgHintFast ) )
    def __del__(self):
        CHK( ipp.ippiMomentFree_64f( self.pState ))
    def fill_moment( self, c_numarray._numarray image ):
        cdef ipp.Ipp8u* pSrc
        cdef int srcStep
        cdef ipp.IppiSize roiSize
        cdef int i

        assert len(image.shape) == 2 # can only handle 2D images for now
        assert image.type() == nx.UInt8 # can only handle 8 bit images for now

        roiSize.height, roiSize.width = image.shape
        
        # allocate IPP memory for image
        pSrc=ipp.ippiMalloc_8u_C1( roiSize.width, roiSize.height, &srcStep )
        if pSrc==NULL: raise MemoryError("Error allocating memory by IPP")
        
        # copy image to IPP memory
        for i from 0 <= i < roiSize.height:
            c_lib.memcpy(pSrc+srcStep*i, # dest
                         image.data+roiSize.width*i, # src
                         roiSize.width) # length
            
        # calculate moments
        self.fill_moments_8u_C1R( pSrc, srcStep, roiSize )
        
        # free memory
        ipp.ippiFree(pSrc)
        
    cdef void fill_moments_8u_C1R( self, ipp.Ipp8u* pSrc, int srcStep, ipp.IppiSize roiSize):
        CHK( ipp.ippiMoments64f_8u_C1R( pSrc, srcStep, roiSize, self.pState ) )
    def get_moment(self, typ, int order_M, int order_N, int channel): 
        cdef double result
        cdef ipp.IppiPoint roi_offset
        if typ=='central':
            CHK( ipp.ippiGetCentralMoment_64f( self.pState,
                                               order_M,
                                               order_N,
                                               channel,
                                               &result ))
        elif typ=='spatial':
            roi_offset.x = 0 
            roi_offset.y = 0
            CHK( ipp.ippiGetSpatialMoment_64f( self.pState,
                                               order_M,
                                               order_N,
                                               channel,
                                               roi_offset,
                                               &result ))
        else:
            raise ValueError("don't understand moment type '%s'"%typ)
        return result

