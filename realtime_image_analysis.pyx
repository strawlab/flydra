#emacs, this is -*-Python-*- mode
# $Id$

import time
import sys
import numarray as nx
from numarray.ieeespecial import nan

cimport c_numarray
c_numarray.import_libnumarray()

cimport c_lib

# start of IPP-requiring code
cimport ipp
cimport c_fit_params
cimport arena_control

class IPPError(Exception):
    pass

cdef void CHK( ipp.IppStatus errval ) except *:
    if errval != 0:
        raise IPPError("IPP status %d"%(errval,))

cdef void print_info_8u(ipp.Ipp8u* im, int im_step, ipp.IppiSize sz, object prefix):
    cdef ipp.Ipp32f minVal, maxVal
    cdef ipp.IppiPoint minIdx, maxIdx
    
    CHK(
        ipp.ippiMinMaxIndx_8u_C1R( im, im_step, sz,
                                   &minVal, &maxVal,
                                   &minIdx, &maxIdx ))
    print prefix,'min: %f, max: %f, minIdx: %d,%d, maxIdx: %d,%d'%(minVal,maxVal,
                                                                   minIdx.x,minIdx.y,
                                                                   maxIdx.x,maxIdx.y)
# end of IPP-requiring code

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
cdef class RealtimeAnalyzer:

    # full image size
    cdef int width, height
    
    # ROI size
    cdef int _left, _bottom, _right, _top
    
    # runtime parameters
    cdef float _diff_threshold
    cdef float _clear_threshold
    
    cdef int _use_arena
    
    cdef c_numarray._numarray last_image
    cdef int arena_control_working

    # start of IPP-requiring code
    cdef ipp.IppiSize _roi_sz
        
    cdef int im1_step, im2_step, sum_image_step, bg_img_step
    cdef int mean_image_step, std_image_step, sq_image_step, std_img_step
    cdef int n_bg_samples
    cdef int n_rot_samples

    cdef ipp.Ipp8u *im1, *im2 # current image
    cdef ipp.Ipp8u *bg_img, *std_img  # 8-bit background
    cdef ipp.Ipp32f *sum_image, *sq_image # FP accumulators
    cdef ipp.Ipp32f *mean_image, *std_image # FP background
    # end of IPP-requiring code

    def __init__(self, int w, int h):
        self.width = w
        self.height = h
        self.roi = ( 0, 0, self.width-1, self.height-1)

        self._diff_threshold = 8.1
        self._clear_threshold = 0.0
        self._use_arena = 0

        # start of IPP-requiring code
        self.n_bg_samples = 100
        self.n_rot_samples = 100*60 # 1 minute

        # pre- and post-processed images of every frame
        self.im1=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.im1_step )
        if self.im1==NULL: raise MemoryError("Error allocating memory by IPP")
        self.im2=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.im2_step )
        if self.im2==NULL: raise MemoryError("Error allocating memory by IPP")

        # 8u background, std images
        self.bg_img=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.bg_img_step )
        if self.bg_img==NULL: raise MemoryError("Error allocating memory by IPP")
        self.std_img=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.std_img_step )
        if self.std_img==NULL: raise MemoryError("Error allocating memory by IPP")

        # 32f statistics and accumulator images for background collection
        self.sum_image=ipp.ippiMalloc_32f_C1( self.width, self.height, &self.sum_image_step )
        if self.sum_image==NULL: raise MemoryError("Error allocating memory by IPP")

        self.sq_image=ipp.ippiMalloc_32f_C1( self.width, self.height, &self.sq_image_step )
        if self.sq_image==NULL: raise MemoryError("Error allocating memory by IPP")
            
        self.mean_image=ipp.ippiMalloc_32f_C1( self.width, self.height, &self.mean_image_step )
        if self.mean_image==NULL: raise MemoryError("Error allocating memory by IPP")

        self.std_image=ipp.ippiMalloc_32f_C1( self.width, self.height, &self.std_image_step )
        if self.std_image==NULL: raise MemoryError("Error allocating memory by IPP")

        # image moment calculation initialization
        # XXX this is global -- could not use more than one camera per computer
        if c_fit_params.init_moment_state() != 0: 
            raise RuntimeError("could not init moment state")

        # initialize background images
        CHK( ipp.ippiSet_8u_C1R(0,self.bg_img,self.bg_img_step, self._roi_sz))
        CHK( ipp.ippiSet_8u_C1R(0,self.std_img,self.std_img_step, self._roi_sz))

        CHK( ipp.ippiSet_32f_C1R(0.0,self.mean_image,self.mean_image_step, self._roi_sz))
        CHK( ipp.ippiSet_32f_C1R(0.0,self.std_image,self.std_image_step, self._roi_sz))

        try:
            arena_error = arena_control.arena_initialize()
            self.arena_control_working = True
            if arena_error != 0:
                print "WARNING: could not initialize arena control"
                self.arena_control_working = False
        except NameError:
            self.arena_control_working = False
        # end of IPP-requiring code

    def __del__(self):
        # start of IPP-requiring code
        print 'de-allocating IPP memory'
        if self.arena_control_working:
            arena_control.arena_finish()
        ipp.ippiFree(self.im1)
        ipp.ippiFree(self.im2)
        ipp.ippiFree(self.bg_img)
        ipp.ippiFree(self.std_img)
        ipp.ippiFree(self.sum_image)
        ipp.ippiFree(self.sq_image)
        ipp.ippiFree(self.mean_image)
        ipp.ippiFree(self.std_image)
        c_fit_params.free_moment_state()
        # end of IPP-requiring code
        return

    def do_work(self, c_numarray._numarray buf):
        cdef double x0, y0
        cdef double orientation
        cdef int i
        cdef int result
        
        # start of IPP-requiring code
        cdef ipp.Ipp8u max_val
        cdef int index_x,index_y
        # end of IPP-requiring code

        self.last_image = buf # useful when no IPP
        x0,y0=320,240
        orientation = 1
        
        # start of IPP-requiring code
        # copy image to IPP memory
        for i from 0 <= i < self.height:
            c_lib.memcpy(self.im1+self.im1_step*i, # dest
                         buf.data+self.width*i, # src
                         self.width) # length

        # do background subtraction & find max pixel in ROI
        CHK( ipp.ippiAbsDiff_8u_C1R(
            (self.bg_img + self._bottom*self.bg_img_step + self._left), self.bg_img_step,
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step, self._roi_sz))
        CHK( ipp.ippiMaxIndx_8u_C1R(
            (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step,
            self._roi_sz, &max_val, &index_x,&index_y))
        # (to avoid big moment arm:) if pixel < .8*max(pixel): pixel=0
        CHK( ipp.ippiThreshold_Val_8u_C1IR(
            (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step,
            self._roi_sz, self._clear_threshold*max_val, 0, ipp.ippCmpLess))

        if max_val < self._diff_threshold:
            x0=-1
            y0=-1
        else:
            result = c_fit_params.fit_params( &x0, &y0, &orientation,
                                              self._roi_sz.width, self._roi_sz.height,
                                              (self.im2 + self._bottom*self.im2_step + self._left),
                                              self.im2_step )
            if result == 0:
                # note that x0 and y0 are now relative to the ROI origin
                orientation = orientation + 1.57079632679489661923 # (pi/2)
            elif result == 31: orientation = nan
            elif result == 32: orientation = nan
            elif result == 33: orientation = nan
            else:
                raise RuntimeError("c_fit_params.fit_params() failed with error %d"%result)

            if self._use_arena: # call out to arena feedback function
                if self.arena_control_working:
                    arena_control.arena_update(
                        x0, y0, orientation, timestamp, framenumber )
                else:
                    print 'ERROR: no arena control'
                    
            # set x0 and y0 relative to whole frame
            x0 = x0+self._left
            y0 = y0+self._bottom
        # end of IPP-requiring code
        
        return [ (x0, y0, orientation) ]

    def get_working_image(self):
        cdef c_numarray._numarray buf
        cdef int i

        buf = self.last_image # useful when no IPP
        
        # start of IPP-requiring code
        # allocate new numarray memory
        buf = <c_numarray._numarray>c_numarray.NA_NewArray(
            NULL, c_numarray.tUInt8, 2,
            self.height, self.width)

        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(buf.data+self.width*i,
                         self.im2+self.im2_step*i,
                         self.width)
        # end of IPP-requiring code
        return buf

    def clear_background_image(self):
        # start of IPP-requiring code
        CHK( ipp.ippiSet_8u_C1R( 0,
                                 (self.bg_img + self._bottom*self.bg_img_step + self._left),
                                 self.bg_img_step, self._roi_sz))
        CHK( ipp.ippiSet_8u_C1R( 0,
                                 (self.std_img + self._bottom*self.std_img_step + self._left),
                                 self.std_img_step, self._roi_sz))
        # end of IPP-requiring code
        return

    def clear_accumulator_image(self):
        # start of IPP-requiring code
        # divide by 4 because 32f = 4 bytes
        CHK( ipp.ippiSet_32f_C1R( 0.0, 
                                  (self.sum_image + self._bottom*self.sum_image_step/4 + self._left),
                                  self.sum_image_step, self._roi_sz)) 
        CHK( ipp.ippiSet_32f_C1R( 0.0,
                                  (self.sq_image + self._bottom*self.sq_image_step/4 + self._left),
                                  self.sq_image_step, self._roi_sz))
        # end of IPP-requiring code
        return

    def accumulate_last_image(self):
        # start of IPP-requiring code
        CHK( ipp.ippiAdd_8u32f_C1IR(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.sum_image + self._bottom*self.sum_image_step/4 + self._left),
            self.sum_image_step, self._roi_sz))
        CHK( ipp.ippiAddSquare_8u32f_C1IR(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.sq_image + self._bottom*self.sq_image_step/4 + self._left),
            self.sq_image_step, self._roi_sz))
        # end of IPP-requiring code
        return

    def convert_accumulator_to_bg_image(self,int n_bg_samples):
        # start of IPP-requiring code
        # find mean
        CHK( ipp.ippiMulC_32f_C1R(
            (self.sum_image + self._bottom*self.sum_image_step/4 + self._left),
            self.sum_image_step, 1.0/n_bg_samples,
            (self.mean_image + self._bottom*self.mean_image_step/4 + self._left),
            self.mean_image_step, self._roi_sz))
        CHK( ipp.ippiConvert_32f8u_C1R(
            (self.mean_image + self._bottom*self.mean_image_step/4 + self._left),
            self.mean_image_step,
            (self.bg_img + self._bottom*self.bg_img_step + self._left),
            self.bg_img_step, self._roi_sz, ipp.ippRndNear ))

        # find STD (use sum_image as temporary variable
        CHK( ipp.ippiSqr_32f_C1R(
            (self.mean_image + self._bottom*self.mean_image_step/4 + self._left),
            self.mean_image_step,
            (self.sum_image + self._bottom*self.sum_image_step/4 + self._left),
            self.sum_image_step, self._roi_sz))
        CHK( ipp.ippiMulC_32f_C1R(
            (self.sq_image + self._bottom*self.sq_image_step/4 + self._left),
            self.sq_image_step, 1.0/n_bg_samples,
            (self.std_image + self._bottom*self.std_image_step/4 + self._left),
            self.std_image_step, self._roi_sz))
        CHK( ipp.ippiSub_32f_C1IR(
            (self.sum_image + self._bottom*self.sum_image_step/4 + self._left),
            self.sum_image_step,
            (self.std_image + self._bottom*self.std_image_step/4 + self._left),
            self.std_image_step, self._roi_sz))
        CHK( ipp.ippiSqrt_32f_C1IR(
            (self.std_image + self._bottom*self.std_image_step/4 + self._left),
            self.std_image_step, self._roi_sz))

        CHK( ipp.ippiConvert_32f8u_C1R(
            (self.std_image + self._bottom*self.std_image_step/4 + self._left),
            self.std_image_step,
            (self.std_img + self._bottom*self.std_img_step + self._left),
            self.std_img_step, self._roi_sz, ipp.ippRndNear ))
        # end of IPP-requiring code
        return

    def rotation_calculation_init(self):
        # start of IPP-requiring code
        cdef int n_rot_samples
        
        arena_control.rotation_calculation_init()
        
        n_rot_samples = 100*60 # 1 minute
        c_fit_params.start_center_calculation( n_rot_samples )
        # end of IPP-requiring code
        return

    def rotation_update(self, float x0, float y0, float orientation):
        # start of IPP-requiring code
        # convert back to ROI-relative coordinates
        x0 = x0 - self._left
        y0 = y0 - self._bottom
        c_fit_params.update_center_calculation( x0, y0, orientation )
        arena_control.rotation_update()
        # end of IPP-requiring code
        return

    def rotation_end(self):
        # start of IPP-requiring code
        cdef double new_x_cent, new_y_cent
        
        c_fit_params.end_center_calculation( &new_x_cent, &new_y_cent )
        arena_control.rotation_calculation_finish( new_x_cent, new_y_cent )
        # end of IPP-requiring code
        return
        
    property clear_threshold:
        def __get__(self):
            return self._clear_threshold
        def __set__(self,value):
            self._clear_threshold = value
        
    property diff_threshold:
        def __get__(self):
            return self._diff_threshold
        def __set__(self,value):
            self._diff_threshold = value
        
    property use_arena:
        def __get__(self):
            return self._use_arena
        def __set__(self,value):
            self._use_arena = value

    property roi:
        def __get__(self):
            return (self._left,self._bottom,self._right,self._top)
        def __set__(self,lbrt):
            self._left = lbrt[0]
            self._bottom = lbrt[1]
            self._right = lbrt[2]
            self._top = lbrt[3]

            assert self._left >= 0
            assert self._bottom >= 0
            assert self._right < self.width
            assert self._top < self.height

            self._roi_sz.width = self._right-self._left+1
            self._roi_sz.height = self._top-self._bottom+1
