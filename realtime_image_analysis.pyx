#emacs, this is -*-Python-*- mode
# $Id$

import time
import sys
import numarray as nx
import numarray.ieeespecial
import numarray.linear_algebra

cdef double nan
nan = numarray.ieeespecial.nan

near_inf = 9.999999e20
MAX_NUM_POINTS = 5

cimport c_numarray

cimport c_lib
cimport c_python

cimport ipp
cimport ArenaController
import ArenaController

cdef extern from "c_fit_params.h":
    ctypedef enum CFitParamsReturnType:
        CFitParamsNoError
        CFitParamsZeroMomentError
        CFitParamsOtherError
        CFitParamsCentralMomentError
    CFitParamsReturnType fit_params( ipp.IppiMomentState_64f *pState, double *x0, double *y0,
                                     double *Mu00,
                                     double *Uu11, double *Uu20, double *Uu02,
                                     int width, int height, unsigned char *img, int img_step )

cdef extern from "eigen.h":
    int eigen_2x2_real( double A, double B, double C, double D,
                        double *evalA, double *evecA1,
                        double *evalB, double *evecB1 )

cdef extern from "flydra_ipp_macros.h":
    ipp.Ipp8u*  IMPOS8u(  ipp.Ipp8u*  im, int step, int bottom, int left)
    ipp.Ipp32f* IMPOS32f( ipp.Ipp32f* im, int step, int bottom, int left)
    void CHK( ipp.IppStatus errval )
    
class IPPError(Exception):
    pass

cdef void SET_ERR( int errval ):
    # This is rather crude at the moment because calls to the Python C
    # API cannot be made.  (This code is executed when we have
    # released the GIL.)
    c_lib.printf("SET_ERR called! (May not have GIL, cannot raise exception.)\n")
    c_lib.exit(2)

if sys.platform == 'win32':
    time_func = time.clock
else:
    time_func = time.time
    
cdef class RealtimeAnalyzer:

    # full image size
    cdef int width, height

    # number of images in accumulator
    cdef float alpha
    
    # ROI size
    cdef int _left, _bottom, _right, _top

    cdef int _roi2_radius
    
    # runtime parameters
    cdef float _diff_threshold
    cdef float _clear_threshold
    
    cdef int _use_arena
    
    cdef c_numarray._numarray last_image

    cdef int hw_roi_w, hw_roi_h, hw_roi_l, hw_roi_b # hardware region of interest

    # calibration matrix
    cdef c_numarray._numarray _pmat, _pmat_inv, camera_center
    cdef object _helper

    cdef int index_x,index_y
    cdef ipp.Ipp8u _despeckle_threshold
    
    cdef ipp.IppiSize _roi_sz
        
    cdef int im1_step, im2_step, accum_image_step, bg_img_step
    cdef int n_rot_samples

    cdef ipp.Ipp8u *im1, *im2 # current image
    cdef ipp.Ipp8u *bg_img # 8-bit background
    cdef ipp.Ipp32f *accum_image # FP accumulator

    cdef ipp.IppiMomentState_64f *pState

    cdef ArenaController.ArenaController arena_controller

    def __init__(self, int w, int h, int _hw_roi_w, int _hw_roi_h,
                 int _hw_roi_l, int _hw_roi_b, float alpha):
        self.width = w
        self.height = h
        self.hw_roi_w = _hw_roi_w # hardware region of interest
        self.hw_roi_h = _hw_roi_h
        self.hw_roi_l = _hw_roi_l
        self.hw_roi_b = _hw_roi_b
        self.alpha = alpha
        self.roi = ( 0, 0, self.width-1, self.height-1)


        self._roi2_radius = 10
        self._diff_threshold = 8.1
        self._clear_threshold = 0.0
        self._use_arena = 0
        
        self._pmat = None
        self._pmat_inv = None

        self._helper = None

        self._despeckle_threshold = 5
        self.n_rot_samples = 100*60 # 1 minute

        # pre- and post-processed images of every frame
        self.im1=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.im1_step )
        if self.im1==NULL: raise MemoryError("Error allocating memory by IPP")
        self.im2=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.im2_step )
        if self.im2==NULL: raise MemoryError("Error allocating memory by IPP")

        # 8u background
        self.bg_img=ipp.ippiMalloc_8u_C1( self.width, self.height, &self.bg_img_step )
        if self.bg_img==NULL: raise MemoryError("Error allocating memory by IPP")

        # 32f statistics and accumulator images for background collection
        self.accum_image=ipp.ippiMalloc_32f_C1( self.width, self.height, &self.accum_image_step )
        if self.accum_image==NULL: raise MemoryError("Error allocating memory by IPP")

        # image moment calculation initialization
        CHK( ipp.ippiMomentInitAlloc_64f( &self.pState, ipp.ippAlgHintFast ) )

        # initialize background images
        self.clear_background_image()

        try:
            self.arena_controller = ArenaController.ArenaController()
        except Exception, exc:
            print 'WARNING: could not create ArenaController:',exc.__class__,str(exc)
            self.arena_controller = None

    def __dealloc__(self):
        CHK( ipp.ippiMomentFree_64f( self.pState ))
        ipp.ippiFree(self.im1)
        ipp.ippiFree(self.im2)
        ipp.ippiFree(self.bg_img)
        ipp.ippiFree(self.accum_image)

    def set_hw_roi(self, int _hw_roi_w, int _hw_roi_h,
                   int _hw_roi_l, int _hw_roi_b):
        self.hw_roi_w = _hw_roi_w # hardware region of interest
        self.hw_roi_h = _hw_roi_h
        self.hw_roi_l = _hw_roi_l
        self.hw_roi_b = _hw_roi_b
        
    def set_reconstruct_helper( self, helper ):
        self._helper = helper

    def do_work(self, c_numarray._numarray framebuffer,
                double timestamp,
                int framenumber,
                int use_roi2):
        """find fly and orientation (fast enough for realtime use)

        inputs
        ------
        
        framebuffer
        timestamp
        framenumber
        use_roi2

        outputs
        -------
        
        [ (x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4) ]
        
        """
        cdef double x0, y0
        cdef double x0_abs, y0_abs, area, x0u, y0u, x1u, y1u
        cdef double orientation
        cdef double slope, eccentricity
        cdef double p1, p2, p3, p4
        cdef double eval1, eval2
        cdef double rise, run
        cdef double evalA, evalB
        cdef double evecA1, evecB1
        
        cdef double Mu00, Uu11, Uu02, Uu20
        cdef int i
        cdef int result, eigen_err
        
        cdef ipp.Ipp8u max_val
        
        cdef ipp.Ipp8u clear_despeckle_thresh
    
        cdef ipp.IppiSize roi2_sz
        cdef int left2, right2, bottom2, top2
        
        cdef int found_point

        self.last_image = framebuffer # useful when no IPP

        all_points_found = []
        
        # release GIL (SEE WARNING BELOW ABOUT RELEASING GIL)
        c_python.Py_BEGIN_ALLOW_THREADS
        # copy image to IPP memory
        for i from 0 <= i < self.hw_roi_h:
            c_lib.memcpy((self.im1 + self._bottom*self.im1_step + self._left)+self.im1_step*i,
                         framebuffer.data+self.hw_roi_w*i, # src
                         self.hw_roi_w)
        # do background subtraction
        CHK( ipp.ippiAbsDiff_8u_C1R(
            IMPOS8u(self.bg_img, self.bg_img_step, self._bottom, self._left), self.bg_img_step,
            IMPOS8u(self.im1,    self.im1_step,    self._bottom,self._left),  self.im1_step,
            IMPOS8u(self.im2,    self.im2_step,    self._bottom,self._left),  self.im2_step,
            self._roi_sz))
        c_python.Py_END_ALLOW_THREADS
        
        #work_done = False
        #while not work_done:
        while len(all_points_found) < MAX_NUM_POINTS:

            # release GIL
            c_python.Py_BEGIN_ALLOW_THREADS

            # WARNING WARNING WARNING WARNING WARNING WARNING WARNING

            # Everything from here to Py_END_ALLOW_THREADS must not make
            # calls to the Python C API.  The Python GIL (Global
            # Interpreter Lock) has been released, meaning that any calls
            # to the Python interpreter will have undefined effects,
            # because the interpreter is presumably in the middle of
            # another thread right now.

            # If you are not sure whether or not calls use the Python C
            # API, check the .c file generated by Pyrex.  Make sure even
            # function calls do not call the Python C API.
            
            # find max pixel in ROI
            CHK( ipp.ippiMaxIndx_8u_C1R(
                (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step,
                self._roi_sz, &max_val, &self.index_x,&self.index_y))

            if use_roi2:
                # find mini-ROI for further analysis (defined in non-ROI space)
                left2 = self.index_x - self._roi2_radius + self._left
                right2 = self.index_x + self._roi2_radius + self._left
                bottom2 = self.index_y - self._roi2_radius + self._bottom
                top2 = self.index_y + self._roi2_radius + self._bottom

                if left2 < self._left: left2 = self._left
                if right2 > self._right: right2 = self._right
                if bottom2 < self._bottom: bottom2 = self._bottom
                if top2 > self._top: top2 = self._top
            else:
                left2 = self._left
                right2 = self._right
                bottom2 = self._bottom
                top2 = self._top
            roi2_sz.width = right2 - left2 + 1
            roi2_sz.height = top2 - bottom2 + 1

            # (to reduce moment arm:) if pixel < self._clear_threshold*max(pixel): pixel=0

            clear_despeckle_thresh = <ipp.Ipp8u>(self._clear_threshold*max_val)
            if clear_despeckle_thresh < self._despeckle_threshold:
                clear_despeckle_thresh = self._despeckle_threshold

            CHK( ipp.ippiThreshold_Val_8u_C1IR(
                (self.im2 + bottom2*self.im2_step + left2), self.im2_step,
                roi2_sz, clear_despeckle_thresh, 0, ipp.ippCmpLess))

            found_point = 1
            if max_val < self._diff_threshold:
                x0=nan
                y0=nan
                x0_abs = nan
                y0_abs = nan
                found_point = 0 # c int (bool)
            else:
                result = fit_params( self.pState, &x0, &y0,
                                     &Mu00,
                                     &Uu11, &Uu20, &Uu02,
                                     roi2_sz.width, roi2_sz.height,
                                     (self.im2 + bottom2*self.im2_step + left2),
                                     self.im2_step )
                # note that x0 and y0 are now relative to the ROI origin
                if result == CFitParamsNoError:
                    area = Mu00
                    eigen_err = eigen_2x2_real( Uu20, Uu11,
                                                Uu11, Uu02,
                                                &evalA, &evecA1,
                                                &evalB, &evecB1)
                    if eigen_err:
                        slope = nan
                        orientation = nan
                        eccentricity = 0.0
                    else:
                        rise = 1.0 # 2nd component of eigenvectors will always be 1.0
                        if evalA > evalB:
                            run = evecA1
                            eccentricity = evalA/evalB
                        else:
                            run = evecB1
                            eccentricity = evalB/evalA
                        slope = rise/run

                        # This "orientation" is purely for J.B.
                        orientation = c_lib.atan2(rise,run)
                        orientation = orientation + 1.57079632679489661923 # (pi/2)
                elif result == CFitParamsZeroMomentError:
                    orientation = nan
                    x0 = nan
                    y0 = nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0
                elif result == CFitParamsCentralMomentError: orientation = nan
                else: SET_ERR(1)

                # set x0 and y0 relative to whole frame
                if found_point:
                    x0_abs = x0+left2
                    y0_abs = y0+bottom2

                if self._use_arena:
                    if self.arena_controller is not None:
                        self.arena_controller.arena_update(x0, y0, orientation,
                                                           timestamp, framenumber )
                    else:
                        SET_ERR(2)
            # grab GIL
            c_python.Py_END_ALLOW_THREADS

            if not found_point:
                #work_done = True
                #continue
                break

            if self._pmat_inv is not None:

                # (If we have self._pmat_inv, we can assume we have
                # self._helper.)

                undistort = self._helper.undistort # shorthand

                # calculate plane containing camera origin and found line
                # in 3D world coords

                # Step 1) Find world coordinates points defining plane:
                #    A) found point
                x0u, y0u = undistort( x0_abs, y0_abs )
                X0=nx.dot(self._pmat_inv,[x0u,y0u,1.0])
                #    B) another point on found line
                x1u, y1u = undistort(x0_abs+run,y0_abs+rise)
                X1=nx.dot(self._pmat_inv,[x1u,y1u,1.0])
                #    C) world coordinates of camera center already known

                # Step 2) Find world coordinates of plane
                svd = numarray.linear_algebra.singular_value_decomposition
                A = nx.array( [ X0, X1, self.camera_center] ) # 3 points define plane
                u,d,vt=svd(A,full_matrices=True)
                Pt = vt[3,:] # plane parameters

                x0_abs = x0u
                y0_abs = y0u

                p1,p2,p3,p4 = Pt[0:4]
                line_found = True
            else:
                p1,p2,p3,p4 = -1, -1, -1, -1 # sentinel value (will be converted to nan)
                line_found = False

            # prevent nan and inf from going across network
            if c_lib.isnan(slope):
                slope = 0.0
                slope_found = False
            else:
                slope_found = True

            if c_lib.isinf(eccentricity):
                eccentricity = near_inf
                
            if c_lib.isinf(slope):
                slope = near_inf
                
            all_points_found.append(
                (x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4, line_found, slope_found)
                )

            # clear roi2 for next iteration
            CHK( ipp.ippiSet_8u_C1R( 0, 
                (self.im2 + bottom2*self.im2_step + left2), self.im2_step,
                roi2_sz))
        return all_points_found

    def get_working_image(self):
        cdef c_numarray._numarray buf
        cdef int i

        buf = self.last_image # useful when no IPP
        
        # allocate new numarray memory
        buf = nx.zeros( (self.height, self.width), nx.UInt8 )

        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(buf.data+self.width*i,
                         self.im2+self.im2_step*i,
                         self.width)
        return buf
    
    def get_last_bright_point(self):
        return (self.index_x, self.index_y)

    def get_image(self,which='bg'):
        cdef c_numarray._numarray buf
        cdef int i
        cdef ipp.Ipp8u* im_base
        cdef int im_step

        if which=='bg':
            im_base = self.bg_img
            im_step = self.bg_img_step
        else:
            raise ValueError()

        buf = self.last_image # useful when no IPP
        
        # allocate new numarray memory
        buf = nx.zeros( (self.height, self.width), nx.UInt8 )

        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(buf.data+self.width*i,
                         im_base+im_step*i,
                         self.width)
        return buf

    def set_background_image(self, numbuf):
        cdef c_numarray._numarray buf
        cdef int i

        buf = <c_numarray._numarray>nx.array( numbuf )
        
        assert buf.shape[0] == self.height
        assert buf.shape[1] == self.width
        assert len(buf.shape) == 2
        assert buf.iscontiguous()
        
        # allocate new numarray memory
        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(self.bg_img+self.bg_img_step*i,
                         buf.data+self.width*i,
                         self.width)

    def take_background_image(self):
        CHK( ipp.ippiCopy_8u_C1R(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.bg_img + self._bottom*self.bg_img_step + self._left), self.bg_img_step,
            self._roi_sz))
        CHK( ipp.ippiConvert_8u32f_C1R(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.accum_image + self._bottom*self.accum_image_step/4 + self._left),
            self.accum_image_step,
            self._roi_sz))

    def clear_background_image(self):
        # start of IPP-requiring code
        CHK( ipp.ippiSet_8u_C1R( 0,
                                 (self.bg_img + self._bottom*self.bg_img_step + self._left),
                                 self.bg_img_step, self._roi_sz))
        CHK( ipp.ippiSet_32f_C1R( 0,
                                  (self.accum_image + self._bottom*self.accum_image_step + self._left),
                                  self.accum_image_step, self._roi_sz))

    def accumulate_last_image(self):
        CHK( ipp.ippiAddWeighted_8u32f_C1IR(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.accum_image + self._bottom*self.accum_image_step/4 + self._left),
            self.accum_image_step, self._roi_sz, self.alpha ))

        # maintain 8 bit unsigned background image
        CHK( ipp.ippiConvert_32f8u_C1R(
            (self.accum_image + self._bottom*self.accum_image_step/4 + self._left),
            self.accum_image_step,
            (self.bg_img + self._bottom*self.bg_img_step + self._left),
            self.bg_img_step, self._roi_sz, ipp.ippRndNear ))

    def rotation_calculation_init(self, int n_rot_samples):
        if self.arena_controller is not None:
            self.arena_controller.rotation_calculation_init( n_rot_samples )

    def rotation_update(self, float x0, float y0, float orientation, double timestamp):
        # convert back to ROI-relative coordinates
        if self.arena_controller is not None:
            x0 = x0 - self._left
            y0 = y0 - self._bottom
            self.arena_controller.rotation_update( x0, y0, orientation, timestamp )

    def rotation_end(self):
        cdef double new_x_cent, new_y_cent
        if self.arena_controller is not None:
            self.arena_controller.rotation_calculation_finish()
        
    property roi2_radius:
        def __get__(self):
            return self._roi2_radius
        def __set__(self,value):
            self._roi2_radius = value
        
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

    property pmat:
        def __get__(self):
            return self._pmat
        def __set__(self,c_numarray._numarray value):
            self._pmat = value

            P = self._pmat
            determinant = numarray.linear_algebra.determinant
            
            # find camera center in 3D world coordinates
            X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
            Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
            Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
            T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

            self.camera_center = nx.array( [ X/T, Y/T, Z/T, 1.0 ] )
            self._pmat_inv = numarray.linear_algebra.generalized_inverse(self._pmat)

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
