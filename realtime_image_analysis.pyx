#emacs, this is -*-Python-*- mode
# $Id$

import time
import sys
import numarray as nx
import numarray.ieeespecial
import numarray.linear_algebra

cdef double nan
nan = numarray.ieeespecial.nan

cimport c_numarray

cimport c_lib
cimport c_python

# start of IPP-requiring code
cimport ipp
cimport ArenaController
import ArenaController

cdef extern from "c_fit_params.h":
    int fit_params( ipp.IppiMomentState_64f *pState, double *x0, double *y0,
                    double *Mu00,
                    double *Uu11, double *Uu20, double *Uu02,
                    int width, int height, unsigned char *img, int img_step )

cdef extern from "eigen.h":
    int eigen_2x2_real( double A, double B, double C, double D,
                        double *evalA, double *evecA1,
                        double *evalB, double *evecB1 )


class IPPError(Exception):
    pass

cdef void CHK( ipp.IppStatus errval ):
    # This is rather crude at the moment because calls to the Python C
    # API cannot be made.  (This code is executed when we have
    # released the GIL.)
    if errval != 0:
        c_lib.printf("ERROR on CHK!\n")
        c_lib.exit(1)

cdef void SET_ERR( int errval ):
    # This is rather crude at the moment because calls to the Python C
    # API cannot be made.  (This code is executed when we have
    # released the GIL.)
    c_lib.printf("SET_ERR called!\n")
    c_lib.exit(2)

# end of IPP-requiring code

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

    cdef int roi2_radius
    
    # runtime parameters
    cdef float _diff_threshold
    cdef float _clear_threshold
    
    cdef int _use_arena
    
    cdef c_numarray._numarray last_image

    # calibration matrix
    cdef c_numarray._numarray _pmat, _pmat_inv, camera_center
    cdef object _helper

    # start of IPP-requiring code
    cdef ipp.IppiSize _roi_sz
        
    cdef int im1_step, im2_step, accum_image_step, bg_img_step
    cdef int n_rot_samples

    cdef ipp.Ipp8u *im1, *im2 # current image
    cdef ipp.Ipp8u *bg_img # 8-bit background
    cdef ipp.Ipp32f *accum_image # FP accumulator

    cdef ipp.IppiMomentState_64f *pState

    cdef ArenaController.ArenaController arena_controller
    # end of IPP-requiring code

    def __init__(self, int w, int h, float alpha):
        self.width = w
        self.height = h
        self.alpha = alpha
        self.roi = ( 0, 0, self.width-1, self.height-1)


        self.roi2_radius = 15
        self._diff_threshold = 8.1
        self._clear_threshold = 0.0
        self._use_arena = 0
        
        self._pmat = None
        self._pmat_inv = None

        self._helper = None

        # start of IPP-requiring code
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
        # end of IPP-requiring code

    def __dealloc__(self):
        # start of IPP-requiring code
        CHK( ipp.ippiMomentFree_64f( self.pState ))
        ipp.ippiFree(self.im1)
        ipp.ippiFree(self.im2)
        ipp.ippiFree(self.bg_img)
        ipp.ippiFree(self.accum_image)
        # end of IPP-requiring code
        return

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
        found_anything
        orientation
        
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
        
        # start of IPP-requiring code
        cdef ipp.Ipp8u max_val
        cdef int index_x,index_y

        
        cdef ipp.IppiSize roi2_sz
        cdef int left2, right2, bottom2, top2
        
        # end of IPP-requiring code
        cdef int found_point
        
        self.last_image = framebuffer # useful when no IPP
        
        x0,y0=320,240
        slope = 1
        eccentricity = 0
        
        # start of IPP-requiring code

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

        # copy image to IPP memory
        for i from 0 <= i < self.height:
            c_lib.memcpy(self.im1+self.im1_step*i, # dest
                         framebuffer.data+self.width*i, # src
                         self.width) # length

        # do background subtraction & find max pixel in ROI
        CHK( ipp.ippiAbsDiff_8u_C1R(
            (self.bg_img + self._bottom*self.bg_img_step + self._left), self.bg_img_step,
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step, self._roi_sz))
        CHK( ipp.ippiMaxIndx_8u_C1R(
            (self.im2 + self._bottom*self.im2_step + self._left), self.im2_step,
            self._roi_sz, &max_val, &index_x,&index_y))

        if use_roi2:
            # find mini-ROI for further analysis (defined in non-ROI space)
            left2 = index_x - self.roi2_radius + self._left
            right2 = index_x + self.roi2_radius + self._left
            bottom2 = index_y - self.roi2_radius + self._bottom
            top2 = index_y + self.roi2_radius + self._bottom

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
        CHK( ipp.ippiThreshold_Val_8u_C1IR(
            (self.im2 + bottom2*self.im2_step + left2), self.im2_step,
            roi2_sz, self._clear_threshold*max_val, 0, ipp.ippCmpLess))

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
            if result == 0:
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

                    # John -- I don't use "orientation" -- fix this however you need it.
                    orientation = c_lib.atan2(rise,run)
                    orientation = orientation + 1.57079632679489661923 # (pi/2)
            elif result == 31: orientation = nan
            elif result == 32: orientation = nan
            elif result == 33: orientation = nan
            else: SET_ERR(1)
                
            # set x0 and y0 relative to whole frame
            if x0 == -1 and y0 == -1:
                x0 = nan
                y0 = nan
                x0_abs = nan
                y0_abs = nan
                found_point = 0
            else:
                x0_abs = x0+left2
                y0_abs = y0+bottom2

            if self._use_arena:
                if self.arena_controller is not None:
                    self.arena_controller.arena_update(x0, y0, orientation,
                                                       timestamp, framenumber )
                    # JB: should maybe set found_point=0 here to avoid doing SVD?
                    # uncertain how this would affect camera server
                else:
                    SET_ERR(2)
        # grab GIL
        c_python.Py_END_ALLOW_THREADS

        if self._pmat_inv is not None and found_point:

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
        else:
            p1,p2,p3,p4 = nan, nan, nan, nan

        # end of IPP-requiring code

        if not c_lib.isnan(x0_abs):
            found_anything = True
        else:
            found_anything = False
        return [ (x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4) ], found_anything, orientation # XXX orientation is hack

    def get_working_image(self):
        cdef c_numarray._numarray buf
        cdef int i

        buf = self.last_image # useful when no IPP
        
        # start of IPP-requiring code
        # allocate new numarray memory
        buf = nx.zeros( (self.height, self.width), nx.UInt8 )

        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(buf.data+self.width*i,
                         self.im2+self.im2_step*i,
                         self.width)
        # end of IPP-requiring code
        return buf

    def get_background_image(self):
        cdef c_numarray._numarray buf
        cdef int i

        buf = self.last_image # useful when no IPP
        
        # start of IPP-requiring code
        # allocate new numarray memory
        buf = nx.zeros( (self.height, self.width), nx.UInt8 )

        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(buf.data+self.width*i,
                         self.bg_img+self.bg_img_step*i,
                         self.width)
        # end of IPP-requiring code
        return buf

    def set_background_image(self, numbuf):
        cdef c_numarray._numarray buf
        cdef int i

        buf = <c_numarray._numarray>nx.array( numbuf )
        
        assert buf.shape[0] == self.height
        assert buf.shape[1] == self.width
        assert len(buf.shape) == 2
        assert buf.iscontiguous()
        
        # start of IPP-requiring code
        # allocate new numarray memory
        # copy image to numarray
        for i from 0 <= i < self.height:
            c_lib.memcpy(self.bg_img+self.bg_img_step*i,
                         buf.data+self.width*i,
                         self.width)
        # end of IPP-requiring code

    def take_background_image(self):
        # start of IPP-requiring code
        CHK( ipp.ippiCopy_8u_C1R(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.bg_img + self._bottom*self.bg_img_step + self._left), self.bg_img_step,
            self._roi_sz))
        CHK( ipp.ippiConvert_8u32f_C1R(
            (self.im1 + self._bottom*self.im1_step + self._left), self.im1_step,
            (self.accum_image + self._bottom*self.accum_image_step/4 + self._left),
            self.accum_image_step,
            self._roi_sz))
        # end of IPP-requiring code
        return

    def clear_background_image(self):
        # start of IPP-requiring code
        CHK( ipp.ippiSet_8u_C1R( 0,
                                 (self.bg_img + self._bottom*self.bg_img_step + self._left),
                                 self.bg_img_step, self._roi_sz))
        CHK( ipp.ippiSet_32f_C1R( 0,
                                  (self.accum_image + self._bottom*self.accum_image_step + self._left),
                                  self.accum_image_step, self._roi_sz))
        # end of IPP-requiring code
        return

    def accumulate_last_image(self):
        # start of IPP-requiring code
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
        
        # end of IPP-requiring code
        return

    def rotation_calculation_init(self, int n_rot_samples):
        # start of IPP-requiring code
        if self.arena_controller is not None:
            self.arena_controller.start_center_calculation( n_rot_samples )
            self.arena_controller.rotation_calculation_init()
        # end of IPP-requiring code
        return

    def rotation_update(self, float x0, float y0, float orientation):
        # start of IPP-requiring code
        # convert back to ROI-relative coordinates
        if self.arena_controller is not None:
            x0 = x0 - self._left
            y0 = y0 - self._bottom
            self.arena_controller.update_center_calculation( x0, y0, orientation )
            self.arena_controller.arena_control.rotation_update()
        # end of IPP-requiring code
        return

    def rotation_end(self):
        # start of IPP-requiring code
        cdef double new_x_cent, new_y_cent
        if self.arena_controller is not None:
            self.arena_controller.end_center_calculation( &new_x_cent, &new_y_cent )
            self.arena_controller.arena_control.rotation_calculation_finish( new_x_cent, new_y_cent )
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

            # start of IPP-requiring code
            self._roi_sz.width = self._right-self._left+1
            self._roi_sz.height = self._top-self._bottom+1
            # end of IPP-requiring code
