#emacs, this is -*-Python-*- mode
# $Id: $

import numarray as nx
import numarray.ieeespecial
import numarray.linear_algebra

cimport FastImage
import FastImage
import weakref

# Define these classes to allow weak refs.

# (It's good to keep these here because then we don't pollute
# FastImage itself with such a hack.

class wrFastImage8u(FastImage.FastImage8u):
    __slots__ = ['__weakref__']

class wrFastImage32f(FastImage.FastImage32f):
    __slots__ = ['__weakref__']

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

cdef extern from "unistd.h":
    ctypedef long intptr_t
        
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
    
class IppError(Exception):
    def __init__(self, int errval):
        cdef char* cmsg
        cmsg = ipp.ippCoreGetStatusString(errval)
        #Exception.__init__(self,"Error %d: %s"%(errval,cmsg))
        Exception.__init__(self,cmsg)

cdef void CHK_HAVEGIL( ipp.IppStatus errval ) except *:
    if (errval):
        raise IppError(errval)

cdef void SET_ERR( int errval ):
    # This is rather crude at the moment because calls to the Python C
    # API cannot be made.  (This code is executed when we have
    # released the GIL.)
    c_lib.printf("SET_ERR called! (May not have GIL, cannot raise exception.)\n")
    c_lib.exit(2)

cdef print_8u_arr(ipp.Ipp8u* src,int width,int height,int src_step):
  cdef int row, col
  cdef ipp.Ipp8u* src_rowstart
  for row from 0 <= row < height:
    src_rowstart = src+(row*src_step);
    for col from 0 <= col < width:
      print "%d"%src_rowstart[col],
    print
  print
    
cdef class RealtimeAnalyzer:

    # full image size
    cdef int maxwidth, maxheight

    # number of images in accumulator
    cdef float alpha
    
    # ROI size
    cdef int _left, _bottom, _right, _top

    cdef int _roi2_radius
    
    # runtime parameters
    cdef float _diff_threshold
    cdef float _clear_threshold
    
    cdef int _use_arena
    
    # calibration matrix
    cdef c_numarray._numarray _pmat, _pmat_inv, camera_center
    cdef object _helper

    cdef ipp.Ipp8u _despeckle_threshold
    
    cdef FastImage.Size _roi_sz
        
    cdef int n_rot_samples

    # class A images:

    # Portions of these images may be out of date. This is done
    # because we only want to malloc each image once, despite changing
    # input image size. The roi is a view of the active part. The roi2
    # is a view of the sub-region of the active part.
    
    cdef FastImage.FastImage8u absdiff_im, cmpdiff_im # also raw_im

    cdef FastImage.FastImage8u absdiff_im_roi_view
    cdef FastImage.FastImage8u cmpdiff_im_roi_view
    cdef FastImage.FastImage8u absdiff_im_roi2_view
    cdef FastImage.FastImage8u cmpdiff_im_roi2_view

    # class B images:

    # These entirety of these images are always valid and have an roi
    # into the active region.

    cdef FastImage.FastImage8u mean_im, cmp_im

    cdef FastImage.FastImage8u mean_im_roi_view, cmp_im_roi_view
    
    cdef ipp.IppiMomentState_64f *pState

    cdef ArenaController.ArenaController arena_controller

    cdef object imname2im

    def __new__(self,*args,**kw):
        # image moment calculation initialization
        CHK( ipp.ippiMomentInitAlloc_64f( &self.pState, ipp.ippAlgHintFast ) )
        try:
            self.arena_controller = ArenaController.ArenaController()
        except Exception, exc:
            print 'WARNING: could not create ArenaController:',exc.__class__,str(exc)
            self.arena_controller = None
            
    def __dealloc__(self):
        CHK_HAVEGIL( ipp.ippiMomentFree_64f( self.pState ))
#        del self.arena_controller

    def __init__(self,int maxwidth, int maxheight, float alpha=0.1):
        # software analysis ROI
        self.maxwidth = maxwidth
        self.maxheight = maxheight

        self.alpha = alpha

        self._roi2_radius = 10
        self._diff_threshold = 11
        self._clear_threshold = 0.2
        self._use_arena = 0
        
        self._pmat = None
        self._pmat_inv = None

        self._helper = None

        self._despeckle_threshold = 5
        self.n_rot_samples = 100*60 # 1 minute

        sz = FastImage.Size(self.maxwidth,self.maxheight)
        # 8u images
        self.absdiff_im=wrFastImage8u(sz)

        # 8u background
        self.mean_im=wrFastImage8u(sz)
        self.cmp_im=wrFastImage8u(sz)
        self.cmpdiff_im=wrFastImage8u(sz)
        
        self.update_imname2im() # create and update self.imname2im dict

        self.roi = 0,0,maxwidth-1,maxheight-1 # sets self._roi_sz to l,b,r,t, assigns roi_views
        
        # initialize background images
        self.mean_im_roi_view.set_val( 0, self._roi_sz )

    def set_reconstruct_helper( self, helper ):
        self._helper = helper

    def do_work(self,
                FastImage.FastImage8u raw_im_small,
                double timestamp,
                int framenumber,
                int use_roi2,
                int use_cmp=0,
                int return_first_xy=0
                ):
        """find fly and orientation (fast enough for realtime use)

        inputs
        ------
        
        timestamp
        framenumber
        use_roi2

        optional inputs (default to 0)
        ------------------------------
        use_cmp -- perform more detailed analysis against cmp image (used with ongoing variance estimation)
        return_first_xy -- for debugging

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
        
        cdef int index_x,index_y
        
        cdef ipp.Ipp8u max_val
        cdef ipp.Ipp8u* max_val_ptr
        cdef ipp.Ipp8u max_std_diff
        
        cdef ipp.Ipp8u clear_despeckle_thresh

        cdef FastImage.Size roi2_sz
        cdef int left2, right2, bottom2, top2
        
        cdef int found_point

        all_points_found = []

        # This is our near-hack to ensure users update .roi before calling .do_work()
        if not ((self._roi_sz.sz.width == raw_im_small.imsiz.sz.width) and
                (self._roi_sz.sz.height == raw_im_small.imsiz.sz.height)):
            raise ValueError("input image size does not correspond to ROI "
                             "(set RealtimeAnalyzer.roi before calling)")

        # find difference from mean
        #c_python.Py_BEGIN_ALLOW_THREADS
        # absdiff_im = | raw_im - mean_im |
        raw_im_small.fast_get_absdiff_put( self.mean_im_roi_view,
                                     self.absdiff_im_roi_view,
                                     self._roi_sz)
        if use_cmp:
            # cmpdiff_im = absdiff_im - cmp_im (saturates 8u)
            self.absdiff_im_roi_view.fast_get_sub_put( self.cmp_im_roi_view,
                                                       self.cmpdiff_im_roi_view,
                                                       self._roi_sz )
        #c_python.Py_END_ALLOW_THREADS

        while len(all_points_found) < MAX_NUM_POINTS:

            # release GIL
            #c_python.Py_BEGIN_ALLOW_THREADS

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
            if use_cmp:
                CHK( ipp.ippiMaxIndx_8u_C1R(
                    <ipp.Ipp8u*>self.cmpdiff_im_roi_view.im,self.cmpdiff_im_roi_view.step,
                    self._roi_sz.sz, &max_std_diff, &index_x, &index_y))
                max_val_ptr = (<ipp.Ipp8u*>self.absdiff_im_roi_view.im)+self.absdiff_im_roi_view.step*index_y+index_x
                max_val = max_val_ptr[0] # value at maximum difference from std
            else:
                CHK( ipp.ippiMaxIndx_8u_C1R(
                    <ipp.Ipp8u*>self.absdiff_im_roi_view.im,self.absdiff_im_roi_view.step,
                    self._roi_sz.sz, &max_val, &index_x, &index_y))

            if use_roi2:
                # find mini-ROI for further analysis (defined in non-ROI space)
                left2 = index_x - self._roi2_radius + self._left
                right2 = index_x + self._roi2_radius + self._left
                bottom2 = index_y - self._roi2_radius + self._bottom
                top2 = index_y + self._roi2_radius + self._bottom

                if left2 < self._left: left2 = self._left
                if right2 > self._right: right2 = self._right
                if bottom2 < self._bottom: bottom2 = self._bottom
                if top2 > self._top: top2 = self._top
                roi2_sz = FastImage.Size(right2 - left2 + 1, top2 - bottom2 + 1)
            else:
                left2 = self._left
                right2 = self._right
                bottom2 = self._bottom
                top2 = self._top
                roi2_sz = self._roi_sz

            self.absdiff_im_roi2_view = self.absdiff_im.roi(left2,bottom2,roi2_sz)
            if use_cmp:
                self.cmpdiff_im_roi2_view = self.cmpdiff_im.roi(left2,bottom2,roi2_sz)

            # (to reduce moment arm:) if pixel < self._clear_threshold*max(pixel): pixel=0

            clear_despeckle_thresh = <ipp.Ipp8u>(self._clear_threshold*max_val)
            if clear_despeckle_thresh < self._despeckle_threshold:
                clear_despeckle_thresh = self._despeckle_threshold

            CHK( ipp.ippiThreshold_Val_8u_C1IR(
                <ipp.Ipp8u*>self.absdiff_im_roi2_view.im,self.absdiff_im_roi2_view.step,
                roi2_sz.sz, clear_despeckle_thresh, 0, ipp.ippCmpLess))

            found_point = 1

            if not use_cmp:
                if max_val < self._diff_threshold:
                    x0=nan
                    y0=nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0 # c int (bool)
            else:
                if max_std_diff == 0:
                    x0=nan
                    y0=nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0 # c int (bool)
            if found_point:
                result = fit_params( self.pState, &x0, &y0,
                                     &Mu00,
                                     &Uu11, &Uu20, &Uu02,
                                     roi2_sz.sz.width, roi2_sz.sz.height,
                                     <unsigned char*>self.absdiff_im_roi2_view.im,
                                     self.absdiff_im_roi2_view.step)
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
            #c_python.Py_END_ALLOW_THREADS

            if return_first_xy:
                #nominal: (x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4, line_found, slope_found)
                rval = [(index_x, index_y, max_std_diff, 0.0, eccentricity, p1, p2, p3, p4, 0, 0)]
                return rval

            if not found_point:
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
            CHK( ipp.ippiSet_8u_C1R(
                0, 
                <ipp.Ipp8u*>self.absdiff_im_roi2_view.im,
                self.absdiff_im_roi2_view.step,
                roi2_sz.sz))
            if use_cmp:
                CHK( ipp.ippiSet_8u_C1R(
                    0,
                    <ipp.Ipp8u*>self.cmpdiff_im_roi2_view.im,self.cmpdiff_im_roi2_view.step,
                    roi2_sz.sz))
            
        return all_points_found

##    def get_image_copy(self,which='mean'):
##        if which=='mean':
##            return nx.array(self.mean_im)
##        elif which=='absdiff':
##            return nx.array(self.absdiff_im)
##        else:
##            raise ValueError()

    def update_imname2im(self):
        """refresh weak references"""
        self.imname2im = {'absdiff' :weakref.ref(self.absdiff_im),
                          'mean'    :weakref.ref(self.mean_im),
                          'cmp'     :weakref.ref(self.cmp_im),
                          'cmpdiff' :weakref.ref(self.cmpdiff_im),
                          }
    
    def get_image_view(self,which='mean'):
        im = self.imname2im[which]()
        if im is None:
            print 'reference to image changed' # are we mallocing and freeing too much?
            # weak reference allowed original source to be deallocated
            self.update_imname2im()
            im = self.imname2im[which]()
        return im
    
##    def take_background_image(self):
####        c_python.Py_BEGIN_ALLOW_THREADS
##        CHK( ipp.ippiCopy_8u_C1R(
##            <ipp.Ipp8u*>self.raw_im_roi_view.im,self.raw_im_roi_view.step,
##            <ipp.Ipp8u*>self.mean_im_roi_view.im,self.mean_im_roi_view.step,
##            self._roi_sz.sz))
##        CHK( ipp.ippiConvert_8u32f_C1R(
##            <ipp.Ipp8u*>self.raw_im_roi_view.im,self.raw_im_roi_view.step,
##            <ipp.Ipp32f*>self.accum_im_roi_view.im,self.accum_im_roi_view.step,
##            self._roi_sz.sz))
####        c_python.Py_END_ALLOW_THREADS

##    def accumulate_last_image(self):
####        c_python.Py_BEGIN_ALLOW_THREADS
##        CHK( ipp.ippiAddWeighted_8u32f_C1IR(
##            <ipp.Ipp8u*>self.raw_im_roi_view.im,self.raw_im_roi_view.step,
##            <ipp.Ipp32f*>self.accum_im_roi_view.im,self.accum_im_roi_view.step,
##            self._roi_sz.sz, self.alpha ))

##        # maintain 8 bit unsigned background image
##        CHK( ipp.ippiConvert_32f8u_C1R(
##            <ipp.Ipp32f*>self.accum_im_roi_view.im,self.accum_im_roi_view.step,
##            <ipp.Ipp8u*>self.mean_im_roi_view.im,self.mean_im_roi_view.step,
##            self._roi_sz.sz, ipp.ippRndNear ))
####        c_python.Py_END_ALLOW_THREADS

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
            assert self._right < self.maxwidth
            assert self._top < self.maxheight

            self._roi_sz = FastImage.Size( self._right-self._left+1, self._top-self._bottom+1 )
            
            self.absdiff_im_roi_view = self.absdiff_im.roi(self._left,self._bottom,self._roi_sz)
            self.mean_im_roi_view = self.mean_im.roi(self._left,self._bottom,self._roi_sz)
            self.cmp_im_roi_view = self.cmp_im.roi(self._left,self._bottom,self._roi_sz)
            self.cmpdiff_im_roi_view = self.cmpdiff_im.roi(self._left,self._bottom,self._roi_sz)
