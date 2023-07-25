#emacs, this is -*-Python-*- mode
#cython: language_level=2

import numpy
inf = numpy.inf

cdef double cinf
cinf = inf


import flydra_core.common_variables
import flydra_core.undistort
import xml.etree.ElementTree as ET
import warnings
import os

from libc.math cimport sqrt

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)
    bint npy_isinf(double x)

cdef float MINIMUM_ECCENTRICITY
MINIMUM_ECCENTRICITY = flydra_core.common_variables.MINIMUM_ECCENTRICITY

STRICT_WATER = int(os.environ.get('STRICT_WATER_HYPOTHESIS_TEST','0'))

class NoAcceptablePointFound(Exception):
    pass

def make_ReconstructHelper_from_rad_file(filename):
    params = {}
    exec(open(filename).read(),params)
    if params['K12'] != 0:
        raise NotImplementedError('need to properly roundtrip alpha_c != 0')
    helper = ReconstructHelper(
        params['K11'], params['K22'], params['K13'], params['K23'],
        params['kc1'], params['kc2'], params['kc3'], params['kc4'])
    return helper

def ReconstructHelper_from_xml(elem):
    assert ET.iselement(elem)
    assert elem.tag == "non_linear_parameters"

    args = []
    for name in ['fc1','fc2','cc1','cc2','k1','k2','p1','p2','alpha_c']:
        args.append( float(elem.find(name).text) )
    args = tuple(args)
    helper = ReconstructHelper(*args)
    return helper

def make_ReconstructHelper(*args,**kw):
    return ReconstructHelper(*args,**kw)

cdef class ReconstructHelper:
    cdef readonly double fc1, fc2, cc1, cc2
    cdef readonly double fc1p, fc2p, cc1p, cc2p
    cdef readonly double k1, k2, k3, p1, p2
    cdef readonly double alpha_c
    cdef readonly int simple

    def __init__(self, fc1, fc2, cc1, cc2, k1, k2, p1, p2, alpha_c=0,
                 fc1p=None, fc2p=None, cc1p=None, cc2p=None, k3=None,
                 ):
        """create instance of ReconstructHelper

        ReconstructHelper(fc1, fc2, cc1, cc2, k1, k2, p1, p2 )
        where:
        fc - focal length
        cc - camera center
        k - radial distortion parameters (non-linear)
        p - tangential distortion parameters (non-linear)
        alpha_c - skew between X and Y pixel axes
        """
        self.fc1 = self.fc1p = fc1
        self.fc2 = self.fc2p = fc2
        self.cc1 = self.cc1p = cc1
        self.cc2 = self.cc2p = cc2
        self.k1 = k1
        self.k2 = k2
        if k3 is None:
            k3 = 0.0
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.alpha_c = alpha_c

        # "simple" means that the forward and back normalizations from
        # pixel coords to normalized coords are identical.

        self.simple=1
        if fc1p is not None and fc1p!=fc1:
            self.simple = 0
            self.fc1p = fc1p
        if fc2p is not None and fc2p!=fc2:
            self.simple = 0
            self.fc2p = fc2p
        if cc1p is not None and cc1p!=cc1:
            self.simple = 0
            self.cc1p = cc1p
        if cc2p is not None and cc2p!=cc2:
            self.simple = 0
            self.cc2p = cc2p
        if not self.simple and self.alpha_c != 0:
            raise NotImplementedError('skew not tested with non-simple distortion model')

    def __reduce__(self):
        """this allows ReconstructHelper to be pickled"""
        args = (self.fc1, self.fc2, self.cc1, self.cc2,
                self.k1, self.k2, self.p1, self.p2, self.alpha_c,
                self.fc1p, self.fc2p, self.cc1p, self.cc2p, self.k3)
        return (make_ReconstructHelper, args)

    def add_element(self, parent):
        """add self as XML element to parent"""
        assert ET.iselement(parent)
        elem = ET.SubElement(parent,"non_linear_parameters")

        if self.k3 != 0.0:
            raise NotImplementedError('k3 != 0.0')

        for name in ['fc1','fc2','cc1','cc2','k1','k2','p1','p2','alpha_c',
                     'fc1p', 'fc2p', 'cc1p', 'cc2p',
                     ]:
            e = ET.SubElement(elem, name)
            e.text = repr( getattr(self,name) )

    def as_obj_for_json(self):
        result = {}
        for name in ['fc1','fc2','cc1','cc2','k1','k2','k3','p1','p2','alpha_c',
                     'fc1p', 'fc2p', 'cc1p', 'cc2p']:
            result[name] = getattr(self,name)
        return result

    def save_to_rad_file( self, fname, comments=None ):
        if not self.simple:
            raise ValueError('cannot encode non-simple intrinsic model as a .rad file')
        rad_fd = open(fname,'w')
        K = self.get_K()
        nlparams = self.get_nlparams()
        k1, k2, p1, p2, k3 = nlparams
        rad_fd.write('K11 = %s;\n'%repr(K[0,0]))
        rad_fd.write('K12 = %s;\n'%repr(K[0,1]))
        rad_fd.write('K13 = %s;\n'%repr(K[0,2]))
        rad_fd.write('K21 = %s;\n'%repr(K[1,0]))
        rad_fd.write('K22 = %s;\n'%repr(K[1,1]))
        rad_fd.write('K23 = %s;\n'%repr(K[1,2]))
        rad_fd.write('K31 = %s;\n'%repr(K[2,0]))
        rad_fd.write('K32 = %s;\n'%repr(K[2,1]))
        rad_fd.write('K33 = %s;\n'%repr(K[2,2]))
        rad_fd.write('\n')
        rad_fd.write('kc1 = %s;\n'%repr(k1))
        rad_fd.write('kc2 = %s;\n'%repr(k2))
        rad_fd.write('kc3 = %s;\n'%repr(p1))
        rad_fd.write('kc4 = %s;\n'%repr(p2))
        rad_fd.write('kc5 = %s;\n'%repr(k3))
        rad_fd.write('\n')
        if comments is not None:
            comments = str(comments)
        rad_fd.write("comments = '%s';\n"%comments)
        rad_fd.write('\n')
        rad_fd.close()

    def __richcmp__(self,other,op):
        # cmp op
        #  < 0
        # <= 1
        # == 2
        # != 3
        #  > 4
        # >= 5

        if isinstance(other, ReconstructHelper):
            isequal = (numpy.allclose(self.get_K(),other.get_K()) and
                       numpy.allclose(self.get_Kp(),other.get_Kp()) and
                       numpy.allclose(self.get_nlparams(),other.get_nlparams()) )
        else:
            isequal = False

        if op in [1,2,5]:
            result = isequal
        elif op == 3:
            result = not isequal
        else:
            result = False
        return result

    def get_K(self):
        # See
        # http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        K = numpy.array((( self.fc1, self.alpha_c*self.fc1, self.cc1),
                         ( 0,        self.fc2,              self.cc2),
                         ( 0,        0,                     1       )))
        return K

    def get_Kp(self):
        K = numpy.array((( self.fc1p, 0,                     self.cc1p),
                         ( 0,         self.fc2p,             self.cc2p),
                         ( 0,         0,                     1       )))
        return K

    def get_nlparams(self):
        return (self.k1, self.k2, self.p1, self.p2, self.k3)

    def undistort(self, double x_kk, double y_kk, int n_iterations=5):
        """undistort 2D coordinate pair

        Iteratively performs an undistortion using camera intrinsic
        parameters.

        Implementation translated from CalTechCal.

        See also the OpenCV reference manual, which has the equation
        used.
        """

        cdef double xl, yl

        cdef double xd, yd, x, y
        cdef double r_2, k_radial, delta_x, delta_y
        cdef int i

        # undoradial.m / CalTechCal/normalize.m

        xd = ( x_kk - self.cc1 ) / self.fc1
        yd = ( y_kk - self.cc2 ) / self.fc2

        xd = xd - self.alpha_c * yd

        # comp_distortion_oulu.m

        # initial guess
        x = xd
        y = yd

        for i from 0<=i<n_iterations:
            r_2 = x*x + y*y
            k_radial = 1.0 + r_2*(self.k1 + r_2*(self.k2 + r_2*self.k3))
            delta_x = 2.0 * (self.p1)*x*y + (self.p2)*(r_2 + 2.0*x*x)
            delta_y = (self.p1) * (r_2 + 2.0*y*y)+2.0*(self.p2)*x*y
            x = (xd-delta_x)/k_radial
            y = (yd-delta_y)/k_radial

        # undoradial.m

        xl = (self.fc1p)*x + (self.fc1p*self.alpha_c)*y + (self.cc1p)
        yl = (self.fc2p)*y + (self.cc2p)
        return (xl, yl)

    def distort(self, double xl, double yl):
        """distort 2D coordinate pair"""

        cdef double x, y, r_2, term1, xd, yd

        assert self.alpha_c==0 # there's no way the following code works otherwise

        x = ( xl - self.cc1p ) / self.fc1p
        y = ( yl - self.cc2p ) / self.fc2p

        r_2 = x*x + y*y
        r_4 = r_2**2
        r_6 = r_2*r_4
        term1 = self.k1*r_2 + self.k2*r_4 + self.k3*r_6

        # OpenCV manual (chaper 6, "3D reconstruction", "camera
        # calibration" section) seems to disagree with
        # http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html

        # Furthermore, implementations in cvundistort.cpp seem still
        # unrelated.  Finally, project_points2.m in Bouguet's code is
        # consistent with his webpage and this below.

        xd = x + x*term1 + (2*self.p1*x*y + self.p2*(r_2+2*x**2))
        yd = y + y*term1 + (self.p1*(r_2+2*y**2) + 2*self.p2*x*y)

        xd = (self.fc1)*xd + (self.cc1)
        yd = (self.fc2)*yd + (self.cc2)

        return (xd, yd)

    def undistort_image(self, numpy_image ):
        assert len(numpy_image.shape)==2
        assert numpy_image.dtype == numpy.uint8

        f = self.fc1, self.fc2 # focal length
        c = self.cc1, self.cc2 # center
        k = self.k1, self.k2, self.p1, self.p2, self.k3  # NL terms: r^2, r^4, tan1, tan2, r^6

        assert self.simple
        assert self.alpha_c==0

        imnew = flydra_core.undistort.rect(numpy_image, f=f, c=c, k=k) # perform the undistortion
        imnew = imnew.astype(numpy.uint8)
        return imnew

cdef int gave_water_warning
gave_water_warning = 0

def hypothesis_testing_algorithm__find_best_3d( object recon, object found_data_dict,
                                                double ACCEPTABLE_DISTANCE_PIXELS,
                                                int debug=0, Py_ssize_t max_n_cams=5,
                                                int with_water = 0):
    """Use hypothesis testing algorithm to find best 3D point

    Finds combination of cameras which uses the most number of cameras
    while minimizing mean reprojection error. Algorithm used accepts
    any camera combination with reprojection error less than the
    variable ACCEPTABLE_DISTANCE_PIXELS.

    """
    cdef int n_cams, best_n_cams, alloc_n_cams
    cdef int i
    cdef int missing_cam_data
    cdef double alpha
    cdef double x, y, orig_x, orig_y, new_x, new_y
    cdef double dist, mean_dist, least_err

    #cdef double *least_err_by_n_cameras # fake dict (index = key)

    svd = numpy.linalg.svd # eliminate global name lookup

    cam_ids = recon.cam_ids # shorthand
    max_n_cams = min(len(cam_ids), max_n_cams)

    # Initialize least_err_by_n_cameras to be infinity.  Note that
    # values at 0th and 1st index will always remain infinity.

    least_err_by_n_cameras = [cinf]*(max_n_cams+1) # allow 0-based indexing to last camera
    allA = numpy.zeros( (2*(len(cam_ids)+1),4), dtype=numpy.float64)
    bad_cam_ids = []
    cam_id2idx = {}
    all2d = {}
    Pmat_fastnx = recon.Pmat # shorthand

    global gave_water_warning
    if with_water:
        if STRICT_WATER:
            raise NotImplementedError('water and hypothesis testing not yet implemented')
        if not gave_water_warning:
            warnings.warn('_reconstruct_utils: Hypothesis test intersection done '
                          'without refraction correction. Result will be wrong. '
                          'Set environment variable STRICT_WATER_HYPOTHESIS_TEST '
                          'to raise an error rather than give this warning.')
            gave_water_warning = 1
    for i,cam_id in enumerate(cam_ids):
        cam_id2idx[cam_id] = i

        # do we have incoming data?
        try:
            value_tuple = found_data_dict[cam_id]
        except KeyError:
            bad_cam_ids.append( cam_id )
            continue # don't build this row

        # was a 2d point found?
        xy_values = value_tuple[:2]
        x,y = xy_values
        if npy_isnan(x):
            bad_cam_ids.append( cam_id )
            continue # don't build this row

        # Similar to code in reconstruct.Reconstructor.find3d()

        Pmat = Pmat_fastnx[cam_id] # Pmat is 3 rows x 4 columns
        row3 = Pmat[2,:]
        allA[ i*2, : ] = x*row3 - Pmat[0,:]
        allA[ i*2+1, :]= y*row3 - Pmat[1,:]

        all2d[cam_id] = xy_values

    cam_ids_for_least_err = {}
    X_for_least_err = {}
    for n_cams from 2<=n_cams<=max_n_cams:

        # Calculate in least reprojection error starting with all
        # possible combinations of 2 cameras, then start increasing
        # the number of cameras used.  For each number of cameras,
        # determine if there exists a combination with an acceptable
        # reprojection error.

        alpha = 1.0/n_cams

        # Can we short-circuit the rest of these computations?
        if not npy_isinf(least_err_by_n_cameras[n_cams-1]): # if it's infinity, it must be n_cams 0 or 1.
            # If we've calculated error for 2 less than n_cams
            if least_err_by_n_cameras[n_cams-1] > ACCEPTABLE_DISTANCE_PIXELS:
                if debug>5:
                    print 'HYPOTHESIS TEST -    shorting for n_cams %d (this=%f, ACCEPTABLE_DISTANCE_PIXELS=%f)'%(
                        n_cams,least_err_by_n_cameras[n_cams-1],ACCEPTABLE_DISTANCE_PIXELS)
                # and if the error for 1 less is too large, don't bother with more.
                break

        for cam_ids_used in recon.cam_combinations_by_size[n_cams]:
            missing_cam_data = 0 #False
            good_A_idx = []
            if debug>5:
                assert len(cam_ids_used)==n_cams
            for cam_id in cam_ids_used:
                if cam_id in bad_cam_ids:
                    missing_cam_data = 1 #True
                    break
                else:
                    i = cam_id2idx[cam_id]
                    good_A_idx.extend( (i*2, i*2+1) )
            if missing_cam_data == 1:
                continue

            # find 3D point
            A = allA[good_A_idx,:]
            u,d,vt=svd(A)
            X = vt[-1,:]/vt[-1,3] # normalize

            # calculate reprojection error
            mean_dist = 0.0
            for cam_id in cam_ids_used:
                orig_x,orig_y = all2d[cam_id]
                Pmat = Pmat_fastnx[cam_id]
                new_xyw = numpy.dot( Pmat, X ) # reproject 3d to 2d
                new_x, new_y = new_xyw[0:2]/new_xyw[2]

                dist = sqrt((orig_x-new_x)**2 + (orig_y-new_y)**2)
                mean_dist = mean_dist + dist*alpha
            if debug>5:
                print 'HYPOTHESIS TEST - mean_dist = %f for cam_ids_used = %s (always pt 0)'%(mean_dist,str(cam_ids_used))

            least_err = least_err_by_n_cameras[n_cams]
            if mean_dist < least_err:
                least_err_by_n_cameras[n_cams] = mean_dist
                cam_ids_for_least_err[n_cams] = cam_ids_used
                X_for_least_err[n_cams] = X[:3]

    # now we have the best estimate for 2 views, 3 views, ...
    best_n_cams = 2
    least_err = least_err_by_n_cameras[2]
    if debug>5:
        print 'HYPOTHESIS TEST - least_err, ACCEPTABLE_DISTANCE_PIXELS: %f, %f'%(least_err,ACCEPTABLE_DISTANCE_PIXELS)
    mean_dist = least_err
    if not (least_err < ACCEPTABLE_DISTANCE_PIXELS):
        raise NoAcceptablePointFound('least error was %f'%least_err)

    for n_cams from 3 <= n_cams <= max_n_cams:
        least_err = least_err_by_n_cameras[n_cams]
        if debug>5:
            print 'HYPOTHESIS TEST - n_cams %d: %f'%(n_cams,least_err)
        if npy_isinf(least_err):
            break # if we don't have e.g. 4 cameras, we won't have 5
        if least_err < ACCEPTABLE_DISTANCE_PIXELS:
            mean_dist = least_err
            best_n_cams = n_cams

    # now calculate final values
    cam_ids_used = cam_ids_for_least_err[best_n_cams]
    if with_water:
        # Even though (for speed reasons) we did not use proper
        # refraction-correct code above, we now recompute X using
        # refraction.
        cam_ids_and_points2d = []
        for cam_id in cam_ids_used:
            xy = found_data_dict[cam_id][:2]
            cam_ids_and_points2d.append(( cam_id, xy ))

        X = recon.find3d(cam_ids_and_points2d,
                         undistort=False, # points are already undistorted
                         return_line_coords = False,
                         )
    else:
        X = X_for_least_err[best_n_cams]
    del X_for_least_err

    # calculate line3d
    P = []
    for cam_id in cam_ids_used:
        x,y,area,slope,eccentricity, p1,p2,p3,p4 = found_data_dict[cam_id]
        if eccentricity > MINIMUM_ECCENTRICITY and not numpy.isnan(p1):
                P.append( (p1,p2,p3,p4) )
    if len(P) < 2:
        Lcoords = None
    else:
        P = numpy.array(P)
        try:
            u,d,vt=svd(P,full_matrices=True)
        except numpy.linalg.LinAlgError, err:
            print 'SVD error, P=',repr(P)
            Lcoords = None
        except:
            print 'Error on P'
            print P
            raise
        else:
            P = vt[0,:] # P,Q are planes (take row because this is transpose(V))
            Q = vt[1,:]

            # directly to Pluecker line coordinates
            Lcoords = ( -(P[3]*Q[2]) + P[2]*Q[3],
                          P[3]*Q[1]  - P[1]*Q[3],
                        -(P[2]*Q[1]) + P[1]*Q[2],
                        -(P[3]*Q[0]) + P[0]*Q[3],
                        -(P[2]*Q[0]) + P[0]*Q[2],
                        -(P[1]*Q[0]) + P[0]*Q[1] )
            if npy_isnan(Lcoords[0]):
                Lcoords = None
    ## c_lib.free(least_err_by_n_cameras)
    return X, Lcoords, cam_ids_used, mean_dist

## def undistort_image( char* src, int srcstep,
##                      char* dst, int dststep,
##                      int width, int height,
##                      float u0, float v0,
##                      float fx, float fy,
##                      float k1, float k2,
##                      float p1, float p2,
##                      int cn):
##     """derived from cvundistort.cpp"""

##     cdef int u, v, i
##     cdef float y, y2, ky, k2y, _2p1y, _3p1y2, p2y2, _fy, _fx
##     cdef float x, x2, kx, d, _u, _v, iu, iv, ifx, ify
##     cdef float a0, a1, b0, b1
##     cdef char* ptr
##     float t0, t1

##     _fx = 1.f/fx
##     _fy = 1.f/fy

##     for v from 0 <= v < height:
##         dst = dst + dststep;

##         y = (v - v0)*_fy
##         y2 = y*y
##         ky = 1 + (k1 + k2*y2)*y2;
##         k2y = 2*k2*y2
##         _2p1y = 2*p1*y
##         _3p1y2 = 3*p1*y2
##         p2y2 = p2*y2

##         for u from 0 <= u < width:
##             x = (u - u0)*_fx
##             x2 = x*x
##             kx = (k1 + k2*x2)*x2
##             d = kx + ky + k2y*x2
##             _u = fx*(x*(d + _2p1y) + p2y2 + (3*p2)*x2) + u0
##             _v = fy*(y*(d + (2*p2)*x) + _3p1y2 + p1*x2) + v0
##             iu = cvRound(_u*(1 << ICV_WARP_SHIFT))
##             iv = cvRound(_v*(1 << ICV_WARP_SHIFT))
##             ifx = iu & ICV_WARP_MASK
##             ify = iv & ICV_WARP_MASK

##             a0 = icvLinearCoeffs[ifx*2]
##             a1 = icvLinearCoeffs[ifx*2 + 1]
##             b0 = icvLinearCoeffs[ify*2]
##             b1 = icvLinearCoeffs[ify*2 + 1]

##             if( <unsigned>iv < <unsigned>(height - 1) &&
##                 <unsigned>iu < <unsigned>(width - 1) ):
##                 ptr = src + iv*srcstep + iu*cn
##                 for i from 0 <= i < cn:
##                     t0 = a1*CV_8TO32F(ptr[i]) + a0*CV_8TO32F(ptr[i+cn])
##                     t1 = a1*CV_8TO32F(ptr[i+srcstep]) + a0*CV_8TO32F(ptr[i + srcstep + cn])
##                     dst[u*cn + i] = <char>cvRound(b1*t0 + b0*t1)
##             else:
##                 for i from 0 <= i < cn:
##                     dst[u*cn + i] = 0

find_best_3d = hypothesis_testing_algorithm__find_best_3d # old name
