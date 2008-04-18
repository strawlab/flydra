#emacs, this is -*-Python-*- mode
# $Id: $

import numpy
import numpy.dual
inf = numpy.inf

cdef double cinf
cinf = inf

import flydra.common_variables
import flydra.undistort

cdef float MINIMUM_ECCENTRICITY
MINIMUM_ECCENTRICITY = flydra.common_variables.MINIMUM_ECCENTRICITY

class NoAcceptablePointFound(Exception):
    pass

cdef extern from "math.h":
    double sqrt(double)
    int isnan(double x)
    int isinf(double x)

def make_ReconstructHelper_from_rad_file(filename):
    params = {}
    execfile(filename,params)
    helper = ReconstructHelper(
        params['K11'], params['K22'], params['K13'], params['K23'],
        params['kc1'], params['kc2'], params['kc3'], params['kc4'])
    return helper

def make_ReconstructHelper(*args,**kw):
    return ReconstructHelper(*args,**kw)

cdef class ReconstructHelper:
    cdef double fc1, fc2, cc1, cc2
    cdef double k1, k2, p1, p2
    cdef double alpha_c

    def __init__(self, fc1, fc2, cc1, cc2, k1, k2, p1, p2, alpha_c=0 ):
        """create instance of ReconstructHelper

        ReconstructHelper(fc1, fc2, cc1, cc2, k1, k2, p1, p2 )
        where:
        fc - focal length
        cc - camera center
        k - radial distortion parameters (non-linear)
        p - tangential distortion parameters (non-linear)
        alpha_c - skew between X and Y pixel axes
        """
        self.fc1 = fc1
        self.fc2 = fc2
        self.cc1 = cc1
        self.cc2 = cc2
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.alpha_c = alpha_c

    def __reduce__(self):
        """this allows ReconstructHelper to be pickled"""
        args = (self.fc1, self.fc2, self.cc1, self.cc2,
                self.k1, self.k2, self.p1, self.p2, self.alpha_c)
        return (make_ReconstructHelper, args)

    def save_to_rad_file( self, fname, comments=None ):
        rad_fd = open(fname,'w')
        K = self.get_K()
        nlparams = self.get_nlparams()
        k1, k2, p1, p2 = nlparams
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
        rad_fd.write('\n')
        if comments is not None:
            comments = str(comments)
        rad_fd.write("comments = '%s';\n"%comments)
        rad_fd.write('\n')
        rad_fd.close()

    def __richcmp__(self,other,op):

        if op == 2 or op == 3:
            result = (numpy.allclose(self.get_K(),other.get_K()) and
                      numpy.allclose(self.get_nlparams(),other.get_nlparams()) )
            if op == 3:
                result = not result
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

    def get_nlparams(self):
        return (self.k1, self.k2, self.p1, self.p2)

    def undistort(self, double x_kk, double y_kk):
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

        for i from 0<=i<20:
            r_2 = x*x + y*y
            k_radial = 1.0 + (self.k1) * r_2 + (self.k2) * r_2*r_2
            delta_x = 2.0 * (self.p1)*x*y + (self.p2)*(r_2 + 2.0*x*x)
            delta_y = (self.p1) * (r_2 + 2.0*y*y)+2.0*(self.p2)*x*y
            x = (xd-delta_x)/k_radial
            y = (yd-delta_y)/k_radial

        # undoradial.m

        xl = (self.fc1)*x + (self.fc1*self.alpha_c)*y + (self.cc1)
        yl = (self.fc2)*y + (self.cc2)
        return (xl, yl)

    def distort(self, double xl, double yl):
        """distort 2D coordinate pair"""

        cdef double x, y, r_2, term1, xd, yd

        x = ( xl - self.cc1 ) / self.fc1
        y = ( yl - self.cc2 ) / self.fc2

        r_2 = x*x + y*y
        r_4 = r_2**2
        term1 = self.k1*r_2 + self.k2*r_4

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
        k = self.k1, self.k2, self.p1, self.p2, 0  # NL terms: r^2, r^4, tan1, tan2, r^6

        imnew = flydra.undistort.rect(numpy_image, f=f, c=c, k=k) # perform the undistortion
        imnew = imnew.astype(numpy.uint8)
        return imnew

def hypothesis_testing_algorithm__find_best_3d( object recon, object d2,
                                                double ACCEPTABLE_DISTANCE_PIXELS,
                                                int debug=0):
    """Use hypothesis testing algorithm to find best 3D point

    Finds combination of cameras which uses the most number of cameras
    while minimizing mean reprojection error. Algorithm used accepts
    any camera combination with reprojection error less than the
    variable ACCEPTABLE_DISTANCE_PIXELS.

    """
    cdef int max_n_cams, n_cams, best_n_cams
    cdef int i
    cdef int missing_cam_data
    cdef double alpha
    cdef double x, y, orig_x, orig_y, new_x, new_y
    cdef double dist, mean_dist, least_err

    cdef double least_err_by_n_cameras[10] # fake dict (index = key)

    svd = numpy.dual.svd # eliminate global name lookup

    cam_ids = recon.cam_ids # shorthand
    max_n_cams = len(cam_ids)

    # Initialize least_err_by_n_cameras to be infinity.  Note that
    # values at 0th and 1st index will always remain infinity.
    for i from 0 <= i <= max_n_cams:
        least_err_by_n_cameras[i] = cinf

    allA = numpy.zeros( (2*max_n_cams,4), dtype=numpy.float64)
    bad_cam_ids = []
    cam_id2idx = {}
    all2d = {}
    Pmat_fastnx = recon.Pmat # shorthand
    for i,cam_id in enumerate(cam_ids):
        cam_id2idx[cam_id] = i

        # do we have incoming data?
        try:
            value_tuple = d2[cam_id]
        except KeyError:
            bad_cam_ids.append( cam_id )
            continue # don't build this row

        # was a 2d point found?
        xy_values = value_tuple[:2]
        x,y = xy_values
        if isnan(x):
            bad_cam_ids.append( cam_id )
            continue # don't build this row


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
        if not isinf(least_err_by_n_cameras[n_cams-1]): # if it's infinity, it must be n_cams 0 or 1.
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
    mean_dist = least_err
    if not (least_err < ACCEPTABLE_DISTANCE_PIXELS):
        raise NoAcceptablePointFound('least error was %f'%least_err)

    for n_cams from 3 <= n_cams <= max_n_cams:
        least_err = least_err_by_n_cameras[n_cams]
        if debug>5:
            print 'HYPOTHESIS TEST - n_cams %d: %f'%(n_cams,least_err)
        if isinf(least_err):
            break # if we don't have e.g. 4 cameras, we won't have 5
        if least_err < ACCEPTABLE_DISTANCE_PIXELS:
            mean_dist = least_err
            best_n_cams = n_cams

    # now calculate final values
    cam_ids_used = cam_ids_for_least_err[best_n_cams]
    X = X_for_least_err[best_n_cams]

    # calculate line3d
    P = []
    for cam_id in cam_ids_used:
        x,y,area,slope,eccentricity, p1,p2,p3,p4 = d2[cam_id]
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
            if isnan(Lcoords[0]):
                Lcoords = None
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
