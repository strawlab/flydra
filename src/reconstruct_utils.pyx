import numpy
nx = numpy
fast_nx = numpy
inf = nx.inf

cdef double cinf
cinf = inf

import flydra.common_variables 

cdef float MINIMUM_ECCENTRICITY
MINIMUM_ECCENTRICITY = flydra.common_variables.MINIMUM_ECCENTRICITY

cdef float ACCEPTABLE_DISTANCE_PIXELS
ACCEPTABLE_DISTANCE_PIXELS = flydra.common_variables.ACCEPTABLE_DISTANCE_PIXELS

cdef extern from "math.h":
    double sqrt(double)
    int isnan(double x)
    int isinf(double x)

cdef class ReconstructHelper:
    cdef float fc1, fc2, cc1, cc2
    cdef float k1, k2, p1, p2

    def __init__(self, fc1, fc2, cc1, cc2, k1, k2, p1, p2 ):
        self.fc1 = fc1
        self.fc2 = fc2
        self.cc1 = cc1
        self.cc2 = cc2
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def get_K(self):
        K = nx.array((( self.fc1, 0, self.cc1),
                      ( 0, self.fc2, self.cc2),
                      ( 0, 0, 1)))
        return K

    def get_nlparams(self):
        return (self.k1, self.k2, self.p1, self.p2)

    def undistort(self, float x_kk, float y_kk):
        """undistort 2D coordinate pair

        Iteratively performs an undistortion using camera intrinsic
        parameters.

        Implementation translated from CalTechCal.

        See also the OpenCV reference manual, which has the equation
        used.
        """
        
        cdef float xl, yl

        cdef float xd, yd, x, y
        cdef float r_2, k_radial, delta_x, delta_y
        cdef int i

        # undoradial.m
        
        xd = ( x_kk - self.cc1 ) / self.fc1
        yd = ( y_kk - self.cc2 ) / self.fc2

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
        
        xl = (self.fc1)*x + (self.cc1)
        yl = (self.fc2)*y + (self.cc2)
        return (xl, yl)

    def distort(self, float xl, float yl):
        """distort 2D coordinate pair"""
        
        cdef float x, y, r_2, term1, xd, yd
        
        x = ( xl - self.cc1 ) / self.fc1
        y = ( yl - self.cc2 ) / self.fc2
        
        r_2 = x*x + y*y
        term1 = self.k1*r_2 + self.k2*r_2**2
        xd = x + x*term1 + (2*self.p1*x*y + self.p2*(r_2+2*x**2))

        # XXX OpenCV manual may be wrong -- double check this eqn
        # (esp. first self.p2 term):
        yd = y + y*term1 + (2*self.p2*x*y + self.p2*(r_2+2*y**2))

        xd = (self.fc1)*xd + (self.cc1)
        yd = (self.fc2)*yd + (self.cc2)
        
        return (xd, yd)
        
def find_best_3d( object recon, object d2):
    """
    
    Finds combination of cameras which uses the most number of cameras
    while minimizing mean reprojection error. Algorithm used accepts
    any camera combination with reprojection error less than the
    global variable ACCEPTABLE_DISTANCE_PIXELS.

    """
    cdef int max_n_cams, n_cams, best_n_cams
    cdef int i
    cdef int missing_cam_data
    cdef double alpha
    cdef double x, y, orig_x, orig_y, new_x, new_y
    cdef double dist, mean_dist, least_err
    
    cdef int MAX_CAMERAS
    # 10 = MAX_CAMERAS
    cdef double least_err_by_n_cameras[10] # fake dict (index = key)
    
    MAX_CAMERAS = 10 # should be a compile-time define
    
    fast_svd = numpy.linalg.svd # eliminate global name lookup
    
    cam_ids = recon.cam_ids # shorthand
    max_n_cams = len(cam_ids)
    if max_n_cams > MAX_CAMERAS:
        raise ValueError("too many cameras -- MAX_CAMERAS = %d"%(MAX_CAMERAS,))

    # Initialize least_err_by_n_cameras to be infinity.  Note that
    # values at 0th and 1st index will always remain infinity.
    for i from 0 <= i <= max_n_cams:
        least_err_by_n_cameras[i] = cinf
    
    allA = fast_nx.zeros( (2*max_n_cams,4),'d')
    bad_cam_ids = []
    cam_id2idx = {}
    all2d = {}
    Pmat_fastnx = {}
    Pmat_fastnx = recon.Pmat_fastnx # shorthand
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
        
        if not isinf(least_err_by_n_cameras[n_cams-2]):
            # If we've calculated error for 2 less than n_cams
            if least_err_by_n_cameras[n_cams-1] > ACCEPTABLE_DISTANCE_PIXELS:
                # and if the error for 1 less is too large, don't bother with more.
                break
            
        for cam_ids_used in recon.cam_combinations_by_size[n_cams]:
            missing_cam_data = 0 #False
            good_A_idx = []
            for cam_id in cam_ids_used:
                if cam_id in bad_cam_ids:
                    missing_cam_data = 1 #True
                    break
                else:
                    i = cam_id2idx[cam_id]
                    good_A_idx.extend( (i*2, i*2+1) )
            if missing_cam_data == 1:
                continue
            A = fast_nx.take(allA,good_A_idx)
            u,d,vt=fast_svd(A)
            X = vt[-1,:]/vt[-1,3] # normalize

            mean_dist = 0.0
            for cam_id in cam_ids_used:
                orig_x,orig_y = all2d[cam_id]
                Pmat = Pmat_fastnx[cam_id]
                new_xyw = fast_nx.matrixmultiply( Pmat, X ) # reproject 3d to 2d
                new_x, new_y = new_xyw[0:2]/new_xyw[2]

                dist = sqrt((orig_x-new_x)**2 + (orig_y-new_y)**2)
                mean_dist = mean_dist + dist*alpha

            least_err = least_err_by_n_cameras[n_cams]
            if mean_dist < least_err:
                least_err_by_n_cameras[n_cams] = mean_dist
                cam_ids_for_least_err[n_cams] = cam_ids_used
                X_for_least_err[n_cams] = X[:3]

    # now we have the best estimate for 2 views, 3 views, ...
    best_n_cams = 2
    least_err = least_err_by_n_cameras[2]
    mean_dist = least_err
    for n_cams from 3 <= n_cams <= max_n_cams:
        least_err = least_err_by_n_cameras[n_cams]
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
        if eccentricity > MINIMUM_ECCENTRICITY:
            P.append( (p1,p2,p3,p4) )
    if len(P) < 2:
        Lcoords = None
    else:
        P = fast_nx.array(P)
        u,d,vt=fast_svd(P,full_matrices=True)

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
