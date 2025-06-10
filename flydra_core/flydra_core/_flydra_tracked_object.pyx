#emacs, this is -*-Python-*- mode
#cython: language_level=2

import numpy as np
cimport numpy as np
cimport _fastgeom
cimport _mahalanobis
cimport _pmat_jacobian

import numpy
import numpy as np
import time, sys
import adskalman.adskalman as kalman

import flydra_core.kalman.ekf as kalman_ekf
#import flydra_core.geom as geom
import _fastgeom as geom
import flydra_core.geom
import _mahalanobis
import math, struct
import flydra_core.data_descriptions
from flydra_core.kalman.point_prob import some_rough_negative_log_likelihood
import collections
from pprint import pprint

cdef double c_inf
c_inf = np.inf

__all__ = ['TrackedObject']

PT_TUPLE_IDX_X = flydra_core.data_descriptions.PT_TUPLE_IDX_X
PT_TUPLE_IDX_Y = flydra_core.data_descriptions.PT_TUPLE_IDX_Y
PT_TUPLE_IDX_AREA = flydra_core.data_descriptions.PT_TUPLE_IDX_AREA
PT_TUPLE_IDX_SLOPE = flydra_core.data_descriptions.PT_TUPLE_IDX_SLOPE
PT_TUPLE_IDX_ECCENTRICITY = flydra_core.data_descriptions.PT_TUPLE_IDX_ECCENTRICITY
PT_TUPLE_IDX_P1 = flydra_core.data_descriptions.PT_TUPLE_IDX_P1
PT_TUPLE_IDX_P2 = flydra_core.data_descriptions.PT_TUPLE_IDX_P2
PT_TUPLE_IDX_P3 = flydra_core.data_descriptions.PT_TUPLE_IDX_P3
PT_TUPLE_IDX_P4 = flydra_core.data_descriptions.PT_TUPLE_IDX_P4

PT_TUPLE_IDX_FRAME_PT_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX
PT_TUPLE_IDX_CUR_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_CUR_VAL_IDX
PT_TUPLE_IDX_MEAN_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_MEAN_VAL_IDX
PT_TUPLE_IDX_SUMSQF_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_SUMSQF_VAL_IDX

NO_LCOORDS = numpy.nan,numpy.nan,numpy.nan,  numpy.nan,numpy.nan,numpy.nan

cdef double cnan
cnan = np.nan

ctypedef int mybool

cdef extern from "math.h":
    double sqrt(double)

cpdef evaluate_pmat_jacobian(object pmats_and_points_cov, np.ndarray[np.double_t, ndim=1] xhatminus):
    cdef int N
    cdef mybool missing_data
    cdef int i
    cdef int ss
    cdef _pmat_jacobian.PinholeCameraModelWithJacobian pinhole_model
    cdef np.ndarray[np.double_t, ndim=1] y
    cdef np.ndarray[np.double_t, ndim=2] C
    cdef np.ndarray[np.double_t, ndim=1] hx
    cdef np.ndarray[np.double_t, ndim=2] R
    cdef np.ndarray[np.double_t, ndim=1] hx_i

    ss = xhatminus.shape[0]

    N = len(pmats_and_points_cov) # number of observations
    if N > 0:
        missing_data = False
    else:
        missing_data = True

    # Create 2N vector of N observations.
    y = numpy.empty((2*N,), dtype=numpy.float64)

    # Create 2N x 4 observation model matrix (jacobian of h() at xhatminus).
    C = numpy.zeros((2*N,ss), dtype=numpy.float64)

    # Create 2N vector h(xhatminus) where xhatminus is the a
    # priori estimate and h() is the observation model.
    hx = numpy.empty((2*N,), dtype=numpy.float64)

    # Create 2N x 2N observation covariance matrix
    R = numpy.zeros((2*N,2*N), dtype=numpy.float64)

    # evaluate jacobian for each participating camera
    for i,(nonlin_model,xy2d_obs,cov) in enumerate(pmats_and_points_cov):
        pinhole_model = nonlin_model

        # fill prediction vector [ h(xhatminus) ]
        hx_i = pinhole_model.evaluate(xhatminus[:3])
        hx[2*i:2*i+2] = hx_i

        # fill observation vector
        y[2*i:2*i+2] = xy2d_obs

        # fill observation model
        C_i = pinhole_model.evaluate_jacobian_at(xhatminus[:3])
        C[2*i:2*i+2,:3] = C_i

        # fill observation covariance
        R[2*i:2*i+2,2*i:2*i+2]=cov

    ## if 1:
    ##     print 'C'
    ##     print C
    ##     print 'y'
    ##     print y
    ##     print 'hx'
    ##     print hx
    ##     print 'R'
    ##     print R
    return y, hx, C, R, missing_data

def obs2d_hashable( arr ):
    # XXX FIXME: this could likely be sped up
    assert arr.dtype == numpy.uint16
    assert len(arr.shape)==1

    # sort array based on camn
    camns = arr[0::2]
    pt_idx = arr[1::2]
    camn_order = numpy.argsort(camns)

    # fill new array with sorted values
    newarr = numpy.zeros( (len(arr),), dtype = numpy.uint16 )
    newarr[ camn_order*2 ] = camns
    newarr[ camn_order*2+1 ] = pt_idx

    val = newarr.tobytes()
    return val

cdef class TrackedObject:
    """
    Track one object using a Kalman filter.

    """
    cdef readonly long current_frameno
    cdef readonly long last_frameno_with_data
    cdef long long max_frames_skipped
    cdef readonly mybool kill_me
    cdef mybool save_all_data
    cdef double area_threshold, area_threshold_for_orientation

    cdef object reconstructor, my_kalman
    cdef object distorted_pixel_euclidian_distance_accept
    cdef double max_variance
    cdef object ekf_observation_covariance_pixels

    cdef readonly object frames, xhats, timestamps, Ps, MLE_position, MLE_Lcoords
    cdef readonly object observations_frames, observations_2d
    cdef int disable_image_stat_gating, orientation_consensus
    cdef object fake_timestamp
    cdef readonly unsigned int obj_id
    cdef object ekf_kalman_A, ekf_kalman_Q

    def __init__(self,
                 reconstructor, # the Reconstructor instance
                 obj_id,
                 long frame, # frame number of first data
                 obs0_position, # first data
                 obs0_Lcoords, # first data
                 first_observation_camns,
                 first_observation_idxs,
                 kalman_model=None,
                 save_all_data=False,
                 double area_threshold=0.0,
                 double area_threshold_for_orientation=0.0,
                 disable_image_stat_gating=False,
                 orientation_consensus=0,
                 fake_timestamp=None
                 ):
        """

        arguments
        =========
        reconstructor - reconstructor instance
        obj_id - unique identifier for each object
        frame - frame number of first observation data
        obs0_position - first observation (in arbitrary units)
        kalman_model - Kalman parameters
        area_threshold - minimum area to consider for tracking use
        """
        self.obj_id = obj_id
        self.area_threshold = area_threshold
        self.area_threshold_for_orientation = area_threshold_for_orientation
        self.save_all_data = save_all_data
        self.kill_me = False
        self.reconstructor = reconstructor
        self.distorted_pixel_euclidian_distance_accept=kalman_model.get('distorted_pixel_euclidian_distance_accept',None)
        self.disable_image_stat_gating = disable_image_stat_gating
        self.orientation_consensus = orientation_consensus
        self.fake_timestamp = fake_timestamp

        self.current_frameno = frame
        self.last_frameno_with_data = frame
        obs0_position = obs0_position
        if obs0_Lcoords is None:
            obs0_Lcoords = NO_LCOORDS
        else:
            PL_line3d = flydra_core.geom.line_from_HZline(obs0_Lcoords) # PlueckerLine instance
            loc = PL_line3d.closest() # closest point on line to origin
            line3d = flydra_core.geom.line_from_points( loc, loc+PL_line3d.direction() )
            obs0_Lcoords = line3d.to_hz()
        ss = kalman_model['ss']
        initial_x = numpy.zeros( (ss,) )
        initial_x[:3] = obs0_position
        P_k1=numpy.eye(ss) # initial state error covariance guess
        for i in range(0,3):
            P_k1[i,i]=kalman_model['initial_position_covariance_estimate']
        for i in range(3,6):
            P_k1[i,i]=kalman_model.get('initial_velocity_covariance_estimate',0)
        if ss >= 9:
            for i in range(6,9):
                P_k1[i,i]=kalman_model['initial_acceleration_covariance_estimate']

        self.max_variance = kalman_model['max_variance_dist_meters']**2 # square so that it is in variance units

        self.max_frames_skipped = kalman_model['max_frames_skipped']

        if kalman_model.get( 'isEKF', False):
            self.ekf_kalman_A = kalman_model['A']
            self.ekf_kalman_Q = kalman_model['Q']
            # EKF
            self.my_kalman = kalman_ekf.EKF(
                initial_x=initial_x,
                initial_P=P_k1,
                )
            self.ekf_observation_covariance_pixels = kalman_model['ekf_observation_covariance_pixels']
        else:
            # non-EKF
            self.my_kalman = kalman.KalmanFilter(kalman_model['A'],
                                                 kalman_model['C'],
                                                 kalman_model['Q'],
                                                 kalman_model['R'],
                                                 initial_x,
                                                 P_k1)
        self.frames = [frame]
        self.xhats = [initial_x]
        if self.fake_timestamp is None:
            self.timestamps = [time.time()]
        else:
            self.timestamps = [self.fake_timestamp]
        self.Ps = [P_k1]

        self.observations_frames = [frame]
        self.MLE_position = [obs0_position]
        self.MLE_Lcoords = [obs0_Lcoords]

        first_observations_2d_pre = [[camn,idx] for camn,idx in zip(first_observation_camns,first_observation_idxs)]
        first_observations_2d = []
        for obs in first_observations_2d_pre:
            first_observations_2d.extend( obs )
        first_observations_2d = numpy.array(first_observations_2d,dtype=numpy.uint16) # if saved as VLArray, should match with atom type

        self.observations_2d = [first_observations_2d]

        self.max_frames_skipped=kalman_model['max_frames_skipped']

        # Don't run kalman filter with initial data, as this would
        # cause error estimates to drop too low.

    def __repr__(self):
        return '<TRO frames[0]=%r observations_2d[0]=%r xhats[0]=%r>' % (
            self.frames[0], self.observations_2d[0], self.xhats[0])

    def debug_info(self,level=3):
        if level > 5:
            sys.stdout.write('%s\n'%self)
            N_pts = len(self.xhats)
            start_idx = max( N_pts-10, 0 )
            for i in range(start_idx,N_pts):
                this_Pmean = math.sqrt(self.Ps[i][0,0]**2 + self.Ps[i][1,1]**2 + self.Ps[i][2,2]**2)
                sys.stdout.write( ' '.join(map(str,['  ',i,self.frames[i],self.xhats[i][:3],this_Pmean,])) )
                if self.frames[i] in self.observations_frames:
                    j =  self.observations_frames.index(  self.frames[i] )
                    sys.stdout.write( '%s\n'%self.MLE_position[j] )
                else:
                    sys.stdout.write('\n')
            sys.stdout.write('\n')
        elif level > 2:
            sys.stdout.write('%d observations, %d estimates for %s\n'%(len(self.xhats),len(self.MLE_position),self))

    def get_distance_and_nsigma( self, testx ):
        xhat = self.xhats[-1][:3]

        dist2 = numpy.sum((testx-xhat)**2) # distance squared
        dist = numpy.sqrt(dist2)

        # XXX This should be mahalanobis distance, probably.
        P = self.Ps[-1]
        Pmean = numpy.sqrt(numpy.sum([P[i,i]**2 for i in range(3)])) # sigma squared
        sigma = numpy.sqrt(Pmean)
        return dist, (dist/sigma)

    def kill(self):
        # called when killed
        if self.save_all_data:
            return

        # find last data
        last_observation_frame = self.observations_frames[-1]

        # eliminate estimates past last observation
        while 1:
            if self.frames[-1] > last_observation_frame:
                self.frames.pop()
                self.xhats.pop()
                self.timestamps.pop()
                self.Ps.pop()
            else:
                break

    cpdef calculate_a_posteriori_estimate(self,
                                        long frame,
                                        object data_dict,
                                        object camn2cam_id,
                                        int debug1=0,
                                        int skip_data_association=0,
                                        object original_camns_and_idxs=None,
                                        object original_cam_ids_and_points2d=None,
                                        ):
        # Step 1. Update Kalman state to a priori estimates for this frame.
        # Step 1.A. Update Kalman state for each skipped frame.

        # Make sure we have xhat_k1 (previous frames' a posteriori)

        # For each frame that was skipped, step the Kalman filter.
        # Since we have no observation, the estimated error will
        # rise.
        cdef long i, frames_since_update
        cdef long frames_skipped
        cdef double Pmean
        frames_since_update = frame-self.current_frameno-1

        if debug1>2:
            print 'doing',self,'============--'
            print 'updating for %d frames since update'%(frames_since_update,)

        for i in range(frames_since_update):
            if isinstance(self.my_kalman, kalman_ekf.EKF):
                xhat, P = self.my_kalman.step(self.ekf_kalman_A,
                                              self.ekf_kalman_Q)
            else:
                xhat, P = self.my_kalman.step()
            ############ save outputs ###############
            self.frames.append( self.current_frameno + i + 1 )
            self.xhats.append( xhat )
            self.timestamps.append( 0.0 )
            self.Ps.append( P )

        self.current_frameno = frame
        this_observations_2d_hash = None
        used_camns_and_idxs = []
        all_close_camn_pt_idxs = []
        Pmean = c_inf
        if not self.kill_me:
            # Step 1.B. Update Kalman to provide a priori estimates for this frame
            if isinstance(self.my_kalman, kalman_ekf.EKF):
                xhatminus, Pminus = self.my_kalman.step1__calculate_a_priori(
                    self.ekf_kalman_A, self.ekf_kalman_Q)
            else:
                xhatminus, Pminus = self.my_kalman.step1__calculate_a_priori()
            if debug1>2:
                print 'xhatminus'
                print xhatminus
                print 'Pminus'
                print Pminus
                print

            # Step 2. Filter incoming 2D data to use informative points (data association)
            if skip_data_association:
                position_MLE = numpy.nan*numpy.ones( (3,)) # skip
                Lcoords = None
                used_camns_and_idxs = original_camns_and_idxs
                cam_ids_and_points2d = original_cam_ids_and_points2d
                all_close_camn_pt_idxs = [] # not important when skipping data association
            else:
                (position_MLE, Lcoords, used_camns_and_idxs,
                 cam_ids_and_points2d,
                 all_close_camn_pt_idxs) = self._filter_data(xhatminus, Pminus,
                                                             data_dict,
                                                             camn2cam_id,
                                                             debug=debug1)
            if debug1>2:
                print 'position MLE, used_camns_and_idxs',position_MLE,used_camns_and_idxs
                print 'Lcoords (3D body orientation) : %s'%str(Lcoords)

            # Step 3. Incorporate observation to estimate a posteriori
            if isinstance(self.my_kalman, kalman_ekf.EKF):
                prediction_3d = xhatminus[:3]
                pmats_and_points_cov = [ (
                                          self.reconstructor.get_model_with_jacobian(cam_id),
                                          value_tuple[:2],#just first 2 components (x,y) become xy2d_observed
                                          self.ekf_observation_covariance_pixels)
                                         for (cam_id,value_tuple) in cam_ids_and_points2d]
                y,hx,C,R,missing_data = evaluate_pmat_jacobian(
                    pmats_and_points_cov,xhatminus)
                xhat, P = self.my_kalman.step2__calculate_a_posteriori(
                    xhatminus, Pminus, y=y,hx=hx,
                    C=C,R=R,missing_data=missing_data)
            else:
                xhat, P = self.my_kalman.step2__calculate_a_posteriori(
                    xhatminus, Pminus,
                    y=position_MLE)

            # calculate mean variance of x y z position (assumes first three components of state vector are position)

            # XXX should probably use trace/N (mean of variances) or determinant (volume of variance)
            Pmean = numpy.sqrt(numpy.sum([P[i,i]**2 for i in range(3)]))

            if debug1>2:
                print 'xhat'
                print xhat
                print 'P'
                print P
                print 'Pmean',Pmean

            # XXX Need to test if error for this object has grown beyond a
            # threshold at which it should be terminated.
            if Pmean > self.max_variance:
                self.kill_me = True
                if debug1>=1:
                    print 'will kill next time because Pmean too large (%f > %f)'%(Pmean,self.max_variance)

            frames_skipped = frame - self.last_frameno_with_data
            if frames_skipped > self.max_frames_skipped:
                self.kill_me = True
                if debug1>=1:
                    print 'will kill next time because frames skipped (%ld > %ld, frame %ld)'%(
                        frames_skipped, self.max_frames_skipped,frame)

            ############ save outputs ###############
            self.frames.append( frame )
            self.xhats.append( xhat )
            if self.fake_timestamp is None:
                self.timestamps.append(time.time())
            else:
                self.timestamps.append(self.fake_timestamp)
            self.Ps.append( P )

            if position_MLE is not None:
                self.last_frameno_with_data = frame
                self.observations_frames.append( frame )
                self.MLE_position.append( position_MLE )
                if Lcoords is None:
                    self.MLE_Lcoords.append( NO_LCOORDS )
                else:
                    self.MLE_Lcoords.append( Lcoords )
                this_observations_2d = []
                for (camn, frame_pt_idx, dd_idx) in used_camns_and_idxs:
                    this_observations_2d.extend( [camn,frame_pt_idx] )
                this_observations_2d = numpy.array( this_observations_2d, dtype=numpy.uint16 ) # convert to numpy
                self.observations_2d.append( this_observations_2d )
                this_observations_2d_hash = obs2d_hashable( this_observations_2d )
            if debug1>2:
                print
        return (used_camns_and_idxs, this_observations_2d_hash,
                Pmean, all_close_camn_pt_idxs)

    def remove_previous_observation(self, int debug1=0):

        # This will remove the just-done observation from my
        # state. Thus, this instance will act as it it skipped data on
        # one observation.

        self.current_frameno -= 1

        # Remove most recent information
        frame = self.frames.pop()
        if debug1>2:
            sys.stdout.write( '%s removing previous observation (from frame %d)\n'%(self,frame,))

        self.xhats.pop()
        self.timestamps.pop()
        self.Ps.pop()
        if self.observations_frames[-1] == frame:
            self.observations_frames.pop()
            self.MLE_position.pop()
            self.MLE_Lcoords.pop()
            self.observations_2d.pop()

        # reset self.my_kalman
        self.my_kalman.xhat_k1 = self.xhats[-1]
        self.my_kalman.P_k1 = self.Ps[-1]

        self.last_frameno_with_data = self.observations_frames[-1]

    cpdef _filter_data(self, object xhatminus, object Pminus,
                       object data_dict, object camn2cam_id,
                       int debug=0):
        """given state estimate, select useful incoming data and make new observation

        This function 'solves' the data association problem. 2D
        observations are associated with a kalman object if they are
        with some distance between a predicted image of the object on
        the camera plane. The distance estimation is made in 3D as the
        Euclidian distance between the ray in 3D space corresponding
        to the image point observed and the predicted 3D location of
        the kalman object. (Other distance metrics could also be
        implemented, such as 2D Euclidian distance on the image
        plane. Note that a raw threshold for this is implemented with
        distorted_pixel_euclidian_distance_accept.)

        """
        # For each camera, predict 2D image location and error distance
        cdef double least_nll, nll_this_point
        nll_this_point = 0.0
        cdef double dist2, dist, p_y_x
        dist2 = 0.0
        dist = 0.0
        cdef int gated_in, pixel_dist_criterion_passed

        cdef double pt_area, mean_val, sumsqf_val, area
        pt_area = 0.0
        mean_val = 0.0
        sumsqf_val = 0.0
        cdef int cur_val
        cur_val = 0
        cdef int camn, frame_pt_idx
        cdef _fastgeom.PlueckerLine projected_line
        cdef _fastgeom.ThreeTuple best_3d_location

        all_close_camn_pt_idxs = [] # store all "maybes"

        prediction_3d = xhatminus[:3]
        cdef _fastgeom.ThreeTuple fast_prediction_3d = _fastgeom.ThreeTuple(xhatminus[:3])
        pixel_dist_cmp = self.distorted_pixel_euclidian_distance_accept
        neg_predicted_3d = -geom.ThreeTuple( prediction_3d )
        cam_ids_and_points2d = []

        used_camns_and_idxs = []
        if debug>2:
            print '_filter_data():'
        for camn, candidate_point_list in data_dict.items():
            cam_id = camn2cam_id[camn]

            if pixel_dist_cmp is not None:
                predicted_2d_distorted = self.reconstructor.find2d(cam_id,prediction_3d,distorted=True)

            if debug>2:
                predicted_2d_undistorted = self.reconstructor.find2d(cam_id,prediction_3d,distorted=False)
                print '  cam_id',cam_id,'camn',camn,'--------'
                print '    predicted_2d (undistorted)',predicted_2d_undistorted

            # Iterate over all 2D data. Find the 2D point which minimizes::
            #    f(area, maxabsdiff, etc) + d(y,xhat).

            # According to Kristin, this is equivalent to choosing the
            # point with maximum cumulative probability::
            #    p( y_t | x_t ) p( x_t, x_{t-1})

            # [In this case f(area,...) represents negative log of
            # the likelihood of observation y given x.]

            # For large numbers of 2d points in data_dict, probably
            # faster to compute 2d image of error ellipsoid and see if
            # data_dict points fall inside that. For now, however,
            # compute distance in 3d. This avoids having to do a
            # (nonlinear) perspective projection of a multivariante
            # normal.

            least_nll = c_inf
            closest_idx = None

            Pminus_inv = None # defer inverting Pminus until necessary.

            for idx,(pt_undistorted,projected_line) in enumerate(
                candidate_point_list):

                # Iterate over each candidate point. Each point has:
                #  * index 'idx'
                #  * a bunch of data (such as 2D coordinates x and y)
                #    'pt_undistorted',
                #  * and the 3D line passing through that ray and the
                #    image plane.

                gated_in = False

                # First, a quick gating based on image plane distance.

                pixel_dist_criterion_passed = True
                if pixel_dist_cmp is not None:
                    # XXX TODO: fixme: should just pass in distorted pixel coordinates, but saves reorganizing all this code.
                    pt_x_undist =  pt_undistorted[PT_TUPLE_IDX_X]
                    pt_y_undist =  pt_undistorted[PT_TUPLE_IDX_Y]
                    pt_x_dist, pt_y_dist = self.reconstructor.distort( cam_id, (pt_x_undist, pt_y_undist) )
                    pixel_dist = numpy.sqrt((predicted_2d_distorted[0] - pt_x_dist)**2 + (predicted_2d_distorted[1] - pt_y_dist)**2)
                    if pixel_dist > pixel_dist_cmp:
                        pixel_dist_criterion_passed = False

                if pixel_dist_criterion_passed:

                    # Second, a quick gating based on image attributes.

                    pt_area = pt_undistorted[PT_TUPLE_IDX_AREA]
                    tmp = pt_undistorted[PT_TUPLE_IDX_CUR_VAL_IDX]
                    if tmp is None:
                        if not self.disable_image_stat_gating:
                            raise ValueError(
                                '--disable-image-stat-gating not specified and '
                                'no image statistics were saved.')
                    else:
                        cur_val = tmp
                        mean_val = pt_undistorted[PT_TUPLE_IDX_MEAN_VAL_IDX]
                        sumsqf_val = pt_undistorted[PT_TUPLE_IDX_SUMSQF_VAL_IDX]

                    if cur_val is None or self.disable_image_stat_gating:
                        p_y_x = 0.0 # no penalty
                    else:
                        # this could even depend on 3d geometry
                        p_y_x = some_rough_negative_log_likelihood(
                            pt_area, cur_val, mean_val, sumsqf_val )

                    if pt_area < self.area_threshold:
                        # This should not fall under
                        # "disable_image_stat_gating" -- it is a separate test.
                        p_y_x = np.inf

                    if np.isfinite(p_y_x):
                        gated_in = True

                        # Find point on ray with closest Mahalanobis distance.

                        if Pminus_inv is None:
                            Pminus_inv = numpy.linalg.inv( Pminus[:3,:3] )
                        best_3d_location = _mahalanobis.line_fit_3d(
                            projected_line, fast_prediction_3d, Pminus_inv )

                        # find closest distance between projected_line and predicted position for each 2d point
                        #   squared distance between prediction and camera ray
                        dist2=_mahalanobis.dist2( best_3d_location,
                                                  fast_prediction_3d,
                                                  Pminus_inv )
                        dist = sqrt(dist2)
                        nll_this_point = p_y_x + dist # negative log likelihood of this point

                frame_pt_idx = pt_undistorted[PT_TUPLE_IDX_FRAME_PT_IDX]
                if debug>2:
                    extra_print = []
                    if not self.disable_image_stat_gating:
                        cur_val = pt_undistorted[PT_TUPLE_IDX_CUR_VAL_IDX]
                        mean_val = pt_undistorted[PT_TUPLE_IDX_MEAN_VAL_IDX]
                        sumsqf_val = pt_undistorted[PT_TUPLE_IDX_SUMSQF_VAL_IDX]
                        if pixel_dist_criterion_passed:
                            extra_print.append('(pt_area %f, cur %d, mean %.1f, sumsqf %.1f)'%(
                                pt_area,cur_val,mean_val,sumsqf_val))

                    if gated_in:
                        extra_print.append('distorted %.1f %.1f (pixel_dist = %.1f, mahal dist = %.1f, criterion passed=%s)'%(
                            pt_x_dist, pt_y_dist, pixel_dist, dist, str(pixel_dist_criterion_passed)))
                    print '    ->', dist2, pt_undistorted[:2], '(idx %d) %s'%(
                        frame_pt_idx,' '.join(extra_print))

                if gated_in:
                    if debug>2:
                        print '       (acceptable)'
                    if nll_this_point < least_nll:
                        if debug>2:
                            print '       (so far the best -- taking)'
                        least_nll = nll_this_point
                        closest_idx = idx
                    all_close_camn_pt_idxs.append( (camn, frame_pt_idx) )
                elif debug>2:
                    print '       (not acceptable)'

            if closest_idx is not None:
                pt_undistorted, projected_line = candidate_point_list[closest_idx]
                area = pt_undistorted[PT_TUPLE_IDX_AREA]
                if area >= self.area_threshold_for_orientation:
                    # with orientation
                    observed_2d = (pt_undistorted[PT_TUPLE_IDX_X],
                                   pt_undistorted[PT_TUPLE_IDX_Y],
                                   pt_undistorted[PT_TUPLE_IDX_AREA],
                                   pt_undistorted[PT_TUPLE_IDX_SLOPE],
                                   pt_undistorted[PT_TUPLE_IDX_ECCENTRICITY],
                                   pt_undistorted[PT_TUPLE_IDX_P1],
                                   pt_undistorted[PT_TUPLE_IDX_P2],
                                   pt_undistorted[PT_TUPLE_IDX_P3],
                                   pt_undistorted[PT_TUPLE_IDX_P4])
                else:
                    # with no orientation
                    observed_2d = (pt_undistorted[PT_TUPLE_IDX_X],
                                   pt_undistorted[PT_TUPLE_IDX_Y],
                                   pt_undistorted[PT_TUPLE_IDX_AREA],
                                   cnan,
                                   cnan,
                                   cnan,
                                   cnan,
                                   cnan,
                                   cnan)
                cam_ids_and_points2d.append( (cam_id,observed_2d) )
                frame_pt_idx = pt_undistorted[PT_TUPLE_IDX_FRAME_PT_IDX]
                used_camns_and_idxs.append( (camn, frame_pt_idx, closest_idx) )
                if debug>2:
                    print 'best match idx %d (%s)'%(closest_idx, str(pt_undistorted[:2]))

        Lcoords = None # default to no line coordinates
        # Now cam_ids_and_points2d has just the 2d points we'll use for this reconstruction
        if len(cam_ids_and_points2d)==1:
            # keep 3D "observation" because we need to save 2d observations
            position_MLE = numpy.nan*numpy.ones( (3,))
        elif len(cam_ids_and_points2d)>=2:
            if self.reconstructor.wateri is not None:
                # do not attempt this if we have refractive boundary
                # keep 3D "observation" because we need to save 2d observations
                position_MLE = numpy.nan*numpy.ones( (3,))
            else:
                position_MLE, Lcoords = self.reconstructor.find3d(
                    cam_ids_and_points2d, return_line_coords = True,
                    orientation_consensus=self.orientation_consensus)
        else:
            position_MLE = None
        return (position_MLE, Lcoords, used_camns_and_idxs,
                cam_ids_and_points2d, all_close_camn_pt_idxs)

    def get_most_recent_data(self):
        if not len(self.xhats):
            return
        xhat = self.xhats[-1]
        P = self.Ps[-1]
        return self.obj_id, xhat,P
