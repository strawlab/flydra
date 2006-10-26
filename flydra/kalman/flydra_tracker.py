import numpy
import adskalman as kalman
import params
import flydra.geom as geom
import math

__all__ = ['TrackedObject','Tracker']

class TrackedObject:
    """
    Track one object using a Kalman filter.

    TrackedObject handles all internal units in meters, but external interfaces are original units

    """
    
    def __init__(self,
                 reconstructor_meters,
                 frame,
                 first_observation_orig_units,
                 scale_factor=None,
                 n_sigma_accept = 3.0, # default: arbitrarily set to 3
                 max_variance_dist_meters = 0.010, # default: allow error to grow to 10 mm before dropping
                 initial_position_covariance_estimate = 1e-6, # default: initial guess 1mm ( (1e-3)**2 meters)
                 initial_acceleration_covariance_estimate = 15, # default: arbitrary initial guess, rather large
                 Q = None,
                 R = None,
                 ):
        """

        arguments
        ---------
        reconstructor_meters - reconstructor instance with internal units of meters
        frame - frame number of first observation data
        first_observation_orig_units - first observation (in arbitrary units)
        scale_factor - how to convert from arbitrary units (of observations) into meters (e.g. 1e-3 for mm)
        n_sigma_accept - gobble 2D data points that are within this distance from predicted 2D location
        max_variance_dist_meters - estimated error (in meters) to allow growth to before killing tracked object
        initial_position_covariance_estimate -
        initial_acceleration_covariance_estimate -
        Q - process covariance matrix
        R - measurement noise covariance matrix
        """
        self.kill_me = False
        self.reconstructor_meters = reconstructor_meters
        self.current_frameno = frame
        self.n_sigma_accept = n_sigma_accept # arbitrary
        self.max_variance_dist_meters = max_variance_dist_meters
        if scale_factor is None:
            print 'WARNING: no scale_factor given in flydra_tracker, assuming 1e-3'
            self.scale_factor = 1e-3
        else:
            self.scale_factor = scale_factor
        first_observation_meters = first_observation_orig_units*self.scale_factor
        initial_x = numpy.hstack((first_observation_meters, # convert to mm from meters
                                  (0,0,0, 0,0,0))) # zero velocity and acceleration
        ss = params.A.shape[0]
        P_k1=numpy.eye(ss) # initial state error covariance guess
        for i in range(0,3):
            P_k1[i,i]=initial_position_covariance_estimate
        for i in range(6,9):
            P_k1[i,i]=initial_acceleration_covariance_estimate

        if Q is None:
            Q = params.Q
        if R is None:
            R = params.R
        self.my_kalman = kalman.KalmanFilter(params.A,
                                             params.C,
                                             Q,
                                             R,
                                             initial_x,
                                             P_k1)
        self.frames = []
        self.xhats = []
        self.Ps = []
        
        self.observations_frames = [frame]
        self.observations_data = [first_observation_meters]
        
        # Don't run kalman filter with initial data, as this would
        # cause error estimates to drop too low.
        
    def gobble_2d_data_and_calculate_a_posteri_estimate(self,frame,data_dict):
        # Step 1. Update Kalman state to a priori estimates for this frame.
        # Step 1.A. Update Kalman state for each skipped frame.
        if self.current_frameno is not None:
            # Make sure we have xhat_k1 (previous frames' a posteri)

            # For each frame that was skipped, step the Kalman filter.
            # Since we have no observation, the estimated error will
            # rise.
            frames_skipped = frame-self.current_frameno-1
            for i in range(frames_skipped):
                xhat, P = self.my_kalman.step()
                ############ save outputs ###############
                self.frames.append( self.current_frameno + i + 1 )
                self.xhats.append( xhat )
                self.Ps.append( P )

        self.current_frameno = frame
        # Step 1.B. Update Kalman to provide a priori estimates for this frame
        xhatminus, Pminus = self.my_kalman.step1__calculate_a_priori()

        # Step 2. Filter incoming 2D data to use informative points
        observation_meters = self._filter_data(xhatminus, Pminus, data_dict)
        
        # Step 3. Incorporate observation to estimate a posteri
        xhat, P = self.my_kalman.step2__calculate_a_posteri(xhatminus, Pminus,
                                                            observation_meters)
        Pmean = numpy.sqrt(numpy.sum([P[i,i] for i in range(3)]))
        
        # XXX Need to test if error for this object has grown beyond a
        # threshold at which it should be terminated.
        if Pmean > self.max_variance_dist_meters:
            self.kill_me = True

        ############ save outputs ###############
        self.frames.append( frame )
        self.xhats.append( xhat )
        self.Ps.append( P )
        
        if observation_meters is not None:
            self.observations_frames.append( frame )
            self.observations_data.append( observation_meters )
        
    def _filter_data(self,xhatminus, Pminus, data_dict):
        """given state estimate, select useful incoming data and make new observation"""
        # 1. For each camera, predict 2D image location and error distance
        
        a_priori_observation_prediction = xhatminus[:3] # equiv. to "dot(self.my_kalman.C,xhatminus)"
        
        variance_estimate = [Pminus[i,i] for i in range(3)] # maybe equiv. to "dot(self.my_kalman.C,Pminus[i,i])"
        variance_estimate_scalar = numpy.sqrt(numpy.sum(variance_estimate)) # put in distance units (meters)
        neg_predicted_3d = -geom.ThreeTuple( a_priori_observation_prediction )
        cam_ids_and_points2d = []
        for cam_id in self.reconstructor_meters.cam_ids:
            if cam_id not in data_dict:
                # no data
                continue
            
            predicted_2d = self.reconstructor_meters.find2d(cam_id,a_priori_observation_prediction)

            # For large numbers of 2d points in data_dict, probably
            # faster to compute 2d image of error ellipsoid and see if
            # data_dict points fall inside that. For now, however,
            # compute distance individually

            candidate_point_list = data_dict[cam_id]
            found_idxs = []
            
            # Use the first acceptable 2d point match as it's probably
            # best from distance-from-mean-image-backgroudn
            # perspective, but remove from further consideration all
            # 2d points that meet consideration critereon.

            match_dist_and_idx = []
            for idx,(pt_undistorted,projected_line_meters) in enumerate(candidate_point_list):
                # find closest distance between projected_line and predicted position for each 2d point
                dist2=projected_line_meters.translate(neg_predicted_3d).dist2()
                dist = numpy.sqrt(dist2)
                if dist<(self.n_sigma_accept*variance_estimate_scalar):
                    # accept point
                    match_dist_and_idx.append( (dist,idx) )
                    found_idxs.append( idx )
            match_dist_and_idx.sort() # sort by distance
            if len(match_dist_and_idx):
                closest_idx = match_dist_and_idx[0][1]
                pt_undistorted = candidate_point_list[closest_idx][0]
                cam_ids_and_points2d.append( (cam_id,(pt_undistorted[0],
                                                      pt_undistorted[1])))
            found_idxs.reverse() # keep indexes OK as we delete them
            for idx in found_idxs:
                del candidate_point_list[idx]
        # Now new_data_dict has just the 2d points we'll use for this reconstruction
        if len(cam_ids_and_points2d)>=2:
            observation_meters = self.reconstructor_meters.find3d( cam_ids_and_points2d )
        else:
            observation_meters = None
        return observation_meters

class Tracker:
    """
    Handle multiple tracked objects using TrackedObject instances.

    This class keeps a list of objects currently being tracked. It
    also keeps a couple other lists for dealing with cases when the
    tracked objects are no longer 'live'.
    
    """
    def __init__(self,
                 reconstructor_meters,
                 scale_factor=None,
                 n_sigma_accept = 3.0, # default: arbitrarily set to 3
                 max_variance_dist_meters = 0.010, # default: allow error to grow to 10 mm before dropping
                 initial_position_covariance_estimate = 1e-6, # default: initial guess 1mm ( (1e-3)**2 meters)
                 initial_acceleration_covariance_estimate = 15, # default: arbitrary initial guess, rather large
                 Q = None,
                 R = None,
                 ):
        """
        
        arguments
        ---------
        reconstructor_meters - reconstructor instance with internal units of meters
        scale_factor - how to convert from arbitrary units (of observations) into meters (e.g. 1e-3 for mm)
        n_sigma_accept - gobble 2D data points that are within this distance from predicted 2D location
        max_variance_dist_meters - estimated error (in meters) to allow growth to before killing tracked object
        initial_position_covariance_estimate -
        initial_acceleration_covariance_estimate -
        Q - process covariance matrix
        R - measurement noise covariance matrix
        
        """
        
        self.reconstructor_meters=reconstructor_meters
        self.live_tracked_objects = []
        self.dead_tracked_objects = [] # save for getting data out
        self.kill_tracker_callbacks = []

        # set values for passing to TrackedObject
        if scale_factor is None:
            print 'WARNING: scale_factor set to 1e-3',__file__
            self.scale_factor = 1e-3
        else:
            self.scale_factor = scale_factor
        self.n_sigma_accept = n_sigma_accept
        self.max_variance_dist_meters = max_variance_dist_meters
        self.initial_position_covariance_estimate = initial_position_covariance_estimate
        self.initial_acceleration_covariance_estimate = initial_acceleration_covariance_estimate
        self.Q = Q
        self.R = R
            
    def gobble_2d_data_and_calculate_a_posteri_estimates(self,frame,data_dict):
        # Allow earlier tracked objects to be greedy and take all the
        # data they want.
        kill_idxs = []
        for idx,tro in enumerate(self.live_tracked_objects):
            tro.gobble_2d_data_and_calculate_a_posteri_estimate(frame,data_dict)
            if tro.kill_me:
                kill_idxs.append( idx )

        # remove tracked objects from live list (when their error grows too large)
        kill_idxs.reverse()
        for kill_idx in kill_idxs:
            tro = self.live_tracked_objects.pop( kill_idx )
            if len(tro.observations_frames)>1:
                # require more than single observation to save
                self.dead_tracked_objects.append( tro )
        self._flush_dead_queue()
    def join_new_obj(self,frame,first_observation_orig_units):
        tro = TrackedObject(self.reconstructor_meters,
                            frame,
                            first_observation_orig_units,
                            scale_factor=self.scale_factor,
                            n_sigma_accept = self.n_sigma_accept,
                            max_variance_dist_meters = self.max_variance_dist_meters,
                            initial_position_covariance_estimate = self.initial_position_covariance_estimate,
                            initial_acceleration_covariance_estimate = self.initial_acceleration_covariance_estimate,
                            Q = self.Q,
                            R = self.R,
                            )
        self.live_tracked_objects.append(tro)
    def kill_all_trackers(self):
        self.dead_tracked_objects.extend( self.live_tracked_objects )
        self.live_tracked_objects = []
        self._flush_dead_queue()
    def set_killed_tracker_callback(self,callback):
        self.kill_tracker_callbacks.append( callback )
        
    def _flush_dead_queue(self):
        while len(self.dead_tracked_objects):
            tro = self.dead_tracked_objects.pop(0)
            for callback in self.kill_tracker_callbacks:
                callback(tro)
