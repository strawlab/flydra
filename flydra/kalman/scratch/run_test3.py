###############################

import kalman
import params
import geom
import math

class TrackedObject:
    """TrackedObject handles all internal units in meters, but external interfaces are mm"""
    def __init__(self,reconstructor_meters,frame,first_observation_mm):
        self.kill_me = False
        self.reconstructor_meters = reconstructor_meters
        self.current_frameno = frame
        self.n_sigma_accept = 3.0 # arbitrary
        self.max_variance_dist_meters = 0.010 # arbitrary, allow error to grow to 10 mm before dropping
        first_observation_meters = first_observation_mm/1000.0
        initial_x = numpy.hstack((first_observation_meters, # convert to mm from meters
                                  (0,0,0, 0,0,0))) # zero velocity and acceleration
        ss = params.A.shape[0]
        P_k1=numpy.eye(ss) # initial state error covariance guess
        for i in range(0,3):
            P_k1[i,i]=(0.001)**2 # initial accuracy 1mm
        for i in range(6,9):
            P_k1[i,i]=15
        self.my_kalman = kalman.KalmanFilter(params.A,
                                             params.C,
                                             params.Q,
                                             params.R,
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
            print '************* Killing tro %s because Pmean = %f (mm)'%(self,Pmean*1000.0)
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
    def __init__(self,reconstructor_meters):
        self.reconstructor_meters=reconstructor_meters
        self.live_tracked_objects = []
        self.dead_tracked_objects = [] # save for getting data out
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
    def join_new_obj(self,frame,first_observation_mm):
        tro = TrackedObject(self.reconstructor_meters,frame,first_observation_mm)
        self.live_tracked_objects.append(tro)
    def kill_all_trackers(self):
        self.dead_tracked_objects.extend( self.live_tracked_objects )
        self.live_tracked_objects = []
        
###############################

import numpy
import copy
import params
import flydra.reconstruct
import flydra.reconstruct_utils as ru
import geom

def convert_format(current_data):
    found_data_dict = {}
    for cam_id, stuff_list in current_data.iteritems():
        if not len(stuff_list):
            # no data for this camera, continue
            continue
        this_point,projected_line = stuff_list[0] # algorithm only accepts 1 point per camera
        if this_point[9]: # only use if found_anything
            found_data_dict[cam_id] = this_point[:9]
    return found_data_dict

from result_utils import get_results, get_f_xyz_L_err, get_caminfo_dicts

try:
    results
except NameError:
    results = get_results('DATA20060719_180955.h5')
reconstructor_mm = flydra.reconstruct.Reconstructor(results)
camn2cam_id, cam_id2camns = get_caminfo_dicts(results)

#frame_range = range(858535,858650)
frame_range = range(858535,859340)
reconstructor_meters = reconstructor_mm.get_scaled(1e-3)

frame_array = numpy.asarray(results.root.data2d_distorted.cols.frame)
tmp_diff = frame_array[1:]-frame_array[:-1]
if numpy.minimum(tmp_diff) < 0:
    raise ValueError("frames not continuously increasing")
1/0

try:
    data2d_struct
except NameError:   
    data2d_struct = {}
    for frame in frame_range:
        if frame%100==0:
            print 'loading',frame
        # get all 2D points for frame
        data2d = results.root.data2d_distorted
        for row in data2d.where( data2d.cols.frame == frame ):
            camn = row['camn']
            cam_id = camn2cam_id[camn]
            x_distorted = row['x']
            if numpy.isnan(x_distorted):
                # drop point -- not found
                continue
            y_distorted = row['y']
            
            (x_undistorted,y_undistorted) = reconstructor_mm.undistort(
                cam_id,(x_distorted,y_distorted))

            area,slope,eccentricity,p1,p2,p3,p4 = (row['area'],
                                                   row['slope'],row['eccentricity'],
                                                   row['p1'],row['p2'],row['p3'],row['p4'])
            pt_undistorted = (x_undistorted,y_undistorted,
                              area,slope,eccentricity,
                              p1,p2,p3,p4, True)

            pluecker_hz_meters=reconstructor_meters.get_projected_line_from_2d(
                cam_id,(x_undistorted,y_undistorted))
            
            projected_line_meters=geom.line_from_HZline(pluecker_hz_meters)

            data2d_struct.setdefault(frame,{}).setdefault(cam_id,[]).append((
                pt_undistorted,projected_line_meters))

tracker = Tracker(reconstructor_meters)

for frame in frame_range:
    if frame not in data2d_struct:
        # no data
        continue
    print 'frame',frame
    
    current_data = copy.deepcopy(data2d_struct[frame])
    
    tracker.gobble_2d_data_and_calculate_a_posteri_estimates(frame,current_data)
    
    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    # Convert to format accepted by find_best_3d()
    found_data_dict = convert_format(current_data)
    if len(found_data_dict) < 2:
        # Can't do any 3D math without at least 2 cameras giving good
        # data.
        continue
    (this_observation_mm, line3d, cam_ids_used,
     min_mean_dist) = ru.find_best_3d(reconstructor_mm,
                                      found_data_dict)
    max_err=10.0 # mm
    if min_mean_dist<max_err:
        ####################################
        #  Now join found point into Tracker
        tracker.join_new_obj( frame, this_observation_mm )

tracker.kill_all_trackers() # done tracking

#############################
import tables as PT
import os

class KalmanEstimates(PT.IsDescription):
    frame      = PT.Int32Col(pos=0,indexed=True)
    x          = PT.Float32Col(pos=1)
    y          = PT.Float32Col(pos=2)
    z          = PT.Float32Col(pos=3)
    xvel       = PT.Float32Col(pos=4)
    yvel       = PT.Float32Col(pos=5)
    zvel       = PT.Float32Col(pos=6)
    xaccel     = PT.Float32Col(pos=7)
    yaccel     = PT.Float32Col(pos=8)
    zaccel     = PT.Float32Col(pos=9)

class FilteredObservations(PT.IsDescription):
    frame      = PT.Int32Col(pos=0,indexed=True)
    x          = PT.Float32Col(pos=1)
    y          = PT.Float32Col(pos=2)
    z          = PT.Float32Col(pos=3)

if 1:
    filename = os.path.splitext(results.filename)[0]+'.tracked.h5'
    if os.path.exists(filename):
        os.unlink(filename)
    h5file = PT.openFile(filename, mode="w", title="tracked Flydra data file")
    ct = h5file.createTable # shorthand
    root = h5file.root # shorthand
    for tro_idx,tro in enumerate(tracker.dead_tracked_objects):

        
        h5_xhat = ct(root,'kalman_%d'%tro_idx, KalmanEstimates, "Kalman a posteri estimates of tracked object",
                     expectedrows=len(tro.xhats))
        tro.frames = numpy.asarray(tro.frames).astype(numpy.int32)
        tro.xhats = numpy.asarray(tro.xhats).astype(numpy.float32)
        list_of_xhats = [tro.xhats[:,col] for col in range(9)]
        names = PT.Description(KalmanEstimates().columns)._v_names
        recarray = numpy.rec.fromarrays([tro.frames]+list_of_xhats,
                                        names = names)
        h5_xhat.append(recarray)
        h5_xhat.flush()

        h5_obs = ct(root,'observations_%d'%tro_idx, FilteredObservations, "observations of tracked object",
                     expectedrows=len(tro.observations_frames))
        tro.observations_frames = numpy.asarray(tro.observations_frames).astype(numpy.int32)
        tro.observations_data = numpy.asarray(tro.observations_data).astype(numpy.float32)
        list_of_obs = [tro.observations_data[:,col] for col in range(3)]
        names = PT.Description(FilteredObservations().columns)._v_names
        recarray = numpy.rec.fromarrays([tro.observations_frames]+list_of_obs,
                                        names = names)
        print recarray.dtype
        print recarray.shape
        print 'formats',PT.Description(FilteredObservations().columns)._v_nestedFormats
        h5_obs.append(recarray)
        h5_obs.flush()
        
    h5file.close()

#############################

if 0:
    import pylab

    colorlist = 'b','g','r','k','c','m'


    pylab.figure()
    ax = None
    varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
    for i in range(ss):
        ax = pylab.subplot(ss,1,i+1,sharex=ax)
        var = varnames[i]
        for tro_idx,tro in enumerate(tracker.dead_tracked_objects):
            color_idx = tro_idx%len(colorlist)
            this_color = colorlist[color_idx]

            if i==0: print 'tro',this_color,tro
            tro.observations_data = numpy.asarray(tro.observations_data)
            if i<os:
                if i==0: print tro.observations_frames[0],'-',tro.observations_frames[-1]
                ax.plot(tro.observations_frames,tro.observations_data[:,i],this_color+'+',label='observations of %s'%var)
            tro.xhats = numpy.asarray(tro.xhats)
            ax.plot(tro.frames,tro.xhats[:,i],this_color+'-',label='estimates of %s'%var)
        pylab.ylabel(var)       
    pylab.xlabel('frame')



    pylab.figure()
    ax = None
    varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
    for i in range(ss):
        ax = pylab.subplot(ss,1,i+1,sharex=ax)
        var = varnames[i]
        for tro_idx,tro in enumerate(tracker.dead_tracked_objects):
            color_idx = tro_idx%len(colorlist)
            this_color = colorlist[color_idx]

            tro.Ps = numpy.asarray(tro.Ps)
            ax.plot(tro.frames,numpy.sqrt(tro.Ps[:,i,i]),this_color+'-',label='estimates of %s'%var)
        pylab.ylabel(var)       
    pylab.xlabel('frame')

    pylab.show()
