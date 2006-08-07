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
        self.max_variance_dist_meters = 0.004 # arbitrary, allow error to grow to 4 mm before dropping
        first_observation_meters = first_observation_mm/1000.0
        initial_x = numpy.hstack((first_observation_meters, # convert to mm from meters
                                  (0,0,0, 0,0,0))) # zero velocity and acceleration
        ss = params.A.shape[0]
        P_k1=numpy.eye(ss) # initial state error covariance guess
        for i in range(3):
            P_k1[i,i]=(0.001)**2 # initial accuracy 1mm
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
                self.xhats.Ps( P )

        self.current_frameno = frame
        # Step 1.B. Update Kalman to provide a priori estimates for this frame
        xhatminus, Pminus = self.my_kalman.step1__calculate_a_priori()

        # Step 2. Filter incoming 2D data to use informative points
        observation_meters = self._filter_data(xhatminus, Pminus, data_dict)
        
        # Step 3. Incorporate observation to estimate a posteri
        print 'CALLING my_kalman with observation_meters =',observation_meters
        xhat, P = self.my_kalman.step2__calculate_a_posteri(xhatminus, Pminus,
                                                            observation_meters)
        print 'xhat',xhat
        print 'P',
        for i in range(3):
            print math.sqrt(P[i,i]),
        Pmean = numpy.sqrt(numpy.sum([P[i,i] for i in range(3)]))
        print 'Pmean (mm)',Pmean*1000.0 # in mm
        print
        
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
        print 'a_priori_observation_prediction',a_priori_observation_prediction
        
        variance_estimate = [Pminus[i,i] for i in range(3)] # maybe equiv. to "dot(self.my_kalman.C,Pminus[i,i])"
        variance_estimate_scalar = numpy.sqrt(numpy.sum(variance_estimate)) # put in distance**2 units (meters)
        predicted_3d = geom.ThreeTuple( a_priori_observation_prediction )
        print 'predicted_3d',predicted_3d
        print 'variance_estimate',variance_estimate
        print 'variance_estimate_scalar',variance_estimate_scalar
        cam_ids_and_points2d = []
        for cam_id in self.reconstructor_meters.cam_ids:
            if cam_id not in data_dict:
                # no data
                continue
            
            predicted_2d = self.reconstructor_meters.find2d(cam_id,a_priori_observation_prediction)
            print cam_id,'predicted_2d:',predicted_2d

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
                dist2=projected_line_meters.translate(-predicted_3d).dist2()
                dist = numpy.sqrt(dist2)
                if dist<(self.n_sigma_accept*variance_estimate_scalar):
                    # accept point
                    match_dist_and_idx.append( (dist,idx) )
                    found_idxs.append( idx )
                print ' found 2d:',pt_undistorted[:2]
                #print '  projected line:',projected_line_meters
                print '  distance from prediction:',math.sqrt(dist2)
                print '  n_sigma:',dist/variance_estimate_scalar
                # do we accept this point?
                if idx in found_idxs:
                    print '  ACCEPTED'
                else:
                    print '  rejected'
            match_dist_and_idx.sort() # sort by distance
            if len(match_dist_and_idx):
                closest_idx = match_dist_and_idx[0][1]
                pt_undistorted = candidate_point_list[closest_idx][0]
                cam_ids_and_points2d.append( (cam_id,(pt_undistorted[0],
                                                      pt_undistorted[1])))
            found_idxs.reverse() # keep indexes OK as we delete them
            for idx in found_idxs:
                print 'deleting',idx
                del candidate_point_list[idx]
            print
        # Now new_data_dict has just the 2d points we'll use for this reconstruction
        print 'USING FOR RECONSTRUCTION:'
        pprint.pprint( cam_ids_and_points2d )
        if len(cam_ids_and_points2d)>=2:
            observation_meters = self.reconstructor_meters.find3d( cam_ids_and_points2d )
        else:
            print 'NOT ENOUGH DATA FOR RECONSTRUCTION'
            observation_meters = None
        print 'observation (in meters):',observation_meters
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
            self.dead_tracked_objects.append( self.live_tracked_objects.pop( kill_idx ) )
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
import pprint
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

A=params.A
C=params.C
Q=params.Q
R=params.R

ss = A.shape[0]
os = C.shape[0]

# initial state error covariance guess
P_k1=numpy.eye(ss)

from result_utils import get_results, get_f_xyz_L_err, get_caminfo_dicts

try:
    results
except NameError:
    results = get_results('DATA20060719_180955.h5')
reconstructor_mm = flydra.reconstruct.Reconstructor(results)
camn2cam_id, cam_id2camns = get_caminfo_dicts(results)

# get "original data" from flydra's hypothesis testing algorithm
try:
    frames_orig3d,y_mm,L,err
except NameError:
    max_err=10.0
    typ='best'
    frames_orig3d,y_mm,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)
    del max_err
    del typ
    
#frame_range = range(858535,858650)
frame_range = range(858535,859340)
reconstructor_meters = reconstructor_mm.get_scaled(1e-3)

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

##            tmpA=reconstructor_meters.find3d_single_cam(cam_id,(x_undistorted,y_undistorted,1.0))
##            tmpB=reconstructor_meters.find3d_single_cam(cam_id,(x_undistorted,y_undistorted,2.0))
##            tmpC=reconstructor_meters.get_camera_center(cam_id)

##            print 'projection:',cam_id,x_undistorted,y_undistorted
##            print '  A',tmpA
##            print '  B',tmpB
##            print '  C',tmpC
            
            pluecker_hz_meters=reconstructor_meters.get_projected_line_from_2d(cam_id,(x_undistorted,y_undistorted))
            
            projected_line_meters=geom.line_from_HZline(pluecker_hz_meters)

##            # put in 3D coords from homogeneous 4D
##            tmpA2 = tmpA[:3,0]/tmpA[3,0]
##            tmpB2 = tmpB[:3,0]/tmpB[3,0]
##            tmpC2 = tmpC[:3,0]
##            print '  A2',tmpA2
##            print '  B2',tmpB2
##            print '  C2',tmpC2


##            tmp_line = geom.line_from_points(geom.ThreeTuple(tmpA2),
##                                             geom.ThreeTuple(tmpB2))

##            tmp_line2 = geom.line_from_points(geom.ThreeTuple(tmpA2),
##                                              geom.ThreeTuple(tmpC2))
            
##            tmp_line3 = geom.line_from_points(geom.ThreeTuple(tmpB2),
##                                              geom.ThreeTuple(tmpC2))

##            print 'projected_line_meters',projected_line_meters
##            print 'tmp_line',tmp_line
##            print 'tmp_line.closest()',tmp_line.closest()
##            print 'tmp_line2',tmp_line2
##            print 'tmp_line2.closest()',tmp_line2.closest()
##            print 'tmp_line3',tmp_line3
##            print 'tmp_line3.closest()',tmp_line3.closest()
##            print
##            projected_line_meters=tmp_line2
            data2d_struct.setdefault(frame,{}).setdefault(cam_id,[]).append((
                pt_undistorted,projected_line_meters))

y = y_mm/1000.0 # put in meters (from millimeters, mm)
xhat_k1 = numpy.hstack((y[0,:],(0,0,0, 0,0,0)))

tracker = Tracker(reconstructor_meters)

xhats = []
Ps = []
y_show = []
for last_k,frame in enumerate(frame_range):
    if frame not in data2d_struct:
        # no data
        continue
    print 'frame',frame
    print
    print
    
    current_data = copy.deepcopy(data2d_struct[frame])
    
    tracker.gobble_2d_data_and_calculate_a_posteri_estimates(frame,current_data)
    
    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    if 1:
        # print original data for comparison
        test_cond = frames_orig3d==frame
        test_cond_nz = numpy.nonzero(test_cond)

        if len(test_cond_nz):
            # we have observation
            k = test_cond_nz[0]
            this_y = y[k,:]
            print 'OLD calc:',this_y*1000.0 # put in mm
        else:
            # no observation
            this_y = None
            print 'no data for frame',frame

        data3d = results.root.data3d_best
        for row in data3d.where(data3d.cols.frame==frame):
            print row

    # Convert to format accepted by find_best_3d()
    found_data_dict = convert_format(current_data)
    print 'data that survived gobbling by tracked objects:'
    pprint.pprint(found_data_dict)
    if len(found_data_dict) < 2:
        # Can't do any 3D math without at least 2 cameras giving good
        # data.
        print 'no more data to attempt hypothesis testing algorithm'
        print
        continue
    (this_observation_mm, line3d, cam_ids_used,
     min_mean_dist) = ru.find_best_3d(reconstructor_mm,
                                      found_data_dict)
    print 'THIS calc mm:',this_observation_mm
    max_err=10.0 # mm
    if min_mean_dist<max_err:
        print 'creating Kalman tracker'

        ####################################
        #  Now join found point into Tracker
        tracker.join_new_obj( frame, this_observation_mm )
        
    #pprint.pprint(current_data)

    print
##    if last_k>=3:
##        break

##    xhat,P = kalman_state.step(this_y,return_error_estimate=True)
##    xhats.append(xhat)
##    Ps.append(P)
    
##    if this_y is None:
##        this_y = numpy.nan*numpy.ones((os,))
##    y_show.append(this_y)


tracker.kill_all_trackers() # done tracking

#############################

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

        print 'tro',this_color,tro
##        print len(tro.observations_data)
        tro.observations_data = numpy.asarray(tro.observations_data)
        if i<os:
##            print 'i',i
            print len(tro.observations_frames)
##            print tro.observations_data.shape
            if len(tro.observations_frames):
                print tro.observations_frames[0],'-',tro.observations_frames[-1]
                ax.plot(tro.observations_frames,tro.observations_data[:,i],this_color+'+',label='observations of %s'%var)
            else:
                print 'not plotting observations -- there are none!'
        print
        tro.xhats = numpy.asarray(tro.xhats)
        ax.plot(tro.frames,tro.xhats[:,i],this_color+'-',label='estimates of %s'%var)
    pylab.ylabel(var)       
pylab.xlabel('frame')

pylab.show()


#############################################################

##xhats = numpy.asarray(xhats)
##Ps = numpy.asarray(Ps)
##y_show = numpy.asarray(y_show)

##import pylab

##pylab.figure()
##ax = None
##varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
##for i in range(ss):
##    ax = pylab.subplot(ss,1,i+1,sharex=ax)
##    var = varnames[i]
##    if i<os:
##        ax.plot(frame_range,y_show[:,i],'k+',label='noisy measurements of %s'%var)
##    ax.plot(frame_range,xhats[:,i],'b-',label='a posteri estimate of %s'%var)
##    pylab.ylabel(var)
    
##pylab.figure()
##ax = None
##for i in range(ss):
##    ax = pylab.subplot(ss,1,i+1,sharex=ax)
##    var = varnames[i]
##    ax.plot(frame_range,Ps[:,i,i],'b-',label='a posteri estimate of variance of %s'%var)


##    ax.legend()
##pylab.xlabel('frame')

##pylab.show()
