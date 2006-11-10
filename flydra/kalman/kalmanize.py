import numpy
import params
import flydra.reconstruct
import flydra.reconstruct_utils as ru
import flydra.geom as geom
import time
from flydra.analysis.result_utils import get_results, get_caminfo_dicts
import tables as PT
import os, sys, pprint
from flydra_tracker import Tracker
import flydra_kalman_utils

assert params.A_model_name == 'fixed_accel'

KalmanEstimates = flydra_kalman_utils.KalmanEstimates
FilteredObservations = flydra_kalman_utils.FilteredObservations
convert_format = flydra_kalman_utils.convert_format

def process_frame(reconst_orig_units,tracker,frame,frame_data,max_err=500.0):
    tracker.gobble_2d_data_and_calculate_a_posteri_estimates(frame,frame_data)

    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    # Convert to format accepted by find_best_3d()
    found_data_dict = convert_format(frame_data)
    if len(found_data_dict) < 2:
        # Can't do any 3D math without at least 2 cameras giving good
        # data.
        return
    (this_observation_mm, line3d, cam_ids_used,
     min_mean_dist) = ru.hypothesis_testing_algorithm__find_best_3d(
        reconst_orig_units,
        found_data_dict)
    if min_mean_dist<max_err:
        ####################################
        #  Now join found point into Tracker
        print 'new point, frame',frame
        tracker.join_new_obj( frame, this_observation_mm )

class KalmanSaver:
    def __init__(self,dest_filename):
        self.h5file = PT.openFile(dest_filename, mode="w", title="tracked Flydra data file")
        self.h5_xhat = self.h5file.createTable(self.h5file.root,'kalman_estimates', KalmanEstimates,
                                               "Kalman a posteri estimates of tracked object")
        self.h5_xhat_names = PT.Description(KalmanEstimates().columns)._v_names
        self.h5_obs = self.h5file.createTable(self.h5file.root,'kalman_observations', FilteredObservations,
                                              "observations of tracked object")
        self.h5_obs_names = PT.Description(FilteredObservations().columns)._v_names
        self.obj_id = -1
        
    def close(self):
        self.h5file.close()
        
    def save_tro(self,tro):
        if not len(tro.xhats):
            # don't save if tracker didn't succeed
            return
        self.obj_id += 1

        # save observations
        
        observations_frames = numpy.array(tro.observations_frames, dtype=numpy.int32)
        observations_data = numpy.array(tro.observations_data, dtype=numpy.float32)
        obj_id_array = self.obj_id * numpy.ones(observations_frames.shape, dtype=numpy.int32)
        list_of_obs = [observations_data[:,i] for i in range(observations_data.shape[1])]
        obs_recarray = numpy.rec.fromarrays([obj_id_array,observations_frames]+list_of_obs,
                                            names = self.h5_obs_names)
        
        self.h5_obs.append(obs_recarray)
        self.h5_obs.flush()

        # save xhat info (kalman estimates)
        
        frames = numpy.array(tro.frames, dtype=numpy.int32)
        xhat_data = numpy.array(tro.xhats, dtype=numpy.float32)
        
        P_data_full = numpy.array(tro.Ps, dtype=numpy.float32)
        P_data_save = P_data_full[:,numpy.arange(9),numpy.arange(9)] # get diagonal
        
        obj_id_array = self.obj_id * numpy.ones(frames.shape, dtype=numpy.int32)
        #print 'xhat_data.shape',xhat_data.shape
        list_of_xhats = [xhat_data[:,i] for i in range(xhat_data.shape[1])]
        list_of_Ps = [P_data_save[:,i] for i in range(P_data_save.shape[1])]
        xhats_recarray = numpy.rec.fromarrays([obj_id_array,frames]+list_of_xhats+list_of_Ps,
                                              names = self.h5_xhat_names)
        
        self.h5_xhat.append(xhats_recarray)
        self.h5_xhat.flush()

def kalmanize(src_filename,
              dest_filename=None,
              reconstructor_filename=None,
              start_frame=None,
              stop_frame=None,
              ):
    results = get_results(src_filename)

    if reconstructor_filename is None:
        reconst_orig_units = flydra.reconstruct.Reconstructor(results)
    else:
        if reconstructor_filename.endswith('h5'):
            fd = PT.openFile(reconstructor_filename,mode='r')
            reconst_orig_units = flydra.reconstruct.Reconstructor(fd)
        else:
            reconst_orig_units = flydra.reconstruct.Reconstructor(reconstructor_filename)
        
    reconstructor_meters = reconst_orig_units.get_scaled(reconst_orig_units.get_scale_factor())
    camn2cam_id, cam_id2camns = get_caminfo_dicts(results)
    
    if dest_filename is None:
        dest_filename = os.path.splitext(results.filename)[0]+'.tracked_fixed_accel.h5'
    if os.path.exists(dest_filename):
        raise ValueError('%s already exists, quitting'%dest_filename)
        #os.unlink(dest_filename)
    h5saver = KalmanSaver(dest_filename)

    tracker = Tracker(reconstructor_meters,scale_factor=reconst_orig_units.get_scale_factor())
    tracker.set_killed_tracker_callback( h5saver.save_tro )
    
    data2d = results.root.data2d_distorted

    done_frames = []

    time1 = time.time()
    print 'loading all frame numbers...'
    frames_array = data2d.read(field='frame',flavor='numpy')
    time2 = time.time()
    print 'done in %.1f sec'%(time2-time1)
    row_idxs = numpy.argsort(frames_array)
    
    print '2D data range: %d<frame<%d'%(frames_array[row_idxs[0]], frames_array[row_idxs[-1]])
    
    if 0:
        print '-='*40
        print '-='*40
        print 'using only first 2000 rows'
        row_idxs = row_idxs[:2000]
        print '-='*40
        print '-='*40

    max_err = 500.0
    print 'max error',max_err
    frame_count = 0
    accum_time = 0.0
    last_frame = None
    frame_data = {}
    time1 = time.time()
    for row_idx in row_idxs:
        new_frame = frames_array[row_idx]
        if start_frame is not None:
            if new_frame < start_frame:
                continue
        if stop_frame is not None:
            if new_frame > stop_frame:
                continue
        time3 = time.time()
        row = data2d[row_idx]
        time4 = time.time()
        accum_time += (time4-time3)
        new_frame_test_cmp = row['frame']
        assert new_frame_test_cmp==new_frame
        if last_frame != new_frame:
            if new_frame < last_frame:
                print 'new_frame',new_frame
                print 'last_frame',last_frame
                raise RuntimeError("expected continuously increasing frame numbers")
            # new frame
            ########################################
            # Data for this frame is complete
            if last_frame is not None:

                if 0:
                    print
                    print 'frame_data'
                    pprint.pprint(frame_data)
                    print
                if 0:
                    for cam_id,data in frame_data.iteritems():
                        if len(data)>1:
                            print '>1'
                            pprint.pprint(frame_data)
                            print
                print 'processing frame',last_frame
                process_frame(reconst_orig_units,tracker,last_frame,frame_data,max_err=max_err)
                frame_count += 1
                if frame_count%1000==0:
                    time2 = time.time()
                    dur = time2-time1
                    fps = frame_count/dur
                    dur2 = dur-accum_time
                    fps2 = frame_count/dur2
                    print 'frame % 10d, mean speed so far: %.1f fps (%.1f fps without pytables)'%(last_frame,fps,fps2)

            ########################################
            frame_data = {}
            last_frame = new_frame

        camn = row['camn']
        cam_id = camn2cam_id[camn]

        x_distorted = row['x']
        if numpy.isnan(x_distorted):
            # drop point -- not found
            continue
        y_distorted = row['y']

        (x_undistorted,y_undistorted) = reconst_orig_units.undistort(
            cam_id,(x_distorted,y_distorted))
        if 1:
            (x_undistorted_m,y_undistorted_m) = reconstructor_meters.undistort(
                cam_id,(x_distorted,y_distorted))
            if x_undistorted != x_undistorted_m:
                raise ValueError('scaled reconstructors have different distortion!?')
            if y_undistorted != y_undistorted_m:
                raise ValueError('scaled reconstructors have different distortion!?')
        
        area,slope,eccentricity,p1,p2,p3,p4 = (row['area'],
                                               row['slope'],row['eccentricity'],
                                               row['p1'],row['p2'],row['p3'],row['p4'])
        if not numpy.isnan(p1):
            line_found = True
        else:
            line_found = False
        pt_undistorted = (x_undistorted,y_undistorted,
                          area,slope,eccentricity,
                          p1,p2,p3,p4, line_found)

        pluecker_hz_meters=reconstructor_meters.get_projected_line_from_2d(
            cam_id,(x_undistorted,y_undistorted))

        projected_line_meters=geom.line_from_HZline(pluecker_hz_meters)

        frame_data.setdefault(cam_id,[]).append((pt_undistorted,projected_line_meters))

    tracker.kill_all_trackers() # done tracking
    h5saver.close()
    results.close()

def main():
    src_filename = sys.argv[1]
    if len(sys.argv)>2:
        reconstructor_filename = sys.argv[2]
    else:
        reconstructor_filename = None
    if len(sys.argv)>3:
        start_frame = int(sys.argv[3])
    else:
        start_frame = None
        
    if len(sys.argv)>4:
        stop_frame = int(sys.argv[4])
    else:
        stop_frame = None
                          
    kalmanize(src_filename,
              reconstructor_filename=reconstructor_filename,
              start_frame=start_frame,
              stop_frame=stop_frame)

if __name__=='__main__':
    main()
