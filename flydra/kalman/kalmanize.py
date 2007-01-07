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
from optparse import OptionParser
import dynamic_models

assert params.A_model_name == 'fixed_accel'

KalmanEstimates = flydra_kalman_utils.KalmanEstimates
FilteredObservations = flydra_kalman_utils.FilteredObservations
convert_format = flydra_kalman_utils.convert_format

def process_frame(reconst_orig_units,tracker,frame,frame_data,camn2cam_id,
                  max_err=500.0, debug=False):
    tracker.gobble_2d_data_and_calculate_a_posteri_estimates(frame,frame_data,camn2cam_id,debug2=debug)

    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    if debug:
        print 'for frame %d: data not gobbled:'%(frame,)
        pprint.pprint(frame_data)
        print
        
    # Convert to format accepted by find_best_3d()
    found_data_dict = convert_format(frame_data,camn2cam_id)
    if len(found_data_dict) < 2:
        # Can't do any 3D math without at least 2 cameras giving good
        # data.
        return
    (this_observation_mm, line3d, cam_ids_used,
     min_mean_dist) = ru.hypothesis_testing_algorithm__find_best_3d(
        reconst_orig_units,
        found_data_dict)
    if debug > 5:
        print 'found new point using hypothesis testing:'
        print 'this_observation_mm',this_observation_mm
        print 'cam_ids_used',cam_ids_used
        print 'min_mean_dist',min_mean_dist
    
    if min_mean_dist<max_err:
        if debug > 5:
            print 'accepting point'
        
        # make mapping from cam_id to camn
        cam_id2camn = {}
        for camn in camn2cam_id:
            if camn not in frame_data:
                continue # this camn not used this frame, ignore
            cam_id = camn2cam_id[camn]
            if cam_id in cam_id2camn:
                print '*'*80
                print """
                
ERROR: It appears that you have >1 camn for a cam_id at a certain
frame. This almost certainly means that you are using a data file
recorded with an older version of flydra.MainBrain and that the
cameras were re-synchronized during the saving of a data file. You
will have to manually find out which camns to ignore (use
flydra_analysis_print_camera_summary) and then use the --exclude-camns
option to this program.

"""
                print '*'*80
                print
                print 'frame',frame
                print 'camn',camn
                print 'frame_data',frame_data
                print
                print 'cam_id2camn',cam_id2camn
                print 'camn2cam_id',camn2cam_id
                print
                raise ValueError('cam_id already in dict')
            cam_id2camn[cam_id]=camn

        # find camns
        this_observation_camns = [cam_id2camn[cam_id] for cam_id in cam_ids_used]
        this_observation_idxs = [0 for camn in this_observation_camns] # zero idx

        if debug>5:
            print 'this_observation_camns',this_observation_camns
            print 'this_observation_idxs',this_observation_idxs

            print 'camn','raw 2d data','reprojected 3d->2d'
            for camn,obs_idx in zip(this_observation_camns,this_observation_idxs):
                cam_id = camn2cam_id[camn]
                repro=reconst_orig_units.find2d( cam_id, this_observation_mm )
                print camn,frame_data[camn][obs_idx][0][:2],repro
        
        ####################################
        #  Now join found point into Tracker
        tracker.join_new_obj( frame,
                              this_observation_mm,
                              this_observation_camns,
                              this_observation_idxs )
    if debug > 5:
        print

class KalmanSaver:
    def __init__(self,dest_filename,reconst_orig_units):
        self.h5file = PT.openFile(dest_filename, mode="w", title="tracked Flydra data file")
        reconst_orig_units.save_to_h5file(self.h5file)
        self.h5_xhat = self.h5file.createTable(self.h5file.root,'kalman_estimates', KalmanEstimates,
                                               "Kalman a posteri estimates of tracked object")
        self.h5_xhat_names = PT.Description(KalmanEstimates().columns)._v_names
        self.h5_obs = self.h5file.createTable(self.h5file.root,'kalman_observations', FilteredObservations,
                                              "observations of tracked object")
        self.h5_obs_names = PT.Description(FilteredObservations().columns)._v_names

        self.h5_2d_obs_next_idx = 0
        self.h5_2d_obs = self.h5file.createVLArray(self.h5file.root,
                                                   'kalman_observations_2d_idxs',
                                                   PT.UInt16Atom(flavor='numpy'), # dtype should match with tro.observations_2d
                                                   "camns and idxs")
        
        self.obj_id = -1
        
    def close(self):
        self.h5file.close()
        
    def save_tro(self,tro):
        MIN_KALMAN_OBSERVATIONS_TO_SAVE = 10
        if len(tro.observations_frames) < MIN_KALMAN_OBSERVATIONS_TO_SAVE:
            # only save data with at least 10 observations
            return
        
        self.obj_id += 1

        # save observation 2d data indexes
        this_idxs = []
        for camns_and_idxs in tro.observations_2d:
            this_idxs.append( self.h5_2d_obs_next_idx )
            self.h5_2d_obs.append( camns_and_idxs )
            self.h5_2d_obs_next_idx += 1
        self.h5_2d_obs.flush()
            
        this_idxs = numpy.array( this_idxs, dtype=numpy.uint64 ) # becomes obs_2d_idx (index into 'kalman_observations_2d_idxs')
        
        # save observations
        observations_frames = numpy.array(tro.observations_frames, dtype=numpy.uint64)
        obj_id_array = numpy.empty(observations_frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(self.obj_id)
        observations_data = numpy.array(tro.observations_data, dtype=numpy.float32)
        list_of_obs = [observations_data[:,i] for i in range(observations_data.shape[1])]
        array_list = [obj_id_array,observations_frames]+list_of_obs+[this_idxs]
        obs_recarray = numpy.rec.fromarrays( array_list, names = self.h5_obs_names)
        
        self.h5_obs.append(obs_recarray)
        self.h5_obs.flush()

        # save xhat info (kalman estimates)
        
        frames = numpy.array(tro.frames, dtype=numpy.uint64)
        xhat_data = numpy.array(tro.xhats, dtype=numpy.float32)
        P_data_full = numpy.array(tro.Ps, dtype=numpy.float32)
        P_data_save = P_data_full[:,numpy.arange(9),numpy.arange(9)] # get diagonal
        obj_id_array = numpy.empty(frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(self.obj_id)
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
              exclude_cam_ids=None,
              exclude_camns=None,
              dynamic_model=None,
              debug=False,
              ):
    if exclude_cam_ids is None:
        exclude_cam_ids = []
        
    if exclude_camns is None:
        exclude_camns = []

    if dynamic_model is None:
        dynamic_model = 'fly dynamics, high precision calibration, units: mm'
    
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
        dest_filename = os.path.splitext(results.filename)[0]+'.kalmanized.h5'
    if os.path.exists(dest_filename):
        raise ValueError('%s already exists, quitting'%dest_filename)
        #os.unlink(dest_filename)
    h5saver = KalmanSaver(dest_filename,reconst_orig_units)

    tracker = Tracker(reconstructor_meters,scale_factor=reconst_orig_units.get_scale_factor())
    model_dict=dynamic_models.get_dynamic_model_dict()
    try:
        kw_dict = model_dict[dynamic_model]
    except KeyError:
        print 'valid model names:',model_dict.keys()
        raise
    for attr in kw_dict:
        setattr(tracker,attr,kw_dict[attr])
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

                if debug > 5:
                    print
                    print 'frame_data for frame %d'%(last_frame,)
                    pprint.pprint(frame_data)
                    print
#                if debug > 5:
#                    for camn,data in frame_data.iteritems():
#                        if len(data)>1:
#                            print '>1'
#                            pprint.pprint(frame_data)
#                            print
                process_frame(reconst_orig_units,tracker,last_frame,frame_data,camn2cam_id,
                              max_err=max_err,debug=debug)
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
        
        if cam_id in exclude_cam_ids:
            # exclude this camera
            continue

        if camn in exclude_camns:
            # exclude this camera
            continue

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
        
        (area,slope,eccentricity,p1,p2,p3,p4,frame_pt_idx) = (row['area'],
                                                              row['slope'],row['eccentricity'],
                                                              row['p1'],row['p2'],
                                                              row['p3'],row['p4'],
                                                              row['frame_pt_idx'])
        if not numpy.isnan(p1):
            line_found = True
        else:
            line_found = False
        pt_undistorted = (x_undistorted,y_undistorted,
                          area,slope,eccentricity,
                          p1,p2,p3,p4, line_found, frame_pt_idx)

        pluecker_hz_meters=reconstructor_meters.get_projected_line_from_2d(
            cam_id,(x_undistorted,y_undistorted))

        projected_line_meters=geom.line_from_HZline(pluecker_hz_meters)

        frame_data.setdefault(camn,[]).append((pt_undistorted,projected_line_meters))

    tracker.kill_all_trackers() # done tracking
    h5saver.close()
    results.close()

def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

    parser.add_option("-r", "--reconstructor", dest="reconstructor_path", type='string',
                      help="calibration/reconstructor path",
                      metavar="RECONSTRUCTOR")

    parser.add_option("--exclude-cam-ids", dest="exclude_cam_ids", type='string',
                      help="camera ids to exclude from reconstruction (space separated)",
                      metavar="EXCLUDE_CAM_IDS")

    parser.add_option("--exclude-camns", dest="exclude_camns", type='string',
                      help="camera numbers to exclude from reconstruction (space separated)",
                      metavar="EXCLUDE_CAMNS")

    parser.add_option("--dynamic-model", dest="dynamic_model", type='string')

    parser.add_option("--start", type="int",
                      help="first frame",
                      metavar="START")
        
    parser.add_option("--stop", type="int",
                      help="last frame",
                      metavar="STOP")

    parser.add_option("--debug", type="int",
                      metavar="DEBUG")

    (options, args) = parser.parse_args()
    if options.exclude_cam_ids is not None:
        options.exclude_cam_ids = options.exclude_cam_ids.split()
        
    if options.exclude_camns is not None:
        options.exclude_camns = [int(camn) for camn in options.exclude_camns.split()]
        
    if options.filename is not None:
        args.append(options.filename)
        
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
        
    if len(args)<1:
        parser.print_help()
        return
    
    src_filename = args[0]
                          
    kalmanize(src_filename,
              reconstructor_filename=options.reconstructor_path,
              start_frame=options.start,
              stop_frame=options.stop,
              exclude_cam_ids=options.exclude_cam_ids,
              exclude_camns=options.exclude_camns,
              dynamic_model = options.dynamic_model,
              debug = options.debug,
              )

if __name__=='__main__':
    main()
