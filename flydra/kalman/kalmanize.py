import numpy
import flydra.reconstruct
import flydra.reconstruct_utils as ru
#import flydra.geom as geom
import flydra.fastgeom as geom
import time, math
from flydra.analysis.result_utils import get_results, get_caminfo_dicts, \
     get_resolution, get_fps
import tables as PT
import os, sys, pprint
from flydra_tracker import Tracker
import flydra_kalman_utils
from optparse import OptionParser
import dynamic_models
import flydra.save_calibration_data as save_calibration_data
import collections
from flydra.MainBrain import TextLogDescription

KalmanEstimates = flydra_kalman_utils.KalmanEstimates
KalmanEstimatesVelOnly = flydra_kalman_utils.KalmanEstimatesVelOnly

FilteredObservations = flydra_kalman_utils.FilteredObservations
convert_format = flydra_kalman_utils.convert_format
kalman_observations_2d_idxs_type = flydra_kalman_utils.kalman_observations_2d_idxs_type

class FakeThreadingEvent:
    def __init__(self):
        self._set = False
    def set(self):
        self._set = True
    def isSet(self):
        return self._set
    def clear(self):
        self._set = False

def process_frame(reconst_orig_units,tracker,frame,frame_data,camn2cam_id,
                  max_err=None, debug=0, kalman_model=None, area_threshold=0):
    if debug is None:
        debug=0
    frame_data = tracker.calculate_a_posteri_estimates(frame,frame_data,camn2cam_id,debug2=debug)

    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    if debug>1:
        print 'for frame %d: data not gobbled:'%(frame,)
        pprint.pprint(dict(frame_data))
        print

    # Convert to format accepted by find_best_3d()
    found_data_dict,first_idx_by_camn = convert_format(frame_data,
                                                       camn2cam_id,
                                                       area_threshold=area_threshold)

    hypothesis_test_found_point = False
    # test to short-circuit rest of function
    if len(found_data_dict) >= 2:

        # Can only do 3D math with at least 2 cameras giving good
        # data.
        try:
            (this_observation_mm, line3d, cam_ids_used,
             min_mean_dist) = ru.hypothesis_testing_algorithm__find_best_3d(
                reconst_orig_units,
                found_data_dict,
                max_err,
                debug=debug)
        except ru.NoAcceptablePointFound, err:
            pass
        else:
            hypothesis_test_found_point = True

    if hypothesis_test_found_point:

        if debug > 5:
            print 'found new point using hypothesis testing:'
            print 'this_observation_mm',this_observation_mm
            print 'cam_ids_used',cam_ids_used
            print 'min_mean_dist',min_mean_dist

        believably_new = tracker.is_believably_new( this_observation_mm, debug=debug)
        if (debug > 5):
                print 'believably_new',believably_new

        if believably_new:
            assert min_mean_dist<max_err
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

            this_observation_idxs = [first_idx_by_camn[camn] for camn in this_observation_camns] # zero idx

            if debug>5:
                print 'this_observation_camns',this_observation_camns
                print 'this_observation_idxs',this_observation_idxs

                print 'camn','raw 2d data','reprojected 3d->2d'
                for camn in this_observation_camns:
                    cam_id = camn2cam_id[camn]
                    repro=reconst_orig_units.find2d( cam_id, this_observation_mm )
                    print camn,frame_data[camn][0][0][:2],repro

            ####################################
            #  Now join found point into Tracker
            tracker.join_new_obj( frame,
                                  this_observation_mm,
                                  this_observation_camns,
                                  this_observation_idxs,
                                  debug=debug,
                                  )
    if debug > 5:
        print
        print 'At end of frame %d, all live tracked objects:'%frame
        for tro in tracker.live_tracked_objects:
            print tro
            for i in range(len(tro.xhats)):
                this_Pmean = math.sqrt(tro.Ps[i][0,0]**2 + tro.Ps[i][1,1]**2 + tro.Ps[i][2,2]**2)
                print '  ',i,tro.frames[i],tro.xhats[i][:3],this_Pmean,
                if tro.frames[i] in tro.observations_frames:
                    j =  tro.observations_frames.index(  tro.frames[i] )
                    print tro.observations_data[j]
                else:
                    print
            print
        print
        print '-'*80
    elif debug > 2:
        print 'At end of frame %d, all live tracked objects:'%frame
        for tro in tracker.live_tracked_objects:
            print '%d observations, %d estimates for %s'%(len(tro.xhats),len(tro.observations_data),tro)
        print

class KalmanSaver:
    def __init__(self,
                 dest_filename,
                 reconst_orig_units,
                 save_cal_dir=None,
                 cam_id2camns=None,
                 min_observations_to_save=0,
                 textlog_save_lines = None,
                 dynamic_model_name=None,
                 dynamic_model=None,
                 debug=0):
        self.cam_id2camns = cam_id2camns
        self.min_observations_to_save = min_observations_to_save
        self.debug = 0

        if save_cal_dir is not None:
            if 0:
                raise NotImplementedError("this code path is not known to work!")

            assert cam_id2camns is not None
            if os.path.exists(save_cal_dir):
                raise RuntimeError('save_cal_dir exists')
            os.mkdir(save_cal_dir)
        self.save_cal_dir = save_cal_dir

        func = flydra.kalman.flydra_kalman_utils.get_kalman_estimates_table_description_for_model_name
        kalman_estimates_description = func(name=dynamic_model_name)

        if os.path.exists(dest_filename):
            self.h5file = PT.openFile(dest_filename, mode="r+")
            test_reconst = flydra.reconstruct.Reconstructor(self.h5file)
            assert test_reconst == reconst_orig_units

            self.h5_xhat = self.h5file.root.kalman_estimates
            self.h5_obs = self.h5file.root.kalman_observations

            obj_ids = self.h5_xhat.read(field='obj_id')
            self.obj_id = obj_ids.max()
            del obj_ids


            self.h5_2d_obs = self.h5file.root.kalman_observations_2d_idxs
            self.h5_2d_obs_next_idx = len(self.h5_2d_obs)

            self.h5textlog = self.h5file.root.textlog

        else:
            self.h5file = PT.openFile(dest_filename, mode="w", title="tracked Flydra data file")
            reconst_orig_units.save_to_h5file(self.h5file)
            self.h5_xhat = self.h5file.createTable(self.h5file.root,'kalman_estimates', kalman_estimates_description,
                                                   "Kalman a posteri estimates of tracked object")
            self.h5_xhat.attrs.dynamic_model_name = dynamic_model_name
            self.h5_xhat.attrs.dynamic_model = dynamic_model

            self.h5_obs = self.h5file.createTable(self.h5file.root,'kalman_observations', FilteredObservations,
                                                  "observations of tracked object")

            self.h5_2d_obs_next_idx = 0
            self.h5_2d_obs = self.h5file.createVLArray(self.h5file.root,
                                                       'kalman_observations_2d_idxs',
                                                       kalman_observations_2d_idxs_type(), # dtype should match with tro.observations_2d
                                                       "camns and idxs")

            self.obj_id = -1

            self.h5textlog = self.h5file.createTable(self.h5file.root,'textlog',TextLogDescription,'text log')

        if 1:
            textlog_row = self.h5textlog.row
            cam_id = 'mainbrain'
            timestamp = time.time()

            list_of_textlog_data = [ (timestamp,cam_id,timestamp,text) for text in textlog_save_lines ]
            for textlog_data in list_of_textlog_data:
                (mainbrain_timestamp,cam_id,host_timestamp,message) = textlog_data
                textlog_row['mainbrain_timestamp'] = mainbrain_timestamp
                textlog_row['cam_id'] = cam_id
                textlog_row['host_timestamp'] = host_timestamp
                textlog_row['message'] = message
                textlog_row.append()

            self.h5textlog.flush()

        self.h5_xhat_names = PT.Description(kalman_estimates_description().columns)._v_names
        self.h5_obs_names = PT.Description(FilteredObservations().columns)._v_names
        self.all_kalman_calibration_data = []

    def close(self):
        if self.save_cal_dir is not None:
            self._save_kalman_calibration_data()
        self.h5file.close()

    def save_tro(self,tro):
        if len(tro.observations_frames) < self.min_observations_to_save:
            # not enough data to bother saving
            return

        self.obj_id += 1

        if self.debug:
            print 'saving %s as obj_id %d'%(repr(self), obj_id)

        # save observation 2d data indexes
        debugADS=False

        if debugADS:
            print '2D indices: ----------------'

        this_idxs = []
        for camns_and_idxs in tro.observations_2d:
            this_idxs.append( self.h5_2d_obs_next_idx )
            self.h5_2d_obs.append( camns_and_idxs )

            if debugADS:
                print ' %d: %s'%(self.h5_2d_obs_next_idx,str(camns_and_idxs))
            self.h5_2d_obs_next_idx += 1
        self.h5_2d_obs.flush()

        if debugADS:
            print

        this_idxs = numpy.array( this_idxs, dtype=numpy.uint64 ) # becomes obs_2d_idx (index into 'kalman_observations_2d_idxs')

        # save observations ####################################
        observations_frames = numpy.array(tro.observations_frames, dtype=numpy.uint64)
        obj_id_array = numpy.empty(observations_frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(self.obj_id)
        observations_data = numpy.array(tro.observations_data, dtype=numpy.float32)
        list_of_obs = [observations_data[:,i] for i in range(observations_data.shape[1])]
        array_list = [obj_id_array,observations_frames]+list_of_obs+[this_idxs]
        obs_recarray = numpy.rec.fromarrays( array_list, names = self.h5_obs_names)
        if 1:
            # End tracking at last non-nan observation (must be > 1 camera for final points).
            idx = numpy.nonzero(~numpy.isnan(observations_data))[0][-1]
            last_observation_frame = observations_frames[idx]
        else:
            # End tracking at last observation (can be 1 camera for final points).
            last_observation_frame = observations_frames[-1]

        if debugADS:
            print 'kalman observations: --------------'
            for row in obs_recarray:
                print row['frame'], row['obs_2d_idx']

        self.h5_obs.append(obs_recarray)
        self.h5_obs.flush()

        # save xhat info (kalman estimates) ##################

        frames = numpy.array(tro.frames, dtype=numpy.uint64)
        xhat_data = numpy.array(tro.xhats, dtype=numpy.float32)
        timestamps = numpy.array(tro.timestamps, dtype=numpy.float64)
        P_data_full = numpy.array(tro.Ps, dtype=numpy.float32)

        # don't guess after last observation
        cond = frames <= last_observation_frame
        frames = frames[cond]
        xhat_data = xhat_data[cond]
        timestamps = timestamps[cond]
        P_data_full = P_data_full[cond]

        ss = P_data_full.shape[1] # state vector size
        P_data_save = P_data_full[:,numpy.arange(ss),numpy.arange(ss)] # get diagonal

        obj_id_array = numpy.empty(frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(self.obj_id)

        list_of_xhats = [xhat_data[:,i] for i in range(xhat_data.shape[1])]
        list_of_Ps = [P_data_save[:,i] for i in range(P_data_save.shape[1])]

        xhats_recarray = numpy.rec.fromarrays([obj_id_array,frames,timestamps]+list_of_xhats+list_of_Ps,
                                              names = self.h5_xhat_names)

        self.h5_xhat.append(xhats_recarray)
        self.h5_xhat.flush()

        # calibration data
        self.all_kalman_calibration_data.extend( tro.saved_calibration_data )
        if self.save_cal_dir is not None:

            # re-save calibration data after every increment...
            self._save_kalman_calibration_data()

    def _save_kalman_calibration_data(self):
        data_to_save = self.all_kalman_calibration_data
        cam_ids = self.cam_id2camns.keys()
        cam_ids.sort()

        Res = []
        for cam_id in cam_ids:
            width,height = get_resolution(self.h5file, cam_id)
            Res.append( [width,height] )
        save_calibration_data.do_save_calibration_data(
            self.save_cal_dir, cam_ids, data_to_save, Res)

def kalmanize(src_filename,
              dest_filename=None,
              reconstructor_filename=None,
              start_frame=None,
              stop_frame=None,
              exclude_cam_ids=None,
              exclude_camns=None,
              dynamic_model_name=None,
              save_cal_dir=None,
              debug=False,
              frames_per_second=None,
              max_err=None,
              area_threshold=0,
              min_observations_to_save=0,
              ):

    if debug:
        numpy.set_printoptions(precision=3,linewidth=120,suppress=False)

    if exclude_cam_ids is None:
        exclude_cam_ids = []

    if exclude_camns is None:
        exclude_camns = []

    if dynamic_model_name is None:
        dynamic_model_name = 'fly dynamics, high precision calibration, units: mm'
        import warnings
        warnings.warn('dynamic model not specified. using "%s"'%dynamic_model_name)
    else:
        print 'using dynamic model "%s"'%dynamic_model_name

    results = get_results(src_filename)

    if reconstructor_filename is None:
        reconstructor_filename = src_filename

    if 1:
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
            raise ValueError('%s already exists and not explicitly requesting append with "--dest-file" option, quitting'%dest_filename)

    if frames_per_second is None:
        frames_per_second = get_fps(results)
        print 'read frames_per_second from file', frames_per_second

    textlog_save_lines = [
        'kalmanize running at %s fps, (hypothesis_test_max_error %s)'%(str(frames_per_second),str(max_err)),
        'original file: %s'%(src_filename,),
        'dynamic model: %s'%(dynamic_model_name,),
        'reconstructor file: %s'%(reconstructor_filename,),
        ]

    dt = 1.0/frames_per_second
    kalman_model = dynamic_models.get_kalman_model( name=dynamic_model_name, dt=dt )

    h5saver = KalmanSaver(dest_filename,
                          reconst_orig_units,
                          save_cal_dir=save_cal_dir,
                          cam_id2camns=cam_id2camns,
                          min_observations_to_save=min_observations_to_save,
                          textlog_save_lines=textlog_save_lines,
                          dynamic_model_name=dynamic_model_name,
                          dynamic_model=kalman_model,
                          debug=debug)

    save_calibration_data = FakeThreadingEvent()
    if save_cal_dir is not None:
        save_calibration_data.set()

    tracker = Tracker(reconstructor_meters,
                      scale_factor=reconst_orig_units.get_scale_factor(),
                      save_calibration_data=save_calibration_data,
                      kalman_model=kalman_model,
                      save_all_data=True,
                      area_threshold=area_threshold,
                      )

    tracker.set_killed_tracker_callback( h5saver.save_tro )

    data2d = results.root.data2d_distorted

    done_frames = []

    time1 = time.time()
    print 'loading all frame numbers...'
    frames_array = numpy.asarray(data2d.read(field='frame'))
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

    print 'max reprojection error to accept new 3D point with hypothesis testing: %.1f (pixels)'%(max_err,)
    frame_count = 0
    accum_time = 0.0
    last_frame = None
    frame_data = {}
    time1 = time.time()
    RAM_HEAVY_BUT_FAST=True
    if RAM_HEAVY_BUT_FAST:
        data2d_recarray = data2d[:] # needs lots of RAM for big data files
    for row_idx in row_idxs:
        new_frame = frames_array[row_idx]
        if start_frame is not None:
            if new_frame < start_frame:
                continue
        if stop_frame is not None:
            if new_frame > stop_frame:
                continue
        if RAM_HEAVY_BUT_FAST:
            row = data2d_recarray[row_idx]
        else:
            row = data2d[row_idx]
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
                    pprint.pprint(dict(frame_data))
                    print
                process_frame(reconst_orig_units,tracker,last_frame,frame_data,camn2cam_id,
                              max_err=max_err,debug=debug, kalman_model=kalman_model, area_threshold=area_threshold)
                frame_count += 1
                if frame_count%1000==0:
                    time2 = time.time()
                    dur = time2-time1
                    fps = frame_count/dur
                    print 'frame % 10d, mean speed so far: %.1f fps'%(last_frame,fps)

            ########################################
            frame_data = collections.defaultdict(list)
            last_frame = new_frame

        camn = row['camn']
        try:
            cam_id = camn2cam_id[camn]
        except KeyError, err:
            # This will happen if cameras were re-synchronized (and
            # thus gain new cam_ids) immediately before saving was
            # turned on in MainBrain. The reason is that the network
            # buffers are still full of old data coming in from the
            # cameras.
            print 'WARNING: no cam_id for camn %d, skipping this row of data'%camn
            continue

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
        if 0:
            (x_undistorted_m,y_undistorted_m) = reconstructor_meters.undistort(
                cam_id,(x_distorted,y_distorted))
            if x_undistorted != x_undistorted_m:
                raise ValueError('scaled reconstructors have different distortion!?')
            if y_undistorted != y_undistorted_m:
                raise ValueError('scaled reconstructors have different distortion!?')

        (area,slope,eccentricity,frame_pt_idx) = (row['area'],
                                                  row['slope'],row['eccentricity'],
                                                  row['frame_pt_idx'])

        try:
            # new columns added to data2d_distorted format.
            cur_val = row['cur_val']
            mean_val = row['mean_val']
            sumsqf_val = row['sumsqf_val']
        except IndexError, err:
            import warnings
            warnings.warn('ignoring IndexError because your 2D does not have expected column',RuntimeWarning)
            # Don't fail if these columns don't exist.
            cur_val = 0
            mean_val = 0
            sumsqf_val = 0

        # XXX for now, do not calculate 3D plane for each point. This
        # is because we are punting on calculating p1,p2,p3,p4 from
        # the point, slope, and reconstructor.

        line_found = False
        p1, p2, p3, p4 = numpy.nan, numpy.nan, numpy.nan, numpy.nan

        # Keep in sync with kalmanize.py and data_descriptions.py
        pt_undistorted = (x_undistorted,y_undistorted,
                          area,slope,eccentricity,
                          p1,p2,p3,p4, line_found, frame_pt_idx, cur_val, mean_val, sumsqf_val)

        pluecker_hz_meters=reconstructor_meters.get_projected_line_from_2d(
            cam_id,(x_undistorted,y_undistorted))

        projected_line_meters=geom.line_from_HZline(pluecker_hz_meters)

        frame_data[camn].append((pt_undistorted,projected_line_meters))

    tracker.kill_all_trackers() # done tracking

    h5saver.close()
    results.close()

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)

    parser.add_option("-d", "--dest-file", dest="dest_filename", type='string',
                      help="save to hdf5 file (append if already present)",
                      metavar="DESTFILE")

    parser.add_option("-r", "--reconstructor", dest="reconstructor_path", type='string',
                      help="calibration/reconstructor path (if not specified, defaults to FILE)",
                      metavar="RECONSTRUCTOR")

    parser.add_option("--save-cal-dir", type='string',
                      help="directory name in which to save new calibration data",
                      default=None,
                      )

    parser.add_option("--fps", dest='fps', type='float',
                      help="frames per second (used for Kalman filtering)")

    parser.add_option("--max-err", type='float',
                      default=50.0,
                      help="maximum mean reprojection error for hypothesis testing algorithm")

    parser.add_option("--exclude-cam-ids", type='string',
                      help="camera ids to exclude from reconstruction (space separated)",
                      metavar="EXCLUDE_CAM_IDS")

    parser.add_option("--exclude-camns", type='string',
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

    parser.add_option("--area-threshold", type='float',
                      default=0.0,
                      help="area threshold (used to filter incoming 2d points)")

    parser.add_option("--min-observations-to-save", type='int', default=2,
                      help='minimum number of observations required for a kalman object to be saved')

    (options, args) = parser.parse_args()
    if options.exclude_cam_ids is not None:
        options.exclude_cam_ids = options.exclude_cam_ids.split()

    if options.exclude_camns is not None:
        options.exclude_camns = [int(camn) for camn in options.exclude_camns.split()]

    if len(args)>1:
        print 'args',args
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return

    if len(args)<1:
        parser.print_help()
        return

    src_filename = args[0]

    args = (src_filename,)
    kwargs = dict(
              dest_filename=options.dest_filename,
              reconstructor_filename=options.reconstructor_path,
              start_frame=options.start,
              stop_frame=options.stop,
              exclude_cam_ids=options.exclude_cam_ids,
              exclude_camns=options.exclude_camns,
              dynamic_model_name = options.dynamic_model,
              save_cal_dir = options.save_cal_dir,
              debug = options.debug,
              frames_per_second = options.fps,
              max_err = options.max_err,
              area_threshold=options.area_threshold,
              min_observations_to_save=options.min_observations_to_save,
              )

    if int(os.environ.get('PROFILE','0')):
        import cProfile
        import lsprofcalltree
        p = cProfile.Profile()
        p.runctx('kalmanize(*args, **kwargs)',globals(),locals())
        k = lsprofcalltree.KCacheGrind(p)
        data = open(os.path.expanduser('~/kalmanize.kgrind'), 'w+')
        k.output(data)
        data.close()
    else:
        kalmanize(*args, **kwargs)

if __name__=='__main__':
    main()
