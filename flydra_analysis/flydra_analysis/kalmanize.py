from __future__ import with_statement
from __future__ import print_function
import numpy
import numpy as np
import flydra_core.reconstruct
import flydra_core._reconstruct_utils as ru

# import flydra_core.geom as geom
import flydra_core._fastgeom as geom
import time
from flydra_analysis.analysis.result_utils import (
    get_caminfo_dicts,
    get_fps,
    read_textlog_header,
)
import tables
import tables as PT
import warnings

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)
import os, sys, pprint, tempfile, shutil
from flydra_core.kalman.flydra_tracker import Tracker
import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils
from optparse import OptionParser
import flydra_core.kalman.dynamic_models as dynamic_models
import collections
import flydra_core.version
from flydra_core.data_descriptions import TextLogDescription
from flydra_core.reconstruct import do_3d_operations_on_2d_point
import flydra_analysis.a2.utils as utils
from flydra_analysis.a2.tables_tools import open_file_safe

# Not really "observations" but ML estimates
FilteredObservations = flydra_kalman_utils.FilteredObservations
convert_format = flydra_kalman_utils.convert_format
tmp = flydra_kalman_utils.ML_estimates_2d_idxs_type
ML_estimates_2d_idxs_type = tmp
del tmp


def process_frame(reconstructor, tracker, frame, frame_data, camn2cam_id, debug=0):
    if debug is None:
        debug = 0
    frame_data = tracker.calculate_a_posteriori_estimates(
        frame, frame_data, camn2cam_id, debug2=debug
    )

    frame_data = tracker.remove_duplicate_detections(frame, frame_data)
    max_err = tracker.kalman_model["hypothesis_test_max_acceptable_error"]

    # Now, tracked objects have been updated (and their 2D data points
    # removed from consideration), so we can use old flydra
    # "hypothesis testing" algorithm on remaining data to see if there
    # are new objects.

    if debug > 1:
        print("for frame %d: data not gobbled:" % (frame,))
        pprint.pprint(dict(frame_data))
        print()

    # Convert to format accepted by find_best_3d()
    found_data_dict, first_idx_by_cam_id = convert_format(
        frame_data, camn2cam_id, area_threshold=tracker.area_threshold
    )
    # found_data_dict contains undistorted points

    hypothesis_test_found_point = False
    # test to short-circuit rest of function
    if len(found_data_dict) >= 2:

        with_water = reconstructor.wateri is not None
        # Can only do 3D math with at least 2 cameras giving good
        # data.
        try:
            (
                this_observation_3d,
                this_observation_Lcoords,
                cam_ids_used,
                min_mean_dist,
            ) = ru.hypothesis_testing_algorithm__find_best_3d(
                reconstructor,
                found_data_dict,
                max_err,
                debug=debug,
                with_water=with_water,
            )
        except ru.NoAcceptablePointFound:
            pass
        else:
            hypothesis_test_found_point = True

    if hypothesis_test_found_point:

        if debug > 5:
            print("found new point using hypothesis testing:")
            print("this_observation_3d", this_observation_3d)
            print("cam_ids_used", cam_ids_used)
            print("min_mean_dist", min_mean_dist)

        believably_new = tracker.is_believably_new(this_observation_3d, debug=debug)
        if debug > 5:
            print("believably_new", believably_new)

        if believably_new:
            assert min_mean_dist < max_err
            if debug > 5:
                print("accepting point")

            # make mapping from cam_id to camn
            cam_id2camn = {}
            for camn in camn2cam_id:
                if camn not in frame_data:
                    continue  # this camn not used this frame, ignore
                cam_id = camn2cam_id[camn]
                if cam_id in cam_id2camn:
                    print("*" * 80)
                    print(
                        """

ERROR: It appears that you have >1 camn for a cam_id at a certain
frame. This almost certainly means that you are using a data file
recorded with an older version of flydra.MainBrain and that the
cameras were re-synchronized during the saving of a data file. You
will have to manually find out which camns to ignore (use
flydra_analysis_print_camera_summary) and then use the --exclude-camns
option to this program.

"""
                    )
                    print("*" * 80)
                    print()
                    print("frame", frame)
                    print("camn", camn)
                    print("frame_data", frame_data)
                    print()
                    print("cam_id2camn", cam_id2camn)
                    print("camn2cam_id", camn2cam_id)
                    print()
                    raise ValueError("cam_id already in dict")
                cam_id2camn[cam_id] = camn

            # find camns
            this_observation_camns = [cam_id2camn[cam_id] for cam_id in cam_ids_used]

            this_observation_idxs = [
                first_idx_by_cam_id[cam_id] for cam_id in cam_ids_used
            ]

            if debug > 5:
                print("this_observation_camns", this_observation_camns)
                print("this_observation_idxs", this_observation_idxs)

                print("camn", "raw 2d data", "reprojected 3d->2d")
                for camn in this_observation_camns:
                    cam_id = camn2cam_id[camn]
                    repro = reconstructor.find2d(cam_id, this_observation_3d)
                    print(camn, frame_data[camn][0][0][:2], repro)

            ####################################
            #  Now join found point into Tracker
            tracker.join_new_obj(
                frame,
                this_observation_3d,
                this_observation_Lcoords,
                this_observation_camns,
                this_observation_idxs,
                debug=debug,
            )
    if debug > 5:
        print()
        print("At end of frame %d, all live tracked objects:" % frame)
        tracker.debug_info(level=debug)
        print()
        print("-" * 80)
    elif debug > 2:
        print("At end of frame %d, all live tracked objects:" % frame)
        tracker.debug_info(level=debug)
        print()


class KalmanSaver:
    def __init__(
        self,
        h5file,
        reconstructor,
        cam_id2camns=None,
        min_observations_to_save=0,
        textlog_save_lines=None,
        dynamic_model_name=None,
        dynamic_model=None,
        fake_timestamp=None,
        debug=False,
    ):
        self.cam_id2camns = cam_id2camns
        self.min_observations_to_save = min_observations_to_save
        self.debug = debug

        self.kalman_saver_info_instance = flydra_kalman_utils.KalmanSaveInfo(
            name=dynamic_model_name
        )
        kalman_estimates_description = self.kalman_saver_info_instance.get_description()

        filters = tables.Filters(1, complib="zlib")  # compress

        self.h5file = h5file
        reconstructor.save_to_h5file(self.h5file)
        self.h5_xhat = self.h5file.create_table(
            self.h5file.root,
            "kalman_estimates",
            kalman_estimates_description,
            "Kalman a posteriori estimates of tracked object",
            filters=filters,
        )
        self.h5_xhat.attrs.dynamic_model_name = dynamic_model_name
        self.h5_xhat.attrs.dynamic_model = dynamic_model

        self.h5_obs = self.h5file.create_table(
            self.h5file.root,
            "ML_estimates",
            FilteredObservations,
            "observations of tracked object",
            filters=filters,
        )

        self.h5_2d_obs_next_idx = 0

        # Note that ML_estimates_2d_idxs_type() should
        # match dtype with tro.observations_2d.

        self.h5_2d_obs = self.h5file.create_vlarray(
            self.h5file.root,
            "ML_estimates_2d_idxs",
            ML_estimates_2d_idxs_type(),
            "camns and idxs",
        )

        self.obj_id = 0

        self.h5textlog = self.h5file.create_table(
            self.h5file.root, "textlog", TextLogDescription, "text log"
        )

        if 1:
            textlog_row = self.h5textlog.row
            cam_id = "mainbrain"
            if fake_timestamp is None:
                timestamp = time.time()
            else:
                timestamp = fake_timestamp

            list_of_textlog_data = [
                (timestamp, cam_id, timestamp, text) for text in textlog_save_lines
            ]
            for textlog_data in list_of_textlog_data:
                (mainbrain_timestamp, cam_id, host_timestamp, message) = textlog_data
                textlog_row["mainbrain_timestamp"] = mainbrain_timestamp
                textlog_row["cam_id"] = cam_id
                textlog_row["host_timestamp"] = host_timestamp
                textlog_row["message"] = message
                textlog_row.append()

            self.h5textlog.flush()

        self.h5_xhat_names = PT.Description(
            kalman_estimates_description().columns
        )._v_names
        self.h5_obs_names = PT.Description(FilteredObservations().columns)._v_names
        self.all_kalman_calibration_data = []

    def close(self):
        pass

    def save_tro(self, tro, force_obj_id=None):
        if len(tro.observations_frames) < self.min_observations_to_save:
            # not enough data to bother saving
            return

        self.obj_id += 1

        if force_obj_id is not None:
            save_obj_id = force_obj_id
        else:
            save_obj_id = self.obj_id

        if self.debug:
            print("saving %s as obj_id %d" % (repr(self), save_obj_id))

        # save observation 2d data indexes
        debugADS = False

        if debugADS:
            print("2D indices: ----------------")

        this_idxs = []
        for camns_and_idxs in tro.observations_2d:
            this_idxs.append(self.h5_2d_obs_next_idx)
            self.h5_2d_obs.append(camns_and_idxs)

            if debugADS:
                print(" %d: %s" % (self.h5_2d_obs_next_idx, str(camns_and_idxs)))
            self.h5_2d_obs_next_idx += 1
        self.h5_2d_obs.flush()

        if debugADS:
            print()

        # becomes obs_2d_idx (index into 'ML_estimates_2d_idxs')
        this_idxs = numpy.array(this_idxs, dtype=numpy.uint64)

        # save observations ####################################
        observations_frames = numpy.array(tro.observations_frames, dtype=numpy.uint64)
        obj_id_array = numpy.empty(observations_frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(save_obj_id)
        MLE_position = numpy.array(tro.MLE_position, dtype=numpy.float32)
        MLE_Lcoords = numpy.array(tro.MLE_Lcoords, dtype=numpy.float32)
        list_of_pos = [MLE_position[:, i] for i in range(MLE_position.shape[1])]
        list_of_lines = [MLE_Lcoords[:, i] for i in range(MLE_Lcoords.shape[1])]
        array_list = (
            [obj_id_array, observations_frames]
            + list_of_pos
            + [this_idxs]
            + list_of_lines
        )
        obs_recarray = numpy.rec.fromarrays(array_list, names=self.h5_obs_names)
        if 0:
            # End tracking at last non-nan observation (must be > 1
            # camera for final points).
            idx = numpy.nonzero(~numpy.isnan(MLE_position))[0][-1]
            last_observation_frame = observations_frames[idx]
        else:
            # End tracking at last observation (can be 1 camera for
            # final points).
            last_observation_frame = observations_frames[-1]

        if debugADS:
            print("kalman observations: --------------")
            for row in obs_recarray:
                print(row["frame"], row["obs_2d_idx"])

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

        obj_id_array = numpy.empty(frames.shape, dtype=numpy.uint32)
        obj_id_array.fill(save_obj_id)

        # one list entry per column
        list_of_xhats = [xhat_data[:, i] for i in range(xhat_data.shape[1])]

        tmp = self.kalman_saver_info_instance.covar_mats_to_covar_entries
        list_of_Ps = tmp(P_data_full)
        del tmp

        xhats_recarray = numpy.rec.fromarrays(
            [obj_id_array, frames, timestamps] + list_of_xhats + list_of_Ps,
            names=self.h5_xhat_names,
        )

        self.h5_xhat.append(xhats_recarray)
        self.h5_xhat.flush()


def kalmanize(
    src_filename,
    do_full_kalmanization=True,
    dest_filename=None,
    reconstructor=None,
    reconstructor_filename=None,
    start_frame=None,
    stop_frame=None,
    exclude_cam_ids=None,
    exclude_camns=None,
    dynamic_model_name=None,
    debug=False,
    frames_per_second=None,
    area_threshold=0,
    min_observations_to_save=0,
    options=None,
):
    if options is None:
        # get default options
        parser = get_parser()
        (options, args) = parser.parse_args([])

    if debug:
        numpy.set_printoptions(precision=3, linewidth=120, suppress=False)

    if exclude_cam_ids is None:
        exclude_cam_ids = []

    if exclude_camns is None:
        exclude_camns = []

    use_existing_filename = True

    if reconstructor is not None:
        assert isinstance(reconstructor, flydra_core.reconstruct.Reconstructor)
        assert reconstructor_filename is None

    with open_file_safe(src_filename, mode="r") as results:
        camn2cam_id, cam_id2camns = get_caminfo_dicts(results)

        if do_full_kalmanization:
            if dynamic_model_name is None:
                if hasattr(results.root, "kalman_estimates"):
                    if hasattr(
                        results.root.kalman_estimates.attrs, "dynamic_model_name"
                    ):
                        dynamic_model_name = (
                            results.root.kalman_estimates.attrs.dynamic_model_name
                        )
                        warnings.warn(
                            "dynamic model not specified. "
                            'using "%s"' % dynamic_model_name
                        )
            if dynamic_model_name is None:
                dynamic_model_name = "EKF mamarama, units: mm"
                warnings.warn(
                    "dynamic model not specified. " 'using "%s"' % dynamic_model_name
                )
            else:
                print('using dynamic model "%s"' % dynamic_model_name)

            if reconstructor_filename is not None:
                if reconstructor_filename.endswith("h5"):
                    with PT.open_file(reconstructor_filename, mode="r") as fd:
                        reconstructor = flydra_core.reconstruct.Reconstructor(
                            fd, minimum_eccentricity=options.force_minimum_eccentricity
                        )
                else:
                    reconstructor = flydra_core.reconstruct.Reconstructor(
                        reconstructor_filename,
                        minimum_eccentricity=options.force_minimum_eccentricity,
                    )
            else:
                # reconstructor_filename is None
                if reconstructor is None:
                    reconstructor = flydra_core.reconstruct.Reconstructor(
                        results, minimum_eccentricity=options.force_minimum_eccentricity
                    )

            if options.force_minimum_eccentricity is not None:
                if (
                    reconstructor.minimum_eccentricity
                    != options.force_minimum_eccentricity
                ):
                    raise ValueError("could not force minimum_eccentricity")

            if dest_filename is None:
                dest_filename = os.path.splitext(results.filename)[0] + ".kalmanized.h5"
        else:
            use_existing_filename = False
            dest_filename = tempfile.mktemp(suffix=".h5")

        if reconstructor is not None and reconstructor.cal_source_type == "pytables":
            save_reconstructor_filename = reconstructor.cal_source.filename
        else:
            warnings.warn(
                "unable to determine reconstructor source "
                "filename for %r" % reconstructor.cal_source_type
            )
            save_reconstructor_filename = None

        if frames_per_second is None:
            frames_per_second = get_fps(results)
            if do_full_kalmanization:
                print("read frames_per_second from file", frames_per_second)

        dt = 1.0 / frames_per_second

        if options.sync_error_threshold_msec is None:
            # default is IFI/2
            sync_error_threshold = 0.5 * dt
        else:
            sync_error_threshold = options.sync_error_threshold_msec / 1000.0

        if os.path.exists(dest_filename):
            if use_existing_filename:
                raise ValueError(
                    "%s already exists. Will not " "overwrite." % dest_filename
                )
            else:
                os.unlink(dest_filename)

        with open_file_safe(
            dest_filename,
            mode="w",
            title="tracked Flydra data file",
            delete_on_error=True,
        ) as h5file:

            if "experiment_info" in results.root:
                results.root.experiment_info._f_copy(h5file.root, recursive=True)

            if do_full_kalmanization:
                parsed = read_textlog_header(results)
                if "trigger_CS3" not in parsed:
                    parsed["trigger_CS3"] = "unknown"
                textlog_save_lines = [
                    "kalmanize running at %s fps, (top %s, trigger_CS3 %s, flydra_version %s)"
                    % (
                        str(frames_per_second),
                        str(parsed.get("top", "unknown")),
                        str(parsed["trigger_CS3"]),
                        flydra_core.version.__version__,
                    ),
                    "original file: %s" % (src_filename,),
                    "dynamic model: %s" % (dynamic_model_name,),
                    "reconstructor file: %s" % (save_reconstructor_filename,),
                ]

                kalman_model = dynamic_models.get_kalman_model(
                    name=dynamic_model_name, dt=dt
                )

                h5saver = KalmanSaver(
                    h5file,
                    reconstructor,
                    cam_id2camns=cam_id2camns,
                    min_observations_to_save=min_observations_to_save,
                    textlog_save_lines=textlog_save_lines,
                    dynamic_model_name=dynamic_model_name,
                    dynamic_model=kalman_model,
                    debug=debug,
                    fake_timestamp=options.fake_timestamp,
                )

                tracker = Tracker(
                    reconstructor,
                    kalman_model=kalman_model,
                    save_all_data=True,
                    area_threshold=area_threshold,
                    area_threshold_for_orientation=options.area_threshold_for_orientation,
                    disable_image_stat_gating=options.disable_image_stat_gating,
                    orientation_consensus=options.orientation_consensus,
                    fake_timestamp=options.fake_timestamp,
                )

                tracker.set_killed_tracker_callback(h5saver.save_tro)

                # copy timestamp data into newly created kalmanized file
                if hasattr(results.root, "trigger_clock_info"):
                    results.root.trigger_clock_info._f_copy(h5file.root)

            data2d = results.root.data2d_distorted

            frame_count = 0
            last_frame = None
            frame_data = collections.defaultdict(list)
            time_frame_all_cam_timestamps = []
            time_frame_all_camns = []

            if 1:
                time1 = time.time()
                if do_full_kalmanization:
                    print("loading all frame numbers...")
                frames_array = numpy.asarray(data2d.read(field="frame"))
                time2 = time.time()
                if do_full_kalmanization:
                    print("done in %.1f sec" % (time2 - time1))
                    if (
                        not options.disable_image_stat_gating
                        and "cur_val" in data2d.colnames
                    ):
                        warnings.warn(
                            "No pre-filtering of data based on zero "
                            "probability -- more data association work is "
                            "being done than necessary"
                        )

            if len(frames_array) == 0:
                # no data
                print("No 2D data. Nothing to do.")
                return

            if do_full_kalmanization:
                print(
                    "2D data range: approximately %d<frame<%d"
                    % (frames_array[0], frames_array[-1])
                )

            if do_full_kalmanization:
                accum_frame_spread = None
            else:
                accum_frame_spread = []
                accum_frame_spread_fno = []
                accum_frame_all_timestamps = []
                accum_frame_all_camns = []

            max_all_check_times = -np.inf

            for row_start, row_stop in utils.iter_non_overlapping_chunk_start_stops(
                frames_array,
                min_chunk_size=500000,
                size_increment=1000,
                status_fd=sys.stdout,
            ):

                print(
                    "Doing initial scan of approx frame range %d-%d."
                    % (frames_array[row_start], frames_array[row_stop - 1])
                )

                this_frames_array = frames_array[row_start:row_stop]
                if start_frame is not None:
                    if this_frames_array.max() < start_frame:
                        continue
                if stop_frame is not None:
                    if this_frames_array.min() > stop_frame:
                        continue

                data2d_recarray = data2d.read(start=row_start, stop=row_stop)
                this_frames = data2d_recarray["frame"]
                print(
                    "Examining frames %d-%d in detail."
                    % (this_frames[0], this_frames[-1])
                )
                this_row_idxs = np.argsort(this_frames)
                for ii in range(len(this_row_idxs) + 1):

                    if ii >= len(this_row_idxs):
                        finish_frame = True
                    else:
                        finish_frame = False

                        this_row_idx = this_row_idxs[ii]

                        row = data2d_recarray[this_row_idx]

                        new_frame = row["frame"]

                        if start_frame is not None:
                            if new_frame < start_frame:
                                continue
                        if stop_frame is not None:
                            if new_frame > stop_frame:
                                continue

                        if last_frame != new_frame:
                            if last_frame is not None and new_frame < last_frame:
                                print("new_frame", new_frame)
                                print("last_frame", last_frame)
                                raise RuntimeError(
                                    "expected continuously increasing " "frame numbers"
                                )
                            finish_frame = True

                    if finish_frame:
                        # new frame
                        ########################################
                        # Data for this frame is complete
                        if last_frame is not None:

                            this_frame_spread = 0.0
                            if len(time_frame_all_cam_timestamps) > 1:
                                check_times = np.array(time_frame_all_cam_timestamps)
                                check_times -= check_times.min()
                                this_frame_spread = check_times.max()
                                if accum_frame_spread is not None:
                                    accum_frame_spread.append(this_frame_spread)
                                    accum_frame_spread_fno.append(last_frame)

                                    accum_frame_all_timestamps.append(
                                        time_frame_all_cam_timestamps
                                    )
                                    accum_frame_all_camns.append(time_frame_all_camns)

                                max_all_check_times = max(
                                    this_frame_spread, max_all_check_times
                                )
                                if this_frame_spread > sync_error_threshold:
                                    if this_frame_spread == max_all_check_times:
                                        print(
                                            "%s frame %d: sync diff: %.1f msec"
                                            % (
                                                os.path.split(results.filename)[-1],
                                                last_frame,
                                                this_frame_spread * 1000.0,
                                            )
                                        )

                            if debug > 5:
                                print()
                                print("frame_data for frame %d" % (last_frame,))
                                pprint.pprint(dict(frame_data))
                                print()
                            if do_full_kalmanization:
                                if this_frame_spread > sync_error_threshold:
                                    if debug > 5:
                                        print(
                                            "frame sync error (spread %.1f msec), "
                                            "skipping" % (this_frame_spread * 1e3,)
                                        )
                                        print()
                                    warnings.warn(
                                        "Synchronization error detected, "
                                        "but continuing analysis without "
                                        "potentially bad data."
                                    )
                                else:
                                    process_frame(
                                        reconstructor,
                                        tracker,
                                        last_frame,
                                        frame_data,
                                        camn2cam_id,
                                        debug=debug,
                                    )
                            frame_count += 1
                            if do_full_kalmanization and frame_count % 1000 == 0:
                                time2 = time.time()
                                dur = time2 - time1
                                fps = frame_count / dur
                                print(
                                    "frame % 10d, kalmanization/data association speed: % 8.1f fps"
                                    % (last_frame, fps)
                                )
                                time1 = time2
                                frame_count = 0

                        ########################################
                        frame_data = collections.defaultdict(list)
                        time_frame_all_cam_timestamps = []  # clear values
                        time_frame_all_camns = []  # clear values
                        last_frame = new_frame

                    camn = row["camn"]
                    try:
                        cam_id = camn2cam_id[camn]
                    except KeyError:
                        # This will happen if cameras were re-synchronized (and
                        # thus gain new cam_ids) immediately before saving was
                        # turned on in MainBrain. The reason is that the network
                        # buffers are still full of old data coming in from the
                        # cameras.
                        warnings.warn(
                            "WARNING: no cam_id for camn "
                            "%d, skipping this row of data" % camn
                        )
                        continue

                    if cam_id in exclude_cam_ids:
                        # exclude this camera
                        continue

                    if camn in exclude_camns:
                        # exclude this camera
                        continue

                    time_frame_all_cam_timestamps.append(row["timestamp"])
                    time_frame_all_camns.append(row["camn"])

                    if do_full_kalmanization:

                        x_distorted = row["x"]
                        if numpy.isnan(x_distorted):
                            # drop point -- not found
                            continue
                        y_distorted = row["y"]

                        (x_undistorted, y_undistorted) = reconstructor.undistort(
                            cam_id, (x_distorted, y_distorted)
                        )

                        (area, slope, eccentricity, frame_pt_idx) = (
                            row["area"],
                            row["slope"],
                            row["eccentricity"],
                            row["frame_pt_idx"],
                        )

                        if "cur_val" in row.dtype.fields:
                            cur_val = row["cur_val"]
                        else:
                            cur_val = None
                        if "mean_val" in row.dtype.fields:
                            mean_val = row["mean_val"]
                        else:
                            mean_val = None
                        if "sumsqf_val" in row.dtype.fields:
                            sumsqf_val = row["sumsqf_val"]
                        else:
                            sumsqf_val = None

                        # FIXME: cache this stuff?
                        pmat_inv = reconstructor.get_pmat_inv(cam_id)
                        camera_center = reconstructor.get_camera_center(cam_id)
                        camera_center = numpy.hstack((camera_center[:, 0], [1]))
                        camera_center_meters = reconstructor.get_camera_center(cam_id)
                        camera_center_meters = numpy.hstack(
                            (camera_center_meters[:, 0], [1])
                        )
                        helper = reconstructor.get_reconstruct_helper_dict()[cam_id]
                        rise = slope
                        run = 1.0
                        if np.isinf(rise):
                            if rise > 0:
                                rise = 1.0
                                run = 0.0
                            else:
                                rise = -1.0
                                run = 0.0

                        (
                            p1,
                            p2,
                            p3,
                            p4,
                            ray0,
                            ray1,
                            ray2,
                            ray3,
                            ray4,
                            ray5,
                        ) = do_3d_operations_on_2d_point(
                            helper,
                            x_undistorted,
                            y_undistorted,
                            pmat_inv,
                            camera_center,
                            x_distorted,
                            y_distorted,
                            rise,
                            run,
                        )
                        line_found = not numpy.isnan(p1)
                        pluecker_hz_meters = (ray0, ray1, ray2, ray3, ray4, ray5)

                        # Keep in sync with kalmanize.py and data_descriptions.py
                        pt_undistorted = (
                            x_undistorted,
                            y_undistorted,
                            area,
                            slope,
                            eccentricity,
                            p1,
                            p2,
                            p3,
                            p4,
                            line_found,
                            frame_pt_idx,
                            cur_val,
                            mean_val,
                            sumsqf_val,
                        )

                        projected_line_meters = geom.line_from_HZline(
                            pluecker_hz_meters
                        )

                        frame_data[camn].append((pt_undistorted, projected_line_meters))

            if do_full_kalmanization:
                tracker.kill_all_trackers()  # done tracking

        if not do_full_kalmanization:
            os.unlink(dest_filename)

    if accum_frame_spread is not None:
        # save spread data to file for analysis
        accum_frame_spread = np.array(accum_frame_spread)
        accum_frame_spread_fno = np.array(accum_frame_spread_fno)
        if options.dest_file is not None:
            accum_frame_spread_filename = options.dest_file
        else:
            accum_frame_spread_filename = src_filename + ".spreadh5"

        cam_ids = cam_id2camns.keys()
        cam_ids.sort()
        camn_order = []
        for cam_id in cam_ids:
            camn_order.extend(cam_id2camns[cam_id])

        camn_order = np.array(camn_order)
        cam_id_array = np.array(cam_ids)

        N_cams = len(camn_order)
        N_frames = len(accum_frame_spread_fno)

        all_timestamps = np.empty((N_frames, N_cams), dtype=np.float64)
        all_timestamps.fill(np.nan)
        for i, (timestamps, camns) in enumerate(
            zip(accum_frame_all_timestamps, accum_frame_all_camns)
        ):

            for j, camn in enumerate(camn_order):
                try:
                    idx = camns.index(camn)
                except ValueError:
                    continue  # not found, skip
                timestamp = timestamps[idx]
                all_timestamps[i, j] = timestamp

        h5 = tables.open_file(accum_frame_spread_filename, mode="w")
        h5.create_array(
            h5.root, "spread", accum_frame_spread, "frame timestamp spreads (sec)"
        )
        h5.create_array(h5.root, "framenumber", accum_frame_spread_fno, "frame number")
        h5.create_array(h5.root, "all_timestamps", all_timestamps, "all timestamps")
        h5.create_array(h5.root, "camn_order", camn_order, "camn_order")
        h5.create_array(h5.root, "cam_id_array", cam_id_array, "cam_id_array")
        h5.close()
        print("saved %s" % accum_frame_spread_filename)

    if max_all_check_times > sync_error_threshold:
        if not options.keep_sync_errors:
            if do_full_kalmanization:
                print("max_all_check_times %.2f msec" % (max_all_check_times * 1000.0))
                handle, target = tempfile.mkstemp(os.path.split(dest_filename)[1])
                os.unlink(target)  # remove original file there
                shutil.move(dest_filename, target)

                raise ValueError(
                    "Synchonization errors exist in the data. Moved result file"
                    " to ensure it is not confused with valid data. The new "
                    "location is: %s" % (target,)
                )

            else:
                sys.exit(1)  # sync error
    else:
        if not do_full_kalmanization:
            print(
                "%s no sync differences greater than %.1f msec"
                % (os.path.split(src_filename)[-1], sync_error_threshold * 1000.0,)
            )


def check_sync():
    usage = """%prog FILE [options]

This command will exit with a non-zero exit code if there are sync errors.
"""

    parser = OptionParser(usage)
    parser.add_option("--start", type="int", help="first frame", metavar="START")

    parser.add_option("--stop", type="int", help="last frame", metavar="STOP")

    parser.add_option(
        "--dest-file",
        type="string",
        help=("filename of .spreadh5 file (otherwise defaults " "to FILE.spreadh5)"),
    )

    parser.add_option("--sync-error-threshold-msec", type="float", default=None)

    parser.add_option(
        "--exclude-cam-ids",
        type="string",
        help="camera ids to exclude from reconstruction (space separated)",
        metavar="EXCLUDE_CAM_IDS",
    )

    parser.add_option(
        "--exclude-camns",
        type="string",
        help="camera numbers to exclude from reconstruction (space separated)",
        metavar="EXCLUDE_CAMNS",
    )

    parser.add_option("--debug", type="int", metavar="DEBUG")

    (options, args) = parser.parse_args()
    if options.exclude_cam_ids is not None:
        options.exclude_cam_ids = options.exclude_cam_ids.split()

    if options.exclude_camns is not None:
        options.exclude_camns = [int(camn) for camn in options.exclude_camns.split()]

    src_filename = args[0]

    kalmanize(
        src_filename,
        do_full_kalmanization=False,  # only check for sync errors
        start_frame=options.start,
        stop_frame=options.stop,
        exclude_cam_ids=options.exclude_cam_ids,
        exclude_camns=options.exclude_camns,
        debug=options.debug,
        options=options,
    )


def get_parser():
    usage = "%prog FILE [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "-d",
        "--dest-file",
        dest="dest_filename",
        type="string",
        help="save to hdf5 file (append if already present)",
        metavar="DESTFILE",
    )

    parser.add_option(
        "-r",
        "--reconstructor",
        dest="reconstructor_path",
        type="string",
        help=("calibration/reconstructor path (if not specified, " "defaults to FILE)"),
        metavar="RECONSTRUCTOR",
    )

    parser.add_option("--sync-error-threshold-msec", type="float", default=None)

    parser.add_option(
        "--save-cal-dir",
        type="string",
        help="directory name in which to save new calibration data",
        default=None,
    )

    parser.add_option(
        "--fake-timestamp",
        type="float",
        help="value of timestamp to use when saving logfile",
        default=None,
    )

    parser.add_option(
        "--fps",
        dest="fps",
        type="float",
        help="frames per second (used for Kalman filtering)",
    )

    parser.add_option(
        "--exclude-cam-ids",
        type="string",
        help="camera ids to exclude from reconstruction (space separated)",
        metavar="EXCLUDE_CAM_IDS",
    )

    parser.add_option(
        "--exclude-camns",
        type="string",
        help="camera numbers to exclude from reconstruction (space separated)",
        metavar="EXCLUDE_CAMNS",
    )

    parser.add_option("--dynamic-model", dest="dynamic_model", type="string")

    parser.add_option("--start", type="int", help="first frame", metavar="START")

    parser.add_option("--stop", type="int", help="last frame", metavar="STOP")

    parser.add_option("--debug", type="int", metavar="DEBUG")

    parser.add_option(
        "--keep-sync-errors",
        action="store_true",
        default=False,
        help="keep files with sync errors",
    )

    parser.add_option(
        "--area-threshold",
        type="float",
        default=0.0,
        help="area threshold (used to filter incoming 2d points)",
    )

    parser.add_option(
        "--area-threshold-for-orientation",
        type="float",
        default=0.0,
        help="minimum area to compute orientation",
    )

    parser.add_option(
        "--min-observations-to-save",
        type="int",
        default=2,
        help=(
            "minimum number of observations required for a kalman object " "to be saved"
        ),
    )

    parser.add_option("--orientation-consensus", type="int", default=0)

    parser.add_option("--force-minimum-eccentricity", type="float", default=None)

    parser.add_option(
        "--disable-image-stat-gating",
        action="store_true",
        help="disable gating the data based on image statistics",
        default=False,
    )
    return parser


def main():
    parser = get_parser()
    (options, args) = parser.parse_args()
    if options.exclude_cam_ids is not None:
        options.exclude_cam_ids = options.exclude_cam_ids.split()

    if options.exclude_camns is not None:
        options.exclude_camns = [int(camn) for camn in options.exclude_camns.split()]

    if len(args) > 1:
        print("args", args)
        print(
            ("arguments interpreted as FILE supplied more " "than once"),
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    if len(args) < 1:
        parser.print_help()
        sys.exit(1)

    src_filename = args[0]

    args = (src_filename,)
    kwargs = dict(
        dest_filename=options.dest_filename,
        reconstructor_filename=options.reconstructor_path,
        start_frame=options.start,
        stop_frame=options.stop,
        exclude_cam_ids=options.exclude_cam_ids,
        exclude_camns=options.exclude_camns,
        dynamic_model_name=options.dynamic_model,
        debug=options.debug,
        frames_per_second=options.fps,
        area_threshold=options.area_threshold,
        min_observations_to_save=options.min_observations_to_save,
        options=options,
    )

    if int(os.environ.get("PROFILE", "0")):
        import cProfile
        import lsprofcalltree

        p = cProfile.Profile()
        print("running kalmanize in profile mode")
        p.runctx("kalmanize(*args, **kwargs)", globals(), locals())
        k = lsprofcalltree.KCacheGrind(p)
        data = open(os.path.expanduser("~/kalmanize.kgrind"), "w+")
        k.output(data)
        data.close()
    else:
        kalmanize(*args, **kwargs)


if __name__ == "__main__":
    main()
