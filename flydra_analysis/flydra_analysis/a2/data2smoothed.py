from __future__ import print_function

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import numpy
import sys, os
import flydra_analysis.a2.core_analysis as core_analysis
import argparse

import flydra_analysis.analysis.flydra_analysis_convert_to_mat
import flydra_core.kalman.dynamic_models as dynamic_models
import tables
import flydra_analysis.analysis.result_utils as result_utils
import flydra_analysis.a2.utils as utils
from flydra_analysis.a2.orientation_ekf_fitter import compute_ori_quality
import warnings


def cam_id2hostname(cam_id, h5_context):
    cam_info = h5_context.get_pytable_node("cam_info", from_2d_file=True)
    ci = cam_info[:]
    if "hostname" in ci.dtype.names:
        # new style
        cond = ci["cam_id"] == cam_id
        rows = ci[cond]
        if len(rows) != 1:
            raise ValueError("multiple/no hostnames for observation: %s" % (rows,))
        hostname = rows["hostname"][0]
    else:
        # old style
        hostname = "_".join(cam_id.split("_")[:-1])
    return hostname


def convert(
    infilename,
    outfilename,
    frames_per_second=None,
    save_timestamps=True,
    file_time_data=None,
    do_nothing=False,  # set to true to test for file existance
    start_obj_id=None,
    stop_obj_id=None,
    obj_only=None,
    dynamic_model_name=None,
    hdf5=False,
    show_progress=False,
    show_progress_json=False,
    **kwargs
):
    if start_obj_id is None:
        start_obj_id = -numpy.inf
    if stop_obj_id is None:
        stop_obj_id = numpy.inf

    smoothed_data_filename = os.path.split(infilename)[1]
    raw_data_filename = smoothed_data_filename

    ca = core_analysis.get_global_CachingAnalyzer()
    with ca.kalman_analysis_context(
        infilename, data2d_fname=file_time_data
    ) as h5_context:

        extra_vars = {}
        tzname = None

        if save_timestamps:
            print("STAGE 1: finding timestamps")
            table_kobs = h5_context.get_pytable_node("ML_estimates")

            tzname = h5_context.get_tzname0()
            fps = h5_context.get_fps()

            try:
                table_data2d = h5_context.get_pytable_node(
                    "data2d_distorted", from_2d_file=True
                )
            except tables.exceptions.NoSuchNodeError as err:
                print(
                    "No timestamps in file. Either specify not to save timestamps ('--no-timestamps') or specify the original .h5 file with the timestamps ('--time-data=FILE2D')",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("caching raw 2D data...", end=" ")
            sys.stdout.flush()
            table_data2d_frames = table_data2d.read(field="frame")
            assert numpy.max(table_data2d_frames) < 2 ** 63
            table_data2d_frames = table_data2d_frames.astype(numpy.int64)
            # table_data2d_frames_find = fastsearch.binarysearch.BinarySearcher( table_data2d_frames )
            table_data2d_frames_find = utils.FastFinder(table_data2d_frames)
            table_data2d_camns = table_data2d.read(field="camn")
            table_data2d_timestamps = table_data2d.read(field="timestamp")
            print("done")
            print(
                "(cached index of %d frame values of dtype %s)"
                % (len(table_data2d_frames), str(table_data2d_frames.dtype))
            )

            drift_estimates = h5_context.get_drift_estimates()
            camn2cam_id, cam_id2camns = h5_context.get_caminfo_dicts()

            gain = {}
            offset = {}
            print("hostname time_gain time_offset")
            print("-------- --------- -----------")
            for i, hostname in enumerate(drift_estimates.get("hostnames", [])):
                tgain, toffset = result_utils.model_remote_to_local(
                    drift_estimates["remote_timestamp"][hostname][::10],
                    drift_estimates["local_timestamp"][hostname][::10],
                )
                gain[hostname] = tgain
                offset[hostname] = toffset
                print("  ", repr(hostname), tgain, toffset)
            print()

            if do_nothing:
                return

            print("caching Kalman obj_ids...")
            obs_obj_ids = table_kobs.read(field="obj_id")
            fast_obs_obj_ids = utils.FastFinder(obs_obj_ids)
            print("finding unique obj_ids...")
            unique_obj_ids = numpy.unique(obs_obj_ids)
            print("(found %d)" % (len(unique_obj_ids),))
            unique_obj_ids = unique_obj_ids[unique_obj_ids >= start_obj_id]
            unique_obj_ids = unique_obj_ids[unique_obj_ids <= stop_obj_id]

            if obj_only is not None:
                unique_obj_ids = numpy.array(
                    [oid for oid in unique_obj_ids if oid in obj_only]
                )
                print("filtered to obj_only", obj_only)

            print("(will export %d)" % (len(unique_obj_ids),))
            print("finding 2d data for each obj_id...")
            timestamp_time = numpy.zeros(unique_obj_ids.shape, dtype=numpy.float64)
            table_kobs_frame = table_kobs.read(field="frame")
            if len(table_kobs_frame) == 0:
                raise ValueError("no 3D data, cannot convert")
            assert numpy.max(table_kobs_frame) < 2 ** 63
            table_kobs_frame = table_kobs_frame.astype(numpy.int64)
            assert (
                table_kobs_frame.dtype == table_data2d_frames.dtype
            )  # otherwise very slow

            all_idxs = fast_obs_obj_ids.get_idx_of_equal(unique_obj_ids)
            for obj_id_enum, obj_id in enumerate(unique_obj_ids):
                idx0 = all_idxs[obj_id_enum]
                framenumber = table_kobs_frame[idx0]
                remote_timestamp = numpy.nan

                this_camn = None
                frame_idxs = table_data2d_frames_find.get_idxs_of_equal(framenumber)
                if len(frame_idxs):
                    frame_idx = frame_idxs[0]
                    this_camn = table_data2d_camns[frame_idx]
                    remote_timestamp = table_data2d_timestamps[frame_idx]

                if this_camn is None:
                    print(
                        "skipping frame %d (obj %d): no data2d_distorted data"
                        % (framenumber, obj_id)
                    )
                    continue

                cam_id = camn2cam_id[this_camn]
                try:
                    remote_hostname = cam_id2hostname(cam_id, h5_context)
                except ValueError as e:
                    print("error getting hostname of cam: %s" % e.message)
                    continue
                if remote_hostname not in gain:
                    warnings.warn(
                        "no host %s in timestamp data. making up "
                        "data." % remote_hostname
                    )
                    gain[remote_hostname] = 1.0
                    offset[remote_hostname] = 0.0
                mainbrain_timestamp = (
                    remote_timestamp * gain[remote_hostname] + offset[remote_hostname]
                )  # find mainbrain timestamp

                timestamp_time[obj_id_enum] = mainbrain_timestamp

            extra_vars["obj_ids"] = unique_obj_ids
            extra_vars["timestamps"] = timestamp_time

            print("STAGE 2: running Kalman smoothing operation")

        # also save the experiment data if present
        uuid = None
        try:
            table_experiment = h5_context.get_pytable_node(
                "experiment_info", from_2d_file=True
            )
        except tables.exceptions.NoSuchNodeError:
            pass
        else:
            try:
                uuid = table_experiment.read(field="uuid")
            except (KeyError, tables.exceptions.HDF5ExtError):
                pass
            else:
                extra_vars["experiment_uuid"] = uuid

        recording_header = h5_context.read_textlog_header_2d()
        recording_flydra_version = recording_header["flydra_version"]

        # -----------------------------------------------

        obj_ids = h5_context.get_unique_obj_ids()
        smoothing_flydra_version = h5_context.get_extra_info()["header"][
            "flydra_version"
        ]

        obj_ids = obj_ids[obj_ids >= start_obj_id]
        obj_ids = obj_ids[obj_ids <= stop_obj_id]

        if obj_only is not None:
            obj_ids = numpy.array(obj_only)
            print("filtered to obj_only", obj_ids)

        if frames_per_second is None:
            frames_per_second = h5_context.get_fps()

        if dynamic_model_name is None:
            extra = h5_context.get_extra_info()
            orig_dynamic_model_name = extra.get("dynamic_model_name", None)
            dynamic_model_name = orig_dynamic_model_name
            if dynamic_model_name is None:
                dynamic_model_name = dynamic_models.DEFAULT_MODEL
                warnings.warn(
                    'no dynamic model specified, using "%s"' % dynamic_model_name
                )
            else:
                print(
                    'detected file loaded with dynamic model "%s"' % dynamic_model_name
                )
            if dynamic_model_name.startswith("EKF "):
                dynamic_model_name = dynamic_model_name[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model_name)

        allrows = []
        allqualrows = []
        failed_quality = False

        if show_progress:
            import progressbar

            class StringWidget(progressbar.Widget):
                def set_string(self, ts):
                    self.ts = ts

                def update(self, pbar):
                    if hasattr(self, "ts"):
                        return self.ts
                    else:
                        return ""

            string_widget = StringWidget()
            objs_per_sec_widget = progressbar.FileTransferSpeed(unit="obj_ids ")
            widgets = [
                string_widget,
                objs_per_sec_widget,
                progressbar.Percentage(),
                progressbar.Bar(),
                progressbar.ETA(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(obj_ids)).start()

        for i, obj_id in enumerate(obj_ids):
            if obj_id > stop_obj_id:
                break
            if show_progress:
                string_widget.set_string("[obj_id: % 5d]" % obj_id)
                pbar.update(i)
            if show_progress_json and i % 100 == 0:
                rough_percent_done = float(i) / len(obj_ids) * 100.0
                result_utils.do_json_progress(rough_percent_done)
            try:
                rows = h5_context.load_data(
                    obj_id,
                    dynamic_model_name=dynamic_model_name,
                    frames_per_second=frames_per_second,
                    **kwargs
                )
            except core_analysis.DiscontiguousFramesError:
                warnings.warn(
                    "discontiguous frames smoothing obj_id %d, skipping." % (obj_id,)
                )
                continue
            except core_analysis.NotEnoughDataToSmoothError:
                # warnings.warn('not enough data to smooth obj_id %d, skipping.'%(obj_id,))
                continue
            except numpy.linalg.linalg.LinAlgError:
                warnings.warn(
                    "linear algebra error smoothing obj_id %d, skipping." % (obj_id,)
                )
                continue
            except core_analysis.CouldNotCalculateOrientationError:
                warnings.warn(
                    "orientation error smoothing obj_id %d, skipping." % (obj_id,)
                )
                continue

            allrows.append(rows)
            try:
                qualrows = compute_ori_quality(
                    h5_context, rows["frame"], obj_id, smooth_len=0
                )
                allqualrows.append(qualrows)
            except ValueError:
                failed_quality = True
        if show_progress:
            pbar.finish()

        allrows = numpy.concatenate(allrows)
        if not failed_quality:
            allqualrows = numpy.concatenate(allqualrows)
        else:
            allqualrows = None
        recarray = numpy.rec.array(allrows)

        smoothed_source = "kalman_estimates"

        flydra_analysis.analysis.flydra_analysis_convert_to_mat.do_it(
            rows=recarray,
            ignore_observations=True,
            newfilename=outfilename,
            extra_vars=extra_vars,
            orientation_quality=allqualrows,
            hdf5=hdf5,
            tzname=tzname,
            fps=fps,
            smoothed_source=smoothed_source,
            smoothed_data_filename=smoothed_data_filename,
            raw_data_filename=raw_data_filename,
            dynamic_model_name=orig_dynamic_model_name,
            recording_flydra_version=recording_flydra_version,
            smoothing_flydra_version=smoothing_flydra_version,
        )
        if show_progress_json:
            result_utils.do_json_progress(100)


def export_flydra_hdf5():
    main(hdf5_only=True)


def main(hdf5_only=False):
    # hdf5_only is to maintain backwards compatibility...
    parser = argparse.ArgumentParser()
    if hdf5_only:
        dest_help = "filename of output .h5 file"
    else:
        dest_help = "filename of output .mat file"
    parser.add_argument("file", type=str, default=None, help="input file")
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=False,
        help="show progress bar on console",
    )
    parser.add_argument(
        "--progress-json",
        dest="show_progress_json",
        action="store_true",
        default=False,
        help="show JSON progress messages",
    )
    parser.add_argument("--dest-file", type=str, default=None, help=dest_help)
    parser.add_argument(
        "--time-data",
        dest="file2d",
        type=str,
        help="hdf5 file with 2d data FILE2D used to calculate timestamp information and take UUID",
        metavar="FILE2D",
    )
    parser.add_argument(
        "--no-timestamps", action="store_true", dest="no_timestamps", default=False
    )
    if not hdf5_only:
        parser.add_argument(
            "--hdf5",
            action="store_true",
            default=False,
            help="save output as .hdf5 file (not .mat)",
        )
    parser.add_argument(
        "--start-obj-id", default=None, type=int, help="last obj_id to save"
    )
    parser.add_argument(
        "--stop-obj-id", default=None, type=int, help="last obj_id to save"
    )
    parser.add_argument("--obj-only", type=str)
    parser.add_argument(
        "--stop", default=None, type=int, help="last obj_id to save (DEPRECATED)"
    )
    parser.add_argument("--profile", action="store_true", dest="profile", default=False)
    parser.add_argument(
        "--dynamic-model", type=str, dest="dynamic_model", default=None,
    )
    core_analysis.add_arguments_to_parser(parser)
    options = parser.parse_args()

    if options.stop_obj_id is not None and options.stop is not None:
        raise ValueError("--stop and --stop-obj-id cannot both be set")

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

        if options.start_obj_id is not None or options.stop_obj_id is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    if options.stop is not None:
        warnings.warn(
            "DeprecationWarning: --stop will be phased out in favor of --stop-obj-id"
        )
        options.stop_obj_id = options.stop

    if hdf5_only:
        do_hdf5 = True
    else:
        do_hdf5 = options.hdf5

    infilename = options.file
    if options.dest_file is None:
        if do_hdf5:
            # import h5py early so if we don't have it we know sooner rather than later.
            import h5py

            outfilename = os.path.splitext(infilename)[0] + "_smoothed.h5"
        else:
            outfilename = os.path.splitext(infilename)[0] + "_smoothed.mat"
    else:
        outfilename = options.dest_file

    kwargs = core_analysis.get_options_kwargs(options)
    if options.profile:
        import cProfile

        out_stats_filename = outfilename + ".profile"
        print("profiling, stats will be saved to %r" % out_stats_filename)
        cProfile.runctx(
            """convert(infilename,outfilename,
                file_time_data=options.file2d,
                save_timestamps = not options.no_timestamps,
                start_obj_id=options.start_obj_id,
                stop_obj_id=options.stop_obj_id,
                obj_only=options.obj_only,
                dynamic_model_name=options.dynamic_model,
                return_smoothed_directions = True,
                hdf5 = do_hdf5,
                show_progress = options.show_progress,
                show_progress_json=options.show_progress_json,
                **kwargs)""",
            globals(),
            locals(),
            out_stats_filename,
        )

    else:
        convert(
            infilename,
            outfilename,
            file_time_data=options.file2d,
            save_timestamps=not options.no_timestamps,
            start_obj_id=options.start_obj_id,
            stop_obj_id=options.stop_obj_id,
            obj_only=options.obj_only,
            dynamic_model_name=options.dynamic_model,
            return_smoothed_directions=True,
            hdf5=do_hdf5,
            show_progress=options.show_progress,
            show_progress_json=options.show_progress_json,
            **kwargs
        )


if __name__ == "__main__":
    main()
