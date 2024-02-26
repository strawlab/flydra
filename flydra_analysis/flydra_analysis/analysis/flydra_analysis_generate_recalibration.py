from __future__ import division
from __future__ import print_function
import tables.flavor

tables.flavor.restrict_flavors(keep=["numpy"])

import numpy
import tables as PT

import os.path
import sys
import progressbar
import tempfile
from optparse import OptionParser

from pymulticamselfcal.execute import MultiCamSelfCal

import warnings

import flydra_core.reconstruct
import flydra_analysis.analysis.result_utils as result_utils


def save_calibration_directory(
    IdMat=None,
    points=None,
    Res=None,
    calib_dir=None,
    cam_ids=None,
    square_pixels=True,
    num_cameras_fill=-1,
    # intrinsics_reconstructor=None,
    # intrinsics_yaml=None,
):
    warnings.warn(
        "DeprecationWarning: save_calibration_directory backward compatibility shim being used",
        DeprecationWarning,
        stacklevel=2,
    )

    mcsc = MultiCamSelfCal(calib_dir)
    mcsc.create_calibration_directory(
        cam_ids=cam_ids,
        IdMat=IdMat,
        points=points,
        Res=Res,
        # cam_calibrations=cam_calibrations,
        # cam_centers=cam_centers,
        # radial_distortion=undo_radial_distortion,
        square_pixels=square_pixels,
        num_cameras_fill=num_cameras_fill,
    )


def create_new_row(
    d2d, this_camns, this_camn_idxs, cam_ids, camn2cam_id, npoints_by_cam_id
):
    n_pts = 0
    IdMat_row = []
    points_row = []
    for cam_id in cam_ids:
        found = False
        for this_camn, this_camn_idx in zip(this_camns, this_camn_idxs):
            if camn2cam_id[this_camn] != cam_id:
                continue

            this_camn_d2d = d2d[d2d["camn"] == this_camn]
            for this_row in this_camn_d2d:  # XXX could be sped up
                if this_row["frame_pt_idx"] == this_camn_idx:
                    found = True
                    break
        if not found:
            IdMat_row.append(0)
            points_row.extend([numpy.nan, numpy.nan, numpy.nan])
        else:
            npoints_by_cam_id[cam_id] = npoints_by_cam_id[cam_id] + 1
            n_pts += 1
            IdMat_row.append(1)
            points_row.extend([this_row["x"], this_row["y"], 1.0])
    return IdMat_row, points_row


def do_it(
    filename,
    efilename,
    use_nth_observation=None,
    h5_2d_data_filename=None,
    use_kalman_data=True,
    start=None,
    stop=None,
    options=None,
):
    if h5_2d_data_filename is None:
        h5_2d_data_filename = filename

    if use_nth_observation is None:
        use_nth_observation = 1

    calib_dir = filename + ".recal"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    results = result_utils.get_results(filename, mode="r+")

    if use_kalman_data:
        mylocals = {}
        myglobals = {}
        exec(open(efilename).read(), myglobals, mylocals)

        use_obj_ids = mylocals["long_ids"]
        if "bad" in mylocals:
            use_obj_ids = set(use_obj_ids)
            bad = set(mylocals["bad"])
            use_obj_ids = list(use_obj_ids.difference(bad))
        kobs = results.root.ML_estimates
        kobs_2d = results.root.ML_estimates_2d_idxs

    h5_2d_data = result_utils.get_results(h5_2d_data_filename, mode="r+")

    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5_2d_data)

    cam_ids = list(cam_id2camns.keys())
    cam_ids.sort()

    data2d = h5_2d_data.root.data2d_distorted
    # use_idxs = numpy.arange(data2d.nrows)
    frames = data2d.cols.frame[:]
    qfi = result_utils.QuickFrameIndexer(frames)

    npoints_by_ncams = {}

    npoints_by_cam_id = {}
    for cam_id in cam_ids:
        npoints_by_cam_id[cam_id] = 0

    IdMat = []
    points = []

    if use_kalman_data:
        row_keys = []
        if start is not None or stop is not None:
            print("start, stop", start, stop)
            print(
                "WARNING: currently ignoring start/stop because Kalman data is being used"
            )
        for obj_id_enum, obj_id in enumerate(use_obj_ids):
            row_keys.append((len(points), obj_id))
            #            print 'obj_id %d (%d of %d)'%(obj_id, obj_id_enum+1, len(use_obj_ids))
            this_obj_id = obj_id
            k_use_idxs = kobs.get_where_list("obj_id==this_obj_id")
            obs_2d_idxs = kobs.read_coordinates(k_use_idxs, field="obs_2d_idx")
            kframes = kobs.read_coordinates(k_use_idxs, field="frame")
            kframes_use = kframes[::use_nth_observation]
            obs_2d_idxs_use = obs_2d_idxs[::use_nth_observation]

            widgets = [
                "obj_id % 5d (% 3d of % 3d) "
                % (obj_id, obj_id_enum + 1, len(use_obj_ids)),
                progressbar.Percentage(),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ]

            pbar = progressbar.ProgressBar(
                widgets=widgets, maxval=len(kframes_use)
            ).start()

            for n_kframe, (kframe, obs_2d_idx) in enumerate(
                zip(kframes_use, obs_2d_idxs_use)
            ):
                pbar.update(n_kframe)
                if 0:
                    k_use_idx = k_use_idxs[n_kframe * use_nth_observation]
                    print(kobs.read_coordinates(numpy.array([k_use_idx])))
                if PT.__version__ <= "1.3.3":
                    obs_2d_idx_find = int(obs_2d_idx)
                    kframe_find = int(kframe)
                else:
                    obs_2d_idx_find = obs_2d_idx
                    kframe_find = kframe
                obj_id_save = int(obj_id)  # convert from possible numpy scalar

                # sys.stdout.write('  reading frame data...')
                # sys.stdout.flush()
                obs_2d_idx_find_next = obs_2d_idx_find + numpy.uint64(1)
                kobs_2d_data = kobs_2d.read(
                    start=obs_2d_idx_find, stop=obs_2d_idx_find_next
                )
                # sys.stdout.write('done\n')
                # sys.stdout.flush()

                assert len(kobs_2d_data) == 1
                kobs_2d_data = kobs_2d_data[0]
                this_camns = kobs_2d_data[0::2]
                this_camn_idxs = kobs_2d_data[1::2]

                # sys.stdout.write('  doing frame selections...')
                # sys.stdout.flush()
                if 1:
                    this_use_idxs = qfi.get_frame_idxs(kframe_find)
                elif 0:
                    this_use_idxs = numpy.nonzero(frames == kframe_find)[0]
                else:
                    this_use_idxs = data2d.get_where_list("frame==kframe_find")
                # sys.stdout.write('done\n')
                # sys.stdout.flush()

                if PT.__version__ <= "1.3.3":
                    this_use_idxs = [int(t) for t in this_use_idxs]

                d2d = data2d.read_coordinates(this_use_idxs)
                if len(this_camns) < options.min_num_points:
                    # not enough points to contribute to calibration
                    continue

                npoints_by_ncams[len(this_camns)] = (
                    npoints_by_ncams.get(len(this_camns), 0) + 1
                )

                IdMat_row, points_row = create_new_row(
                    d2d,
                    this_camns,
                    this_camn_idxs,
                    cam_ids,
                    camn2cam_id,
                    npoints_by_cam_id,
                )

                IdMat.append(IdMat_row)
                points.append(points_row)
            ##             print 'running total of points','-'*20
            ##             for cam_id in cam_ids:
            ##                 print 'cam_id %s: %d points'%(cam_id,npoints_by_cam_id[cam_id])
            ##             print
            pbar.finish()

    if start is None:
        start = 0
    if stop is None:
        stop = int(frames.max())

    if not use_kalman_data:
        row_keys = None
        count = 0
        for frameno in range(start, stop + 1, use_nth_observation):
            this_use_idxs = qfi.get_frame_idxs(frameno)

            d2d = data2d.read_coordinates(this_use_idxs)
            d2d = d2d[~numpy.isnan(d2d["x"])]
            this_camns = d2d["camn"]

            unique_camns = numpy.unique(this_camns)
            if len(this_camns) != len(unique_camns):
                # ambiguity - a camera has > 1 point
                continue
            this_camn_idxs = numpy.array([0] * len(this_camns))

            if len(this_camns) < options.min_num_points:
                # not enough points to contribute to calibration
                continue

            npoints_by_ncams[len(this_camns)] = (
                npoints_by_ncams.get(len(this_camns), 0) + 1
            )
            count += 1

            IdMat_row, points_row = create_new_row(
                d2d, this_camns, this_camn_idxs, cam_ids, camn2cam_id, npoints_by_cam_id
            )
            IdMat.append(IdMat_row)
            points.append(points_row)
    print("%d points" % len(IdMat))

    print("by camera id:")
    for cam_id in cam_ids:
        print(" %s: %d" % (cam_id, npoints_by_cam_id[cam_id]))
    print("by n points:")
    max_npoints = 0
    for ncams in npoints_by_ncams:
        print(" %d: %d" % (ncams, npoints_by_ncams[ncams]))
        max_npoints = max(max_npoints, npoints_by_ncams[ncams])
    print()

    if max_npoints < 10:
        print("not enough points, aborting", file=sys.stderr)
        results.close()
        h5_2d_data.close()
        sys.exit(1)

    IdMat = numpy.array(IdMat, dtype=numpy.uint8).T
    points = numpy.array(points, dtype=numpy.float32).T

    # resolution
    Res = []
    for cam_id in cam_ids:
        image_table = results.root.images
        arr = getattr(image_table, cam_id)
        imsize = arr.shape[1], arr.shape[0]
        Res.append(imsize)
    Res = numpy.array(Res)

    cam_centers = []
    cam_calibrations = []

    if options.camera_center_reconstructor:
        creconstructor = flydra_core.reconstruct.Reconstructor(
            cal_source=options.camera_center_reconstructor
        )
        cam_centers = numpy.asarray(
            [creconstructor.get_camera_center(cam_id)[:, 0] for cam_id in cam_ids]
        )
        flydra_core.reconstruct.save_ascii_matrix(
            cam_centers, os.path.join(calib_dir, "original_cam_centers.dat")
        )

    intrinsics_reconstructor = options.undistort_intrinsics_reconstructor
    if intrinsics_reconstructor:
        tdir = tempfile.mkdtemp()
        reconstructor = flydra_core.reconstruct.Reconstructor(
            cal_source=intrinsics_reconstructor
        )
        for i, cam_id in enumerate(cam_ids):
            fname = os.path.join(tdir, "%s.rad" % cam_id)
            scc = reconstructor.get_SingleCameraCalibration(cam_id)
            scc.helper.save_to_rad_file(fname)
            cam_calibrations.append(fname)

    intrinsics_yaml = options.undistort_intrinsics_yaml
    if intrinsics_yaml:
        for cam_id in cam_ids:
            fname = os.path.join(intrinsics_yaml, "%s.yaml" % cam_id)
            cam_calibrations.append(fname)

    undo_radial_distortion = len(cam_calibrations) == len(cam_ids)

    mcsc = MultiCamSelfCal(calib_dir)
    mcsc.create_calibration_directory(
        cam_ids=cam_ids,
        IdMat=IdMat,
        points=points,
        Res=Res,
        cam_calibrations=cam_calibrations,
        cam_centers=cam_centers,
        radial_distortion=undo_radial_distortion,
        square_pixels=1,
        num_cameras_fill=options.num_cameras_fill,
    )

    results.close()
    h5_2d_data.close()

    if row_keys is not None:
        row_keys = numpy.array(row_keys)
        flydra_core.reconstruct.save_ascii_matrix(
            row_keys, os.path.join(calib_dir, "obj_ids_zero_indexed.dat"), isint=True
        )

    if options.run_mcsc:
        caldir = mcsc.execute()
        print("\nfinished: result in ", caldir)
        if options.output_xml:
            fname = os.path.join(caldir, "reconstructor.xml")
            recon = flydra_core.reconstruct.Reconstructor(cal_source=caldir)
            recon.save_to_xml_filename(fname)
            print("\nfinished: new reconstructor in", fname)

def build_parser():
    usage = "%prog FILE EFILE [options]"

    usage += """

The basic idea is to watch some trajectories with::

  kdviewer <DATAfilename.h5> --n-top-traces=10

Find the top traces, reject any bad ones, and put them in an "efile".

The form of the efile is::

  # Lots of traces
  long_ids = [1,2,3,4]
  # Exclude from above
  bad = [3]

Then run this program::

  flydra_analysis_generate_recalibration <DATAfilename.h5> [EFILENAME] [options]

To ignore 3D trajectories and simply use all data::

  export DATA2D=<DATAfilename.h5>
  flydra_analysis_generate_recalibration --2d-data ${DATA2D} --disable-kalman-objs ${DATA2D}
"""

    parser = OptionParser(usage)

    parser.add_option(
        "--use-nth-observation", type="int", dest="use_nth_observation", default=1
    )

    parser.add_option(
        "--2d-data", type="string", dest="h5_2d_data_filename", default=None
    )

    parser.add_option(
        "--disable-kalman-objs",
        action="store_false",
        default=True,
        dest="use_kalman_data",
    )

    parser.add_option(
        "--start", dest="start", type="int", help="first frame", metavar="START"
    )

    parser.add_option(
        "--stop", dest="stop", type="int", help="last frame", metavar="STOP"
    )

    parser.add_option("--min-num-points", type="int", default=3)

    parser.add_option(
        "--num-cameras-fill",
        type="int",
        help="when a point is missing from one camera, how many other cameras should be should "
        "be used to calculate the position of the missing point. In general, set this to 0 "
        "(disable) or -1 (use all cameras). Only choose other values if you know what you are doing.",
        default=-1,
    )

    parser.add_option(
        "--undistort-intrinsics-yaml",
        help="path to a directory containing .yaml files, as created by the OpenCV/ROS "
        "camera calibration tools. If yaml files corresponding to the ID of every camera "
        "are found then UNDO_RADIAL is set in the multicamselfcal config, and radial distortion "
        "is corrected before computing calibration",
    )

    parser.add_option(
        "--camera-center-reconstructor",
        help="path to a reconstructor dir/xml file. The camera centers are"
        "used to maximally align the new calibration to the centers of the"
        "cameras in this reconstructor",
    )

    parser.add_option(
        "--undistort-intrinsics-reconstructor",
        help="path to a reconstructor dir/xml file. If intrinsics for "
        "every camera are known then radial distortion "
        "is corrected before computing calibration",
    )

    parser.add_option(
        "--run-mcsc",
        action="store_true",
        default=False,
        help="run multicamselfcal on the exported data",
    )

    parser.add_option(
        "--output-xml",
        action="store_true",
        default=False,
        help="save the new reconstructor in xml format",
    )
    return parser

def main():
    parser = build_parser()
    (options, args) = parser.parse_args()

    if len(args) > 2:
        print(
            "arguments interpreted as FILE and EFILE supplied more than once",
            file=sys.stderr,
        )
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]
    if len(args) == 2:
        efilename = args[1]
    else:
        efilename = None
        if options.use_kalman_data is not False:
            raise ValueError(
                "Kalman objects have not been disabled, but you did not specify an EFILE (hint: specify an EFILE or use --disable-kalman-objs"
            )

    call_with_options(h5_filename,
        efilename,options)

def call_with_options(
        filename,
        efilename,
        options):

    return do_it(
        filename,
        efilename,
        use_nth_observation=options.use_nth_observation,
        h5_2d_data_filename=options.h5_2d_data_filename,
        use_kalman_data=options.use_kalman_data,
        start=options.start,
        stop=options.stop,
        options=options,
    )


if __name__ == "__main__":
    main()

def test_generate_recalibration():
    parser = build_parser()
    (options, args) = parser.parse_args()

    options.run_mcsc = True
    options.use_kalman_data = False
    call_with_options(filename="flydra_analysis/a2/sample_datafile-v0.4.28.h5",
        efilename=None, options=options)
