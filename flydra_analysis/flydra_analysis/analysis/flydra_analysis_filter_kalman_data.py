from __future__ import division
from __future__ import print_function

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import numpy
from numpy import nan, pi
import tables as PT
import flydra_core.reconstruct

import sys
from optparse import OptionParser
import flydra_core.kalman.flydra_kalman_utils as flydra_kalman_utils


def do_filter(
    filename, obj_start=None, obj_end=None, obj_only=None, min_length=None,
):
    KalmanEstimates = flydra_kalman_utils.KalmanEstimates
    FilteredObservations = flydra_kalman_utils.FilteredObservations

    output = PT.open_file(filename + ".output", mode="w")
    output_xhat = output.create_table(
        output.root,
        "kalman_estimates",
        KalmanEstimates,
        "Kalman a posteri estimates of tracked object",
    )
    output_obs = output.create_table(
        output.root,
        "ML_estimates",
        FilteredObservations,
        "observations of tracked object",
    )

    kresults = PT.open_file(filename, mode="r")

    reconst = flydra_core.reconstruct.Reconstructor(kresults)
    reconst.save_to_h5file(output)

    obj_ids = kresults.root.kalman_estimates.read(field="obj_id")
    use_obj_ids = obj_ids
    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_end is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]
    if obj_only is not None:
        use_obj_ids = numpy.array(obj_only)
    # find unique obj_ids:
    use_obj_ids = numpy.unique(use_obj_ids)

    objid_by_n_observations = {}
    for obj_id_enum, obj_id in enumerate(use_obj_ids):
        if obj_id_enum % 100 == 0:
            print("reading %d of %d" % (obj_id_enum, len(use_obj_ids)))

        if PT.__version__ <= "1.3.3":
            obj_id_find = int(obj_id)
        else:
            obj_id_find = obj_id

        observation_frame_idxs = kresults.root.ML_estimates.get_where_list(
            "obj_id==obj_id_find"
        )
        observation_frames = kresults.root.ML_estimates.read_coordinates(
            observation_frame_idxs, field="frame"
        )
        max_observation_frame = observation_frames.max()

        row_idxs = numpy.nonzero(obj_ids == obj_id)[0]
        estimate_frames = kresults.root.kalman_estimates.read_coordinates(
            row_idxs, field="x"
        )
        valid_condition = estimate_frames <= max_observation_frame
        row_idxs = row_idxs[valid_condition]
        n_observations = len(observation_frames)

        objid_by_n_observations.setdefault(n_observations, []).append(obj_id)

        if n_observations < min_length:
            print(
                "obj_id %d: %d observation frames, skipping" % (obj_id, n_observations,)
            )
            continue

        obs_recarray = kresults.root.ML_estimates.read_coordinates(
            observation_frame_idxs
        )
        output_obs.append(obs_recarray)
        xhats_recarray = kresults.root.kalman_estimates.read_coordinates(row_idxs)
        output_xhat.append(xhats_recarray)
    output_xhat.flush()
    output_obs.flush()
    output.close()
    kresults.close()


def main():
    usage = """%prog DATAFILE3D.h5 [options]

    Filter DATAFILE3D.h5 to include only ceratin trajectories.
    """

    parser = OptionParser(usage)

    parser.add_option("--obj-only", type="string", dest="obj_only")

    parser.add_option(
        "--min-length",
        type="int",
        help="minimum number of observations required to export",
        dest="min_length",
        default=10,
        metavar="MIN_LENGTH",
    )

    parser.add_option(
        "--start", type="int", help="first object ID to export", metavar="START"
    )

    parser.add_option(
        "--stop", type="int", help="last object ID to export", metavar="STOP"
    )

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = options.obj_only.replace(",", " ")
        seq = map(int, options.obj_only.split())
        options.obj_only = seq

        if options.start is not None or options.stop is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]

    do_filter(
        filename=h5_filename,
        obj_start=options.start,
        obj_end=options.stop,
        obj_only=options.obj_only,
        min_length=options.min_length,
    )


if __name__ == "__main__":
    main()
