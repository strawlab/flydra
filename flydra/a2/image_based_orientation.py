from __future__ import division
from __future__ import with_statement

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, contextlib

import pkg_resources
import numpy as np
import tables as PT
from optparse import OptionParser
import flydra.reconstruct as reconstruct
import motmot.ufmf.ufmf as ufmf
import flydra.a2.utils as utils
import flydra.analysis.result_utils as result_utils
import core_analysis

import matplotlib.pyplot as plt

@contextlib.contextmanager
def openFileSafe(*args,**kwargs):
    result = PT.openFile(*args,**kwargs)
    try:
        yield result
    finally:
        result.close()

def doit(h5_filename=None,
         ufmf_filenames=None,
         kalman_filename=None,
         ):
    """

    Copy all data in .h5 file (specified by h5_filename) to a new .h5
    file in which orientations are set based on image analysis of
    .ufmf files. Tracking data to associate 2D points from subsequent
    frames is read from the .h5 kalman file specified by
    kalman_filename.

    """
    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)
    kalman_observations_2d_idxs = data_file.root.kalman_observations_2d_idxs[:] # cache in RAM
    # TODO: make copy of .h5 file with all 2D orientations set to nan.

    with openFileSafe( h5_filename, mode='r' ) as h5:

        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

        for ufmf_filename in ufmf_filenames:
            fmf = ufmf.FlyMovieEmulator(ufmf_filename)
            fmf_timestamps = fmf.get_all_timestamps()

            camns = cam_id2camns[cam_id]
            assert len(camns)==1
            camn = camns[0]

            # associate framenumbers with timestamps using 2d .h5 file
            h5_timestamps = []
            h5_framenumbers = []
            for row in h5.root.data2d_distorted:
                if not row['camn']==camn:
                    continue
                h5_timestamps.append( row['timestamp'] )
                h5_framenumbers.append( row['frame'] )
            h5_timestamps = np.array(h5_timestamps)
            h5_framenumbers = np.array(h5_framenumbers)

            qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

            for obj_id in use_obj_ids:

                # get all images for this camera and this obj_id

                obj_3d_rows = ca.load_data( obj_id, data_file,
                                            use_kalman_smoothing=False, # no need to take the time
                                            return_smoothed_directions = False,
                                            )
                for this_3d_row in obj_3d_rows:
                    # iterate over each sample in the current camera
                    framenumber = this_3d_row['framenumber']
                    h5_2d_row_idxs = qfi.get_frame_idxs(framenumber)
                    obs_2d_idx = this_3d_row['obs_2d_idx']
                    kobs_2d_data = kalman_observations_2d_idxs[int(obs_2d_idx)]
                    # parse VLArray
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]
                    for obs_camn, obs_camn_pt_no in zip(this_camns, this_camn_idxs):
                        # find 2D point corresponding to object
                        if camn!=obs_camn:
                            continue
                        for h5_2d_row_idx in h5_2d_row_idxs:
                            if obs_camn_pt_no !=

            if 1:
                # guess cam_id
                n = 0
                for cam_id in cam_id2camns.keys():
                    if cam_id in ufmf_filename:
                        n+=1
                if n!=1:
                    print >> sys.stderr, 'Could not automatically determine cam_id from ufmf_filename. Exiting'
                    sys.exit(1)





            plt.plot(h5_timestamps,np.ones( (len(h5_timestamps),)) ,'.')
            plt.plot(fmf_timestamps,np.zeros( (len(fmf_timestamps),)), '.')

        plt.show()
    data_file.close()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--ufmfs", type='string',
                      help="sequence of .ufmf filenames (e.g. 'cam1.ufmf:cam2.ufmf')")

    parser.add_option("--h5", type='string',
                      help=".h5 file with data2d_distorted (REQUIRED)")

    parser.add_option("--kalman", dest="kalman_filename", type='string',
                      help=".h5 file with kalman data and 3D reconstructor")

    (options, args) = parser.parse_args()

    if options.ufmfs is None:
        raise ValueError('--ufmfs option must be specified')

    if options.ufmfs is None:
        raise ValueError('--h5 option must be specified')

    ufmf_filenames = options.ufmfs.split(os.pathsep)

    doit(ufmf_filenames=ufmf_filenames,
         h5_filename=options.h5,
         kalman_filename=options.kalman_filename,
         )

if __name__=='__main__':
    main()
