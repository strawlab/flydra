from __future__ import division
from __future__ import with_statement

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, contextlib, collections

import pkg_resources
import numpy as np
import tables as PT
from optparse import OptionParser
import flydra.reconstruct as reconstruct
import motmot.ufmf.ufmf as ufmf
import flydra.a2.utils as utils
import flydra.analysis.result_utils as result_utils
import core_analysis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

@contextlib.contextmanager
def openFileSafe(*args,**kwargs):
    result = PT.openFile(*args,**kwargs)
    try:
        yield result
    finally:
        result.close()


def get_cam_id_from_filename(filename, all_cam_ids):
    # guess cam_id
    n = 0
    found_cam_id = None
    for cam_id in all_cam_ids:
        if cam_id in filename:
            n+=1
            if found_cam_id is not None:
                raise ValueError('cam_id found more than once in filename')
            found_cam_id = cam_id
    return found_cam_id

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
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
        kalman_filename)
    kalman_observations_2d_idxs = data_file.root.kalman_observations_2d_idxs[:]

    # TODO: make copy of .h5 file with all 2D orientations set to nan.
    with openFileSafe( h5_filename, mode='r' ) as h5:

        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

        cam_id2fmfs = collections.defaultdict(list)
        for ufmf_filename in ufmf_filenames:
            fmf = ufmf.FlyMovieEmulator(ufmf_filename)
            timestamps = fmf.get_all_timestamps()

            cam_id = get_cam_id_from_filename( fmf.filename, cam_id2camns.keys() )
            cam_id2fmfs[cam_id].append( (fmf,
                                         result_utils.Quick1DIndexer(timestamps)))



        # associate framenumbers with timestamps using 2d .h5 file
        data2d = h5.root.data2d_distorted[:] # load to RAM
        h5_framenumbers = data2d['frame']
        h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

        for obj_id in use_obj_ids:

            # get all images for this camera and this obj_id

            obj_3d_rows = ca.load_dynamics_free_MLE_position( obj_id, data_file)

            for this_3d_row in obj_3d_rows:
                # iterate over each sample in the current camera
                framenumber = this_3d_row['frame']
                h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)

                frame2d = data2d[h5_2d_row_idxs]

                obs_2d_idx = this_3d_row['obs_2d_idx']
                kobs_2d_data = kalman_observations_2d_idxs[int(obs_2d_idx)]

                # parse VLArray
                this_camns = kobs_2d_data[0::2]
                this_camn_idxs = kobs_2d_data[1::2]
                for obs_camn, obs_camn_pt_no in zip(this_camns, this_camn_idxs):
                    # find 2D point corresponding to object
                    cam_id = camn2cam_id[obs_camn]

                    movie_tups_for_this_camn = cam_id2fmfs[cam_id]
                    cond = ((frame2d['camn']==obs_camn) &
                            (frame2d['frame_pt_idx']==obs_camn_pt_no))
                    idxs = np.nonzero(cond)[0]
                    assert len(idxs)==1
                    idx = idxs[0]
                    frame_timestamp = frame2d[idx]['timestamp']
                    found = None
                    for fmf, fmf_timestamp_qi in movie_tups_for_this_camn:
                        fmf_fnos = fmf_timestamp_qi.get_idxs(frame_timestamp)
                        if not len(fmf_fnos):
                            continue
                        assert len(fmf_fnos)==1

                        # should only be one .ufmf with this frame and cam_id
                        assert found is None

                        fmf_fno = fmf_fnos[0]
                        found = (fmf, fmf_fno )
                    if found is None:
                        print 'no image data for frame timestamp %s cam_id %s'%(
                            repr(frame_timestamp),cam_id)
                        continue
                    fmf, fmf_fno = found
                    image, fmf_timestamp = fmf.get_frame( fmf_fno )

                    fname = 'obj%05d_camn%03d_frame%07d_pt%02d.png'%(
                        obj_id,obs_camn,framenumber,obs_camn_pt_no)
                    print 'saving',fname
                    if 1:
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        ax.imshow( image, origin='lower', cmap=matplotlib.cm.gray )

                        ax.plot( frame2d[idxs]['x'], frame2d[idxs]['y'], 'rx' )
                        fig.savefig(fname)
                        plt.close(fig)
                        del fig






        ##     plt.plot(h5_timestamps,np.ones( (len(h5_timestamps),)) ,'.')
        ##     plt.plot(fmf_timestamps,np.zeros( (len(fmf_timestamps),)), '.')

        ## plt.show()
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

    if options.h5 is None:
        raise ValueError('--h5 option must be specified')

    if options.kalman_filename is None:
        raise ValueError('--kalman option must be specified')

    ufmf_filenames = options.ufmfs.split(os.pathsep)
    ## print 'ufmf_filenames',ufmf_filenames
    ## print 'options.h5',options.h5

    doit(ufmf_filenames=ufmf_filenames,
         h5_filename=options.h5,
         kalman_filename=options.kalman_filename,
         )

if __name__=='__main__':
    main()
