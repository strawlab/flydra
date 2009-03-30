from __future__ import with_statement
import motmot.ufmf.ufmf as ufmf_mod
import sys, os, tempfile, re, contextlib, warnings
from optparse import OptionParser
import flydra.a2.auto_discover_ufmfs as auto_discover_ufmfs
import numpy as np
import tables
import flydra.a2.utils as utils
import flydra.analysis.result_utils as result_utils
import scipy.misc
import subprocess

ufmf_fname_regex = re.compile(r'small_([0-9]+)_([0-9]+)_(.*)')
def get_cam_id_from_ufmf_fname(ufmf_fname):
    a = os.path.split( ufmf_fname )[-1]
    b = os.path.splitext(a)[0]
    matchobj = ufmf_fname_regex.search(b)
    date, time, cam_id= matchobj.groups()
    return cam_id

@contextlib.contextmanager
def openFileSafe(*args,**kwargs):
    result = tables.openFile(*args,**kwargs)
    try:
        yield result
    finally:
        result.close()

def get_tile(N):
    rows = int(np.ceil(np.sqrt(float(N))))
    cols = rows
    return '%dx%d'%(rows,cols)

def make_montage( h5_filename,
                  ufmf_dir=None,
                  dest_dir = None,
                  white_background=False,
                  save_ogv_movie = False,
                  no_remove = False,
                  max_n_frames = None,
                  start = None,
                  stop = None,
                  ):
    ufmf_fnames = auto_discover_ufmfs.find_ufmfs( h5_filename,
                                                  ufmf_dir=ufmf_dir,
                                                  careful=True )

    if dest_dir is None:
        dest_dir = os.curdir
    else:
        if not os.path.exists( dest_dir ):
            os.makedirs(dest_dir)

    # First pass: get intersection of timestamps
    first_ufmf_ts = -np.inf
    last_ufmf_ts = np.inf
    ufmfs = {}
    blank_images = {}
    cam_ids = []
    for ufmf_fname in ufmf_fnames:
        cam_id = get_cam_id_from_ufmf_fname(ufmf_fname)
        cam_ids.append( cam_id )
        ufmf = ufmf_mod.FlyMovieEmulator(ufmf_fname,
                                         white_background=white_background,
                                         )
        tss = ufmf.get_all_timestamps()
        ufmfs[ufmf_fname] = (ufmf, cam_id, tss)
        min_ts = np.min(tss)
        max_ts = np.max(tss)
        if min_ts > first_ufmf_ts:
            first_ufmf_ts = min_ts
        if max_ts < last_ufmf_ts:
            last_ufmf_ts = max_ts
        blank_images[ufmf_fname] = 255*np.ones(
            (ufmf.get_height(),ufmf.get_width()), dtype=np.uint8)

    # get name of data

    datetime_str = os.path.splitext(os.path.split(h5_filename)[-1])[0]
    datetime_str = datetime_str[4:19]

    print '%s: timestamp overlap: %s-%s'%(datetime_str,
                                          repr(first_ufmf_ts),
                                          repr(last_ufmf_ts))
    assert first_ufmf_ts < last_ufmf_ts, ".ufmf files don't all overlap in time"

    ufmf_fnames.sort()
    cam_ids.sort()

    with openFileSafe( h5_filename, mode='r' ) as h5:
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        parsed = result_utils.read_textlog_header(h5)
        flydra_version = parsed.get('flydra_version',None)
        if flydra_version is not None and flydra_version >= '0.4.45':
            # camnode.py saved timestamps into .ufmf file given by
            # time.time() (camn_receive_timestamp). Compare with
            # mainbrain's data2d_distorted column
            # 'cam_received_timestamp'.
            old_camera_timestamp_source = False
        else:
            # camnode.py saved timestamps into .ufmf file given by
            # camera driver. Compare with mainbrain's data2d_distorted
            # column 'timestamp'.
            old_camera_timestamp_source = True
        h5_data = h5.root.data2d_distorted[:]
        if old_camera_timestamp_source:
            h5_timestamps = h5_data['timestamp']
        else:
            h5_timestamps = h5_data['cam_received_timestamp']

        h5_frames = h5_data['frame']
        h5_camns = h5_data['camn']

    cam_id2camn = {}
    for cam_id in cam_ids:
        camns = cam_id2camns[cam_id]
        assert len(camns)==1, "can't handle multiple camns per cam_id"
        cam_id2camn[cam_id] = camns[0]

    if 1:
        cond = ((first_ufmf_ts <= h5_timestamps) &
                (h5_timestamps <= last_ufmf_ts))
        use_frames = h5_frames[cond]
        ff = utils.FastFinder(h5_frames)
        unique_frames = list(np.unique1d(use_frames))
        unique_frames.sort()
        unique_frames = np.array( unique_frames )
        if start is not None:
            unique_frames = unique_frames[ unique_frames >= start ]
        if stop is not None:
            unique_frames = unique_frames[ unique_frames <= stop ]

        if max_n_frames is not None:
            unique_frames = unique_frames[:max_n_frames]
        all_frame_montages = []
        for frame_enum,frame in enumerate(unique_frames):
            if (frame_enum%100)==0:
                print '%s: frame %d of %d'%(datetime_str,
                                            frame_enum,len(unique_frames))
            idxs = ff.get_idxs_of_equal(frame)

            # trim data under consideration to just this frame
            this_camns = h5_camns[idxs]
            this_tss = h5_timestamps[idxs]

            this_frames = h5_frames[idxs]

            saved_fnames = []
            for ufmf_fname in ufmf_fnames:
                ufmf, cam_id, tss = ufmfs[ufmf_fname]
                camn = cam_id2camn[cam_id]
                this_camn_cond = this_camns == camn
                this_camn_tss = this_tss[this_camn_cond]
                if len(this_camn_tss):
                    this_camn_ts=np.unique1d(this_camn_tss)
                    assert len(this_camn_ts)==1
                    this_camn_ts = this_camn_ts[0]

                    # optimistic: get next frame. it's probably the one we want
                    image, image_ts = ufmf.get_next_frame()
                    if this_camn_ts != image_ts:
                        print 'fail',frame_enum,cam_id
                        # It was not the frame we wanted. Find it.
                        ufmf_frame_idxs = np.nonzero(tss == this_camn_ts)[0]
                        if (len(ufmf_frame_idxs)==0 and
                            old_camera_timestamp_source):
                            warnings.warn(
                                'low-precision timestamp comparison in '
                                'use due to outdated .ufmf timestamp '
                                'saving.')
                            # 2.5 msec precision required
                            ufmf_frame_idxs = np.nonzero(
                                abs( tss - this_camn_ts ) < 0.0025)[0]
                        assert len(ufmf_frame_idxs)==1
                        ufmf_frame_no = ufmf_frame_idxs[0]
                        image, image_ts = ufmf.get_frame(ufmf_frame_no)
                        del ufmf_frame_no, ufmf_frame_idxs
                else:
                    # no image data this frame
                    image = blank_images[ufmf_fname]
                save_fname = 'tmp_frame%07d_%s.bmp'%(frame,cam_id)
                save_fname_path = os.path.join(dest_dir, save_fname)
                scipy.misc.pilutil.imsave(save_fname_path, image)
                saved_fnames.append( save_fname_path )

            target = os.path.join(dest_dir, 'movie%s_frame%07d.jpg'%(
                datetime_str,frame_enum+1 ))
            tile = get_tile( len(saved_fnames) )
            imnames = ' '.join(saved_fnames)
            # All cameras saved for this frame, make montage
            CMD=("montage %s -mode Concatenate -tile %s -bordercolor white "
                 "-title '%s frame %d' "
                 "-border 2 %s"%(imnames, tile, datetime_str, frame, target))
            #print CMD
            subprocess.check_call(CMD,shell=True)
            all_frame_montages.append( target )
            if not no_remove:
                for fname in saved_fnames:
                    os.unlink(fname)

    if save_ogv_movie:
        orig_dir = os.path.abspath(os.curdir)
        os.chdir(dest_dir)
        try:
            CMD = 'ffmpeg2theora -v 10 movie%s_frame%%07d.jpg -o movie%s.ogv'%(
                datetime_str,datetime_str)
            subprocess.check_call(CMD,shell=True)
        finally:
            os.chdir(orig_dir)

        if not no_remove:
            for fname in all_frame_montages:
                os.unlink(fname)

def main():
    usage = '%prog DATAFILE2D.h5 [options]'

    parser = OptionParser(usage)

    parser.add_option("--dest-dir", type='string',
                      help="destination directory to save resulting files")

    parser.add_option("--ufmf-dir", type='string',
                      help="directory with .ufmf files")

    parser.add_option("--max-n-frames", type='int', default=None,
                      help="maximum number of frames to save")

    parser.add_option("--start", type='int', default=None,
                      help="start frame")

    parser.add_option("--stop", type='int', default=None,
                      help="stop frame")

    parser.add_option("--ogv", action='store_true', default=False,
                      help="export .ogv video")

    parser.add_option('-n', "--no-remove", action='store_true', default=False,
                      help="don't remove intermediate images")

    parser.add_option("--white-background", action='store_true', default=False,
                      help="don't display background information")

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    h5_filename = args[0]
    make_montage( h5_filename,
                  ufmf_dir = options.ufmf_dir,
                  dest_dir = options.dest_dir,
                  save_ogv_movie = options.ogv,
                  no_remove = options.no_remove,
                  white_background = options.white_background,
                  max_n_frames = options.max_n_frames,
                  start = options.start,
                  stop = options.stop,
                  )
