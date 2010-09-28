# This file forked from flydra_analysis_generate_recalibration.py
from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sys, os, sets
from optparse import OptionParser
import flydra.reconstruct
import flydra.analysis.result_utils as result_utils
import numpy
from flydra.reconstruct import save_ascii_matrix

def create_new_row(d2d, this_camns, this_camn_idxs, cam_ids, camn2cam_id, npoints_by_cam_id):
    n_pts = 0
    IdMat_row = []
    points_row = []
    for cam_id in cam_ids:
        found = False
        for this_camn,this_camn_idx in zip(this_camns,this_camn_idxs):
            if camn2cam_id[this_camn] != cam_id:
                continue

            this_camn_d2d = d2d[d2d['camn'] == this_camn]
            for this_row in this_camn_d2d: # XXX could be sped up
                if this_row['frame_pt_idx'] == this_camn_idx:
                    found = True
                    break
        if not found:
            IdMat_row.append( 0 )
            points_row.extend( [numpy.nan, numpy.nan, numpy.nan] )
        else:
            npoints_by_cam_id[cam_id] = npoints_by_cam_id[cam_id] + 1
            n_pts += 1
            IdMat_row.append( 1 )
            points_row.extend( [this_row['x'], this_row['y'], 1.0] )
    return IdMat_row, points_row

def do_it(filename,
          use_nth_observation=None,
          start=None,
          stop=None,
          options=None,
          ):

    h5_2d_data_filename = filename

    calib_dir = filename+'.recal'
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    results = result_utils.get_results(filename,mode='r')
    if False and hasattr(results.root,'calibration'):
        # get image size from saved reconstructor?
        reconstructor = flydra.reconstruct.Reconstructor(results)
    else:
        reconstructor = None

    h5_2d_data = results
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5_2d_data)

    cam_ids = cam_id2camns.keys()
    cam_ids.sort()

    data2d = h5_2d_data.root.data2d_distorted
    frames = data2d.cols.frame[:]
    qfi = result_utils.QuickFrameIndexer(frames)

    npoints_by_ncams = {}

    npoints_by_cam_id = {}
    for cam_id in cam_ids:
        npoints_by_cam_id[cam_id] = 0

    IdMat = []
    points = []

    if start is None:
        start = 0
    if stop is None:
        stop = frames.max()

    row_keys = None
    count = 0
    for frameno in range(int(start),int(stop+1),use_nth_observation):
        this_use_idxs=qfi.get_frame_idxs(frameno)

        d2d = data2d.readCoordinates( this_use_idxs )
        d2d = d2d[ ~numpy.isnan(d2d['x']) ]
        this_camns = d2d['camn']

        unique_camns = numpy.unique(this_camns)
        if len(this_camns) != len(unique_camns):
            # ambiguity - a camera has > 1 point
            continue
        this_camn_idxs = numpy.array([0]*len(this_camns))

        if len(this_camns) < options.min_num_points:
            # not enough points to contribute to calibration
            continue

        npoints_by_ncams[ len(this_camns) ] = npoints_by_ncams.get( len(this_camns), 0 ) + 1
        count +=1

        IdMat_row, points_row = create_new_row( d2d, this_camns, this_camn_idxs, cam_ids, camn2cam_id, npoints_by_cam_id )
        IdMat.append( IdMat_row )
        points.append( points_row )
    print '%d points'%len(IdMat)

    print 'by camera id:'
    for cam_id in cam_ids:
        print ' %s: %d'%(cam_id, npoints_by_cam_id[cam_id])
    print 'by n points:'
    for ncams in npoints_by_ncams:
        print ' %d: %d'%(ncams, npoints_by_ncams[ncams])
    print

    IdMat = numpy.array(IdMat,dtype=numpy.uint8).T
    points = numpy.array(points,dtype=numpy.float32).T

    # resolution
    Res = []
    for cam_id in cam_ids:
        if reconstructor is not None:
            imsize = reconstructor.get_resolution(cam_id)
        else:
            image_table = results.root.images
            arr = getattr(image_table,cam_id)
            imsize = arr.shape[1], arr.shape[0]
        Res.append( imsize )
    Res = numpy.array( Res )

    if reconstructor is not None:
        fd = open(os.path.join(calib_dir,'calibration_units.txt'),mode='w')
        fd.write(reconstructor.get_calibration_unit()+'\n')
        fd.close()

    results.close()

    if reconstructor is not None:
        cam_centers = numpy.asarray([reconstructor.get_camera_center(cam_id)[:,0]
                                     for cam_id in cam_ids])
        save_ascii_matrix(cam_centers,os.path.join(calib_dir,'original_cam_centers.dat'))
    save_ascii_matrix(IdMat,os.path.join(calib_dir,'IdMat.dat'))
    save_ascii_matrix(points,os.path.join(calib_dir,'points.dat'))
    save_ascii_matrix(Res,os.path.join(calib_dir,'Res.dat'),isint=True)

    fd = open(os.path.join(calib_dir,'camera_order.txt'),'w')
    for cam_id in cam_ids:
        fd.write('%s\n'%cam_id)
    fd.close()

    if row_keys is not None:
        row_keys = numpy.array(row_keys)
        save_ascii_matrix(row_keys,os.path.join(calib_dir,'obj_ids_zero_indexed.dat'),isint=True)

def main():
    usage = '%prog FILE'

    parser = OptionParser(usage)

    parser.add_option('--use-nth-observation', type='int',
                      dest='use_nth_observation', default=1)

    parser.add_option("--start", dest='start',
                      type="int",
                      help="first frame",
                      metavar="START")

    parser.add_option("--stop", dest='stop',
                      type="int",
                      help="last frame",
                      metavar="STOP")

    parser.add_option("--min-num-points",
                      type="int",
                      default=3)

    (options, args) = parser.parse_args()

    if len(args)>1:
        print >> sys.stderr,  "argument interpreted as FILE supplied more than once"
        parser.print_help()
        return

    if len(args)<1:
        parser.print_help()
        return
    assert len(args)==1

    h5_filename=args[0]

    do_it(h5_filename,
          use_nth_observation=options.use_nth_observation,
          start=options.start,
          stop=options.stop,
          options=options,
          )

if __name__=='__main__':
    main()
