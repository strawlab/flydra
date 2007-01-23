from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import flydra.reconstruct
import sets

import sys
from optparse import OptionParser
import flydra.kalman.flydra_kalman_utils as flydra_kalman_utils

def do_filter(filename,
              obj_start=None,
              obj_end=None,
              min_length=None,
              ):
    KalmanEstimates = flydra_kalman_utils.KalmanEstimates
    FilteredObservations = flydra_kalman_utils.FilteredObservations
    
    output = PT.openFile(filename+'.output',mode="w")
    output_xhat = output.createTable(output.root,'kalman_estimates', KalmanEstimates,
                                     "Kalman a posteri estimates of tracked object")
    output_obs = output.createTable(output.root,'kalman_observations', FilteredObservations,
                                    "observations of tracked object")
    
    kresults = PT.openFile(filename,mode="r")
    
    reconst = flydra.reconstruct.Reconstructor(kresults)
    reconst.save_to_h5file(output)

    obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
    use_obj_ids = obj_ids
    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_end is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]
    # find unique obj_ids:
    use_obj_ids = numpy.array(list(sets.Set([int(obj_id) for obj_id in use_obj_ids])))

    objid_by_n_observations = {}
    for obj_id_enum,obj_id in enumerate(use_obj_ids):
        if obj_id_enum%100==0:
            print 'reading %d of %d'%(obj_id_enum,len(use_obj_ids))
        
        if PT.__version__ <= '1.3.3':
            obj_id_find=int(obj_id)
        else:
            obj_id_find=obj_id

        observation_frame_idxs = kresults.root.kalman_observations.getWhereList(
            kresults.root.kalman_observations.cols.obj_id==obj_id_find,
            flavor='numpy')
        observation_frames = kresults.root.kalman_observations.readCoordinates(
            observation_frame_idxs,
            field='frame',
            flavor='numpy')
        max_observation_frame=observation_frames.max()

        row_idxs = numpy.nonzero( obj_ids == obj_id )[0]
        estimate_frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
        valid_condition = estimate_frames <= max_observation_frame
        row_idxs = row_idxs[valid_condition]
        n_observations = len( observation_frames )
        
        objid_by_n_observations.setdefault(n_observations,[]).append(obj_id)
            
        
        if n_observations < min_length:
            print 'obj_id %d: %d observation frames, skipping'%(obj_id,n_observations,)
            continue

        obs_recarray = kresults.root.kalman_observations.readCoordinates(
            observation_frame_idxs,flavor='numpy')
        output_obs.append( obs_recarray )
        xhats_recarray = kresults.root.kalman_estimates.readCoordinates(row_idxs,flavor='numpy')
        output_xhat.append( xhats_recarray )
    output_xhat.flush()
    output_obs.flush()
    output.close()
    kresults.close()

def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

    parser.add_option("--min-length", type="int",
                      help="minimum number of observations required to plot",
                      dest="min_length",
                      default=10,
                      metavar="MIN_LENGTH")
    
    (options, args) = parser.parse_args()

    if options.filename is not None:
        args.append(options.filename)
        
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return
        
    h5_filename=args[0]

    do_filter(filename=h5_filename,
              min_length = options.min_length,
              )
    
if __name__=='__main__':
    main()
