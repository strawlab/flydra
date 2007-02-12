from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import datetime

import pylab
import sys, os, pickle, sets

def doit(filename,
         obj_start=None,
         obj_end=None,
         debug=0,
         min_length=10,
         show = False,
         ):
    
    # create bins
    binsize = 0.01 # 10 mm
    x_boundaries = numpy.arange(0.1, 0.8, binsize)
    y_boundaries = numpy.arange(0.02, .38,binsize)
    z_boundaries = numpy.arange(-0.06,.3, binsize)

    # count bins where fly _left_ this bin
    from_counts = numpy.zeros( (x_boundaries.shape[0]+1,
                                y_boundaries.shape[0]+1,
                                z_boundaries.shape[0]+1),
                               dtype=numpy.uint32 )
    
    # count bins where fly _entered_ this bin
    to_counts = numpy.zeros_like(from_counts)

    recompute = True
    output_fname = filename+'.pkl'
    if os.path.exists(output_fname):
        fd = open(output_fname,mode='rb')
        dd = pickle.load(fd)
        loaded_to_counts = dd['to_counts']
        fd.close()

        if loaded_to_counts.shape == to_counts.shape:
            # assume, since shape is same, that parameters haven't changed
            recompute = False
            to_counts = loaded_to_counts

    if recompute:
        # open data file
        kresults = PT.openFile(filename,mode="r")

        obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
        use_obj_ids = obj_ids
        if obj_start is not None:
            use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
        if obj_end is not None:
            use_obj_ids = use_obj_ids[use_obj_ids <= obj_end]
        # find unique obj_ids:
        use_obj_ids = numpy.array(list(sets.Set([int(obj_id) for obj_id in use_obj_ids])))

        if debug >=1:
            print 'DEBUG: obj_ids.min()',obj_ids.min()
            print 'DEBUG: obj_ids.max()',obj_ids.max()

        printed_warning = False

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
            if len(observation_frames) > 0:
                max_observation_frame=observation_frames.max()

                row_idxs = numpy.nonzero( obj_ids == obj_id )[0]
                estimate_frames = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
                valid_condition = estimate_frames <= max_observation_frame
                row_idxs = row_idxs[valid_condition]
                
            n_observations = len( observation_frames )
            
            if n_observations < min_length:
                if not printed_warning:
                    print 'SKIPPING DATA because it is shorter than minimum observation length of %d'%min_length
                    printed_warning = True
                #print 'obj_id %d: %d observation frames, skipping'%(obj_id,n_observations,)
                #print 'obj_id %d: %d frames, skipping'%(obj_id,this_len,)
    
                continue

            xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
            ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
            zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')

            # find bin location for each point
            xidx = x_boundaries.searchsorted( xs )
            yidx = y_boundaries.searchsorted( ys )
            zidx = z_boundaries.searchsorted( zs )

            # find transitions between bins

            bin_diff = ( abs(xidx[1:]-xidx[:-1]) +
                         abs(yidx[1:]-yidx[:-1]) +
                         abs(zidx[1:]-zidx[:-1]) )

            transition_idxs = bin_diff.nonzero()[0]

            from_bin_x_idxs = xidx[transition_idxs]
            from_bin_y_idxs = yidx[transition_idxs]
            from_bin_z_idxs = zidx[transition_idxs]

            to_bin_x_idxs = xidx[transition_idxs+1]
            to_bin_y_idxs = yidx[transition_idxs+1]
            to_bin_z_idxs = zidx[transition_idxs+1]

            for xi,yi,zi in zip( from_bin_x_idxs, from_bin_y_idxs, from_bin_z_idxs ):
                from_counts[xi,yi,zi] += 1
            for xi,yi,zi in zip( to_bin_x_idxs, to_bin_y_idxs, to_bin_z_idxs ):
                to_counts[xi,yi,zi] += 1

        if 1:
            fd = open(output_fname,mode='wb')
            dd = {'to_counts':to_counts,
                  'x_boundaries':x_boundaries,
                  'y_boundaries':y_boundaries,
                  'z_boundaries':z_boundaries}
            pickle.dump(dd,fd)
            fd.close()
            
        kresults.close()

    to_counts_xy = numpy.sum( to_counts, axis=2 )

    if show:
        pylab.imshow(to_counts_xy[1:-1,1:-1],interpolation='nearest')
        pylab.show()
        
def main():
    filename = sys.argv[1]
    doit(filename,show=True)

if __name__=='__main__':
    main()
