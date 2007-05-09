import numpy
import sys, os, time
import core_analysis
from optparse import OptionParser
import flydra.analysis.flydra_analysis_convert_to_mat
import tables

def convert(infilename,outfilename,frames_per_second=100.0,save_timestamps=True):

    if save_timestamps:
        print 'STAGE 1: finding timestamps'
        print 'opening file %s...'%infilename
        h5file_raw = tables.openFile(infilename,mode='r')
        table_data2d = h5file_raw.root.data2d_distorted # table to get timestamps from
        table_kobs   = h5file_raw.root.kalman_observations # table to get framenumbers from
        kobs_2d = h5file_raw.root.kalman_observations_2d_idxs # VLArray linking two
        
        print 'caching Kalman obj_ids...'
        obs_obj_ids = table_kobs.read(field='obj_id',flavor='numpy')
        print 'finding unique obj_ids...'
        unique_obj_ids = numpy.unique(obs_obj_ids)
        
        print 'finding 2d data for each obj_id...'
        timestamp_time = numpy.zeros( unique_obj_ids.shape, dtype=numpy.float64)
        for obj_id_enum,obj_id in enumerate(unique_obj_ids):
            if obj_id_enum%100==0:
                print '%d of %d'%(obj_id_enum,len(unique_obj_ids))
            valid_cond = obs_obj_ids == obj_id
            idxs = numpy.nonzero(valid_cond)[0]
            idx0 = idxs[0]
            framenumber = table_kobs[idx0]['frame']
            if tables.__version__ <= '1.3.3': # pytables numpy scalar workaround
                framenumber = int(framenumber)
            this_timestamp = numpy.nan
            for row in table_data2d.where(table_data2d.cols.frame == framenumber):
                this_timestamp = row['timestamp']
                break

            timestamp_time[obj_id_enum] = this_timestamp
            if obj_id_enum%100==0:
                try:
                    print time.asctime(time.localtime(this_timestamp))
                except:
                    print '** no timestamp **'
                print
                    
        h5file_raw.close()
            
        extra_vars = {'obj_ids':unique_obj_ids,
                      'timestamps':timestamp_time,
                      }
        print 'STAGE 2: running Kalman smoothing operation'
    else:
        extra_vars = None

    ca = core_analysis.CachingAnalyzer()
    obj_ids = ca.get_obj_ids(infilename)
    allrows = []
    for i,obj_id in enumerate(obj_ids):
        if i%100 == 0:
            print '%d of %d'%(i,len(obj_ids))
        results = ca.get_smoothed(obj_id,
                                  infilename,
                                  frames_per_second=frames_per_second)
        rows = results['kalman_smoothed_rows']
        allrows.append(rows)

        #
        
    allrows = numpy.concatenate( allrows )
    recarray = numpy.rec.array(allrows)
    
    flydra.analysis.flydra_analysis_convert_to_mat.do_it(
        rows=recarray,
        ignore_observations=True,
        newfilename=outfilename,
        extra_vars=extra_vars,
        )
        

def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return

    infilename = args[0]
    outfilename = os.path.splitext(infilename)[0] + '_smoothed.mat'
    convert(infilename,outfilename)
    
if __name__=='__main__':
    main()
