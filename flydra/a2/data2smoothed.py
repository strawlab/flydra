import numpy
import sys, os
import core_analysis
from optparse import OptionParser
import flydra.analysis.flydra_analysis_convert_to_mat

def convert(infilename,outfilename,frames_per_second=100.0):
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
    allrows = numpy.concatenate( allrows )
    recarray = numpy.rec.array(allrows)
    flydra.analysis.flydra_analysis_convert_to_mat.do_it(
        rows=recarray,
        ignore_observations=True,
        newfilename=outfilename)
        

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
