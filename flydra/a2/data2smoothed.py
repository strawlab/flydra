if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import numpy
import sys, os, time
import flydra.a2.core_analysis as core_analysis
from optparse import OptionParser
import flydra.analysis.flydra_analysis_convert_to_mat
import tables
import flydra.analysis.result_utils as result_utils

def cam_id2hostname(cam_id):
    hostname = '_'.join(   cam_id.split('_')[:-1] )
    return hostname

def convert(infilename,
            outfilename,
            frames_per_second=100.0,
            save_timestamps=True,
            file_time_data=None,
            do_nothing=False, # set to true to test for file existance
            ):

    if save_timestamps:
        print 'STAGE 1: finding timestamps'
        print 'opening file %s...'%infilename

        h5file_raw = tables.openFile(infilename,mode='r')
        table_kobs   = h5file_raw.root.kalman_observations # table to get framenumbers from
        kobs_2d = h5file_raw.root.kalman_observations_2d_idxs # VLArray linking two

        if file_time_data is None:
            h52d = h5file_raw
            close_h52d = False
        else:
            h52d = tables.openFile(file_time_data,mode='r')
            close_h52d = True

        try:
            table_data2d = h52d.root.data2d_distorted # Table to get timestamps from. (If you don't have timestamps, use the '--no-timestamps' option.)
            drift_estimates = result_utils.drift_estimates( h52d )
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h52d)
        except:
            print 'Error reading from file',h52d.filename
            raise

        hostnames = drift_estimates['hostnames']
        gain = {}; offset = {};
        for i,hostname in enumerate(hostnames):
            tgain, toffset = result_utils.model_remote_to_local(
                drift_estimates['remote_timestamp'][hostname][::10],
                drift_estimates['local_timestamp'][hostname][::10])
            gain[hostname]=tgain
            offset[hostname]=toffset
            print repr(hostname),tgain,toffset

        if do_nothing:
            h5file_raw.close()
            if close_h52d:
                h52d.close()
            return

        print 'caching Kalman obj_ids...'
        obs_obj_ids = table_kobs.read(field='obj_id')
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

            remote_timestamp = numpy.nan
            this_camn = None
            for row in table_data2d.where('frame == framenumber'):
                this_camn = row['camn']
                remote_timestamp = row['timestamp']
                break

            if this_camn is None:
                print 'skipping frame %d (obj %d): no data2d_distorted data'%(framenumber,obj_id)
                continue

            cam_id = camn2cam_id[this_camn]
            remote_hostname = cam_id2hostname(cam_id)
            mainbrain_timestamp = remote_timestamp*gain[remote_hostname] + offset[remote_hostname] # find mainbrain timestamp

            timestamp_time[obj_id_enum] = mainbrain_timestamp
            if obj_id_enum%100==0:
                try:
                    print time.asctime(time.localtime(mainbrain_timestamp))
                except:
                    print '** no timestamp **'
                print

        h5file_raw.close()
        if close_h52d:
            h52d.close()

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
    parser.add_option("--time-data", dest="file2d", type='string',
                      help="hdf5 file with 2d data FILE2D used to calculate timestamp information",
                      metavar="FILE2D")
    parser.add_option("--no-timestamps",action='store_true',dest='no_timestamps',default=False)
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
    convert(infilename,outfilename,file_time_data=options.file2d,
            save_timestamps = not options.no_timestamps,
            )

if __name__=='__main__':
    main()
