import numpy
import result_browser

def reject_data( projector_trig_times ):
    # update here to reflect data which is unusable for whatever reason

    # 2006-02-24 datae:
    # Sat, 25 Feb 2006 06:20:33 AM - 10:40 AM
    reject_idx = (projector_trig_times > 1140877233.0) & (projector_trig_times < 1140893467)

    unrejected = ~reject_idx
    return unrejected

def get_results_and_times(logfiles,h5files,get_orig_times=False):
    # open H5 files
    all_results = [result_browser.get_results(h5file,mode='r+') for h5file in h5files]
    all_results_times = [result_browser.get_start_stop_times( results ) for results in all_results ]

    # filter to h5 files with 3D data
    all_results       = [all_results[i]       for i in range(len(all_results_times)) if all_results_times[i] is not None]
    all_results_times = [all_results_times[i] for i in range(len(all_results_times)) if all_results_times[i] is not None]

    # parse log file
    fno_by_projector_trig_time = {}
    tf_hzs = []
    projector_trig_times = []
    fnos = []
    for logfile in logfiles:
        fd = open(logfile,'rb')
        for line in fd.readlines():
            if line.startswith('#'):
                continue
            fno, projector_time, tf_hz = line.split()
            fno = int(fno)
            projector_time = float(projector_time)
            tf_hz = float(tf_hz)

            fnos.append(fno)
            projector_trig_times.append(projector_time)
            tf_hzs.append( tf_hz )

    fnos = numpy.array(fnos,dtype=numpy.int64)
    projector_trig_times = numpy.array(projector_trig_times,dtype=numpy.float64)
    tf_hzs = numpy.array(tf_hzs,dtype=numpy.float64)

    unrejected_idx = reject_data( projector_trig_times )
    
    fnos = fnos[unrejected_idx]
    projector_trig_times = projector_trig_times[unrejected_idx]
    tf_hzs = tf_hzs[unrejected_idx]

    # get original time of each frame
    if get_orig_times:
        print 'finding trigger frames in h5 files...'
        orig_times = []
        for fno in fnos:
            orig_time = None
            for results in all_results:
                data3d = results.root.data3d_best
                for row in data3d.where( data3d.cols.frame == int(fno) ):
                    if orig_time is not None:
                        raise RuntimeError('more than one file/timestamp for frame %d'%fno)
                    orig_time = row['timestamp']
            orig_times.append(orig_time)
        print 'done'

        assert len(fnos)==len(orig_times)
        assert len(fnos)==len(projector_trig_times)

        orig_times = numpy.array(orig_times,dtype=numpy.float64)

    rval = [all_results, all_results_times, fnos, 
            projector_trig_times, tf_hzs]
    
    if get_orig_times:
        rval.append( orig_times )
        
    return tuple(rval)
