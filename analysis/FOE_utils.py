import numpy
import result_browser
import datetime
import time
import pytz # from http://pytz.sourceforge.net/
time_fmt = '%Y-%m-%d %H:%M:%S %Z%z'

def reject_data( trig_times ):
    pacific = pytz.timezone('US/Pacific')

    # convert to datetime objects
    att = numpy.array( [datetime.datetime.fromtimestamp(x,pacific)
                        for x in trig_times] )
    
    # update here to reflect data which is unusable for whatever reason
    dt = datetime.datetime
    reject_idx = numpy.zeros( att.shape, dtype=numpy.Bool) # all false
    
    # 2006-02-14 data: # frame numbers off, can't trust data
    bad_start = dt(2006, 2, 14, 23, 4, 25, tzinfo=pacific)
    bad_stop = dt(2006, 2, 14, 23, 54, 8, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))
    
    # 2006-02-20 data:
    # Sat, 25 Feb 2006 06:20:33 AM - 10:51:07 AM # tail end of experiments
    bad_start = dt(2006, 2, 21,  6, 20, 33, tzinfo=pacific)
    bad_stop  = dt(2006, 2, 21, 12, 00, 00, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

    # 2006-02-24 data:
    # Sat, 25 Feb 2006 06:20:33 AM - 10:51:07 AM # tail end of experiments
    bad_start = dt(2006, 2, 25,  6, 20, 33, tzinfo=pacific)
    bad_stop  = dt(2006, 2, 25, 10, 51, 07, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

    # 2006-02-25 data:
    # Sat, 25 Feb 2006 01:03:00 PM - 6:30 PM # missed frames on cam5:0
    bad_start = dt(2006, 2, 25, 13, 03, 00, tzinfo=pacific)
    bad_stop  = dt(2006, 2, 25, 18, 30, 00, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

    # Sun, 26 Feb 2006 08:49:00 AM - 6:30 PM # missed frames on cam2:0
    bad_start = dt(2006, 2, 26,  8, 49, 00, tzinfo=pacific)
    bad_stop  = dt(2006, 2, 26, 18, 30, 00, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

    # 10 Mar 2006 00:35:00 AM - 2:00 PM # missed frames on cam1:0
    bad_start = dt(2006, 3, 10,  0, 35, 00, tzinfo=pacific)
    bad_stop  = dt(2006, 3, 10, 14, 00, 00, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

    # Mon, 13 Mar 2006 10:30:00 AM - 4:00 PM # missed frames on cam3:0
    bad_start = dt(2006, 3, 13, 10, 30, 00, tzinfo=pacific)
    bad_stop  = dt(2006, 3, 13, 16, 00, 00, tzinfo=pacific)
    reject_idx = reject_idx | ((att > bad_start) & (att < bad_stop))

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
    fno_by_logfile_trig_time = {}
    stimulus_conditions = []
    logfile_trig_times = []
    fnos = []
    film_trig_fnos = []
    film_trig_times = []
    for logfile in logfiles:
        fd = open(logfile,'rb')
        for line in fd.readlines():
            if line.startswith('#'):
                if line.startswith('# film trigger'):
                    tmp, tmp2, tmp3, fno, projector_time = line.split()
                    fno = int(fno)
                    projector_time = float(projector_time)
                    film_trig_fnos.append( fno )
                    film_trig_times.append( projector_time )
                continue
            fno, projector_time, stimulus_condition = line.split()
            fno = int(fno)
            projector_time = float(projector_time)

            fnos.append(fno)
            logfile_trig_times.append(projector_time)
            stimulus_conditions.append( stimulus_condition )
            
    #film_trig_fnos = numpy.array(film_trig_fnos)
    #film_trig_times = numpy.array(film_trig_times)
    fnos = numpy.array(fnos,dtype=numpy.int64)
    logfile_trig_times = numpy.array(logfile_trig_times,dtype=numpy.float64)
    stimulus_conditions = numpy.array(stimulus_conditions)#,dtype=numpy.float64)

    unrejected_idx = reject_data( logfile_trig_times )
    
    fnos = fnos[unrejected_idx]
    logfile_trig_times = logfile_trig_times[unrejected_idx]
    stimulus_conditions = stimulus_conditions[unrejected_idx]

    for fno,ts in zip(film_trig_fnos,film_trig_times):
        for results in all_results:
            data3d = results.root.data3d_best
            for row in data3d.where( data3d.cols.frame == int(fno) ):
                tts = time.strftime(time_fmt, time.localtime(ts))
                print 'trigger at frame %d (%s): %.1f %.1f %.1f'%(
                    fno, tts, row['x'], row['y'], row['z'])

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
        assert len(fnos)==len(logfile_trig_times)

        orig_times = numpy.array(orig_times,dtype=numpy.float64)

    rval = [all_results, all_results_times, fnos, 
            logfile_trig_times, stimulus_conditions]
    
    if get_orig_times:
        rval.append( orig_times )
        
    return tuple(rval)
