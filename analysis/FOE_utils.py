import result_browser

def get_results_and_times(logfiles,h5files):
    trig_fnos = {}
    tf_hzs = {}
    all_tf_hzs = []
    for logfile in logfiles:
        fd = open(logfile,'rb')
        for line in fd.readlines():
            if line.startswith('#'):
                continue
            fno, ftime, tf_hz = line.split()
            fno = int(fno)
            ftime = float(ftime)
            tf_hz = float(tf_hz)
            trig_fnos[ftime] = fno
            tf_hzs[ftime] = tf_hz
            if tf_hz not in all_tf_hzs:
                all_tf_hzs.append( tf_hz )

    print 'TFs of escape wall:', all_tf_hzs

    all_results = [result_browser.get_results(h5file,mode='r+') for h5file in h5files]
    all_results_times = [result_browser.get_start_stop_times( results ) for results in all_results ]

    # filter to h5 files with 3D data
    all_results       = [all_results[i]       for i in range(len(all_results_times)) if all_results_times[i] is not None]
    all_results_times = [all_results_times[i] for i in range(len(all_results_times)) if all_results_times[i] is not None]

    trig_times = trig_fnos.keys()
    trig_times.sort()
    
    return all_results, all_results_times, trig_fnos, trig_times, tf_hzs
