import tables
import tables.flavor
tables.flavor.restrict_flavors(keep=['numpy']) # ensure pytables 2.x
import numpy
import pylab
import flydra.analysis.flydra_analysis_plot_clock_drift

if 1:
    #filename='DATA20070319_172634.h5'
    #filename='DATA20070319_175152.h5'
    #filename = 'DATA20070319_191223.h5'
    #filename = 'DATA20070319_195045.h5'
    filename = 'DATA20070319_201543.h5'
    filename = 'DATA20070319_204145.h5'
    filename = 'DATA20070319_204348.h5'
    filename = 'DATA20071115_202838.h5'
    kresults = tables.openFile(filename,mode="r")

    textlog = kresults.root.textlog.readCoordinates([0])
    infostr = textlog['message'].tostring().strip('\x00')
    timer_max = int( textlog['message'].tostring().strip('\x00').split()[-1][:-1] )
    print 'I found the timer maximum ("top") to be %d. I parsed this from "%s"'%(timer_max,infostr)
    
    tci = kresults.root.trigger_clock_info
    tbl = tci.read()

    kest = kresults.root.kalman_estimates.read()

    meas_err = (-tbl['start_timestamp'] + tbl['stop_timestamp'])
    print 'meas_err.max()',meas_err.max()*1e3
    print 'meas_err.min()',meas_err.min()*1e3

    #cond = meas_err < 3e-3
    cond = meas_err >-1e100
    mb_timestamp = ((tbl['start_timestamp'][cond] + tbl['stop_timestamp'][cond])/2.0)
    framenumber = tbl['framecount'][cond]
    
    frac = tbl['tcnt'][cond]/float(timer_max)

    print "tbl['tcnt'][cond][-1]",tbl['tcnt'][cond][-1]
    print 'framenumber[:5]',framenumber[:5]
    print 'frac[:5]',frac[:5]
    framestamp = framenumber + frac
    print 'framestamp[:5]',framestamp[:5]
    print 'framenumber.argmax()',framenumber.argmax()

    if 0:
        #gain, offset0 = flydra.analysis.flydra_analysis_plot_clock_drift.model_remote_to_local( mb_timestamp- mb_timestamp[0], framestamp )
        gain, offset0 = flydra.analysis.flydra_analysis_plot_clock_drift.model_remote_to_local( framestamp, mb_timestamp- mb_timestamp[0])
        offset = offset0 + mb_timestamp[0]
    else:
        gain, offset = flydra.analysis.flydra_analysis_plot_clock_drift.model_remote_to_local( framestamp, mb_timestamp )

    if 1:

        if abs((framestamp[0]*gain + offset) - mb_timestamp[0]) > 1:
            raise RuntimeError('interpolated off by more than 1 second!')

        done_time = kest['timestamp']
        cond = done_time != 0.0
        done_time = done_time[cond]

        done_frame = kest['frame'][cond]
        start_time = done_frame*gain + offset

        latency = done_time-start_time
        latency_msec = latency*1e3

        lmsi = numpy.argsort(latency_msec)
        N = len(lmsi)
        num95 = int(N*0.95)
        num90 = int(N*0.9)
        latency95 = latency_msec[lmsi[num95]]

        print 
        print 'numpy.min(latency_msec),numpy.median(latency_msec)',numpy.min(latency_msec),numpy.median(latency_msec)
        print
        if 0:
            median = numpy.median
            print 'latency_msec[:num95].median()',median(latency_msec[lmsi[:num95]])
            print 'latency_msec[:num90].median()',median(latency_msec[lmsi[:num90]])
            print 'latency_msec[lmsi[int(N*0.95)]]',latency_msec[lmsi[int(N*0.95)]]
            print 'latency_msec[lmsi[int(N*0.9)]]',latency_msec[lmsi[int(N*0.9)]]
            print 'latency_msec[lmsi[int(N*0.85)]]',latency_msec[lmsi[int(N*0.85)]]
            print 'latency_msec[lmsi[int(N*0.8)]]',latency_msec[lmsi[int(N*0.8)]]
            
        if 1:
            #####################
            # 2D camera computer timestamps
            tbl = kresults.root.data2d_distorted.read()

            camns = tbl['camn']
            ucamns = numpy.unique(camns)
            ucamns.sort()
            for camn in ucamns:
                cond = camns == camn
                frame = tbl['frame'][cond]
                
                frame_2d_computer_start_timestamp = tbl['timestamp'][cond]
                frame_2d_computer_timestamp = tbl['cam_received_timestamp'][cond]
                
                frame_trigger_timestamp = frame*gain + offset
                
                camn_start_latencies_msec = (frame_2d_computer_start_timestamp - frame_trigger_timestamp)*1e3
                camn_latencies_msec = (frame_2d_computer_timestamp - frame_trigger_timestamp)*1e3

                print camn
                print ' numpy.median(camn_start_latencies_msec)',numpy.min(camn_start_latencies_msec),numpy.median(camn_start_latencies_msec)
                print ' numpy.median(camn_latencies_msec.)', numpy.min(camn_latencies_msec),numpy.median(camn_latencies_msec)

                acq_dur = (frame_2d_computer_timestamp-frame_2d_computer_start_timestamp)*1e3
                print 'acq_dur',numpy.min(acq_dur),numpy.median(acq_dur)
                print

        kresults.close()

        if 0:
            pylab.plot(    framenumber,'.')
            pylab.show()

    if 1:
        pylab.hist( latency_msec, bins= 250)
        pylab.xlabel('latency (msec)')
        pylab.show()

    if 0:
    
        pylab.plot( framestamp, mb_timestamp- mb_timestamp[0], 'r.' )
        pylab.plot( framestamp, (framestamp*gain + offset) - mb_timestamp[0], 'b-' )
        pylab.ylabel('time elapsed (sec)')
        pylab.xlabel('frame')
        pylab.show()
