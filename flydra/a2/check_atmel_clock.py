import tables
import tables.flavor
tables.flavor.restrict_flavors(keep=['numpy']) # ensure pytables 2.x
import numpy
import pylab
import flydra.analysis.flydra_analysis_plot_clock_drift

if 1:
    # load the file
    
    #filename='DATA20070319_172634.h5'
    #filename='DATA20070319_175152.h5'
    #filename = 'DATA20070319_191223.h5'
    #filename = 'DATA20070319_195045.h5'
    filename = 'DATA20070319_201543.h5'
    filename = 'DATA20070319_204145.h5'
    filename = 'DATA20070319_204348.h5'
    filename = 'DATA20071115_202838.h5'
    kresults = tables.openFile(filename,mode="r")

    # get the timer top value
    
    textlog = kresults.root.textlog.readCoordinates([0])
    infostr = textlog['message'].tostring().strip('\x00')
    timer_max = int( textlog['message'].tostring().strip('\x00').split()[-1][:-1] )
    print 'I found the timer maximum ("top") to be %d. I parsed this from "%s"'%(timer_max,infostr)

    # open the log of at90usb clock info
    
    tci = kresults.root.trigger_clock_info
    tbl = tci.read()

    # these are timestamps from the host's (main brain's) clock

    meas_err = (-tbl['start_timestamp'] + tbl['stop_timestamp'])
    print 'meas_err.max() msec',meas_err.max()*1e3
    print 'meas_err.min() msec',meas_err.min()*1e3

    #cond = meas_err < 3e-3 # take data with only small measurement errors
    cond = meas_err >-1e100 # take all data (expect measurement errors to be positive)

    # approximate timestamp (assume symmetric delays) at which clock was sampled
    mb_timestamp = ((tbl['start_timestamp'][cond] + tbl['stop_timestamp'][cond])/2.0)
    
    # get framenumber + fraction of next frame at which mb_timestamp estimated to happen
    framenumber = tbl['framecount'][cond]
    frac = tbl['tcnt'][cond]/float(timer_max)

    # create floating point number with this information
    framestamp = framenumber + frac

    # fit linear model of relationship mainbrain timestamp and usb trigger_device framestamp
    gain, offset = flydra.analysis.flydra_analysis_plot_clock_drift.model_remote_to_local( framestamp, mb_timestamp )

    if 1:
        
        # Calculate reconstruction latencies (this is only valid on original data).
        
        if abs((framestamp[0]*gain + offset) - mb_timestamp[0]) > 1:
            raise RuntimeError('interpolated off by more than 1 second!')

        kest = kresults.root.kalman_estimates.read()
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

                # timestamp of estimated frame time (assumes cam
                # computer time near mainbrain time)
                frame_2d_computer_start_timestamp = tbl['timestamp'][cond]
                
                # timestamp that the frame arrived in the camera
                # computer (assumes cam computer time near mainbrain
                # time)
                frame_2d_computer_timestamp = tbl['cam_received_timestamp'][cond]

                # calculate time of trigger (mainbrain time, which is
                # assumed near camera computer time)
                frame_trigger_timestamp = frame*gain + offset

                # calculate latency to start taking image
                camn_start_latencies_msec = (frame_2d_computer_start_timestamp - frame_trigger_timestamp)*1e3
                # calculate latency to receiving image
                camn_latencies_msec = (frame_2d_computer_timestamp - frame_trigger_timestamp)*1e3

                print 'camn',camn
                #print ' numpy.median(camn_start_latencies_msec)',numpy.min(camn_start_latencies_msec),numpy.median(camn_start_latencies_msec)
                print ' min and median of (camn_latencies_msec)', numpy.min(camn_latencies_msec),numpy.median(camn_latencies_msec)

                #acq_dur = (frame_2d_computer_timestamp-frame_2d_computer_start_timestamp)*1e3
                #print 'acq_dur',numpy.min(acq_dur),numpy.median(acq_dur)
                print

        kresults.close()

        if 0:
            pylab.figure()
            pylab.plot(    framenumber,'.')

    if 1:
        pylab.figure()
        pylab.hist( latency_msec, bins= 250)
        pylab.title('total latency to 3D reconstruction')
        pylab.ylabel('n occurances')
        pylab.xlabel('latency (msec)')

    if 1:
        pylab.figure()
        pylab.plot( framestamp, mb_timestamp- mb_timestamp[0], 'r.', label='mainbrain clock' )
        pylab.plot( framestamp, (framestamp*gain + offset) - mb_timestamp[0], 'b-', label='corrected USB device counter' )
        pylab.legend()
        pylab.ylabel('elapsed time (sec)')
        pylab.xlabel('frame number')
        pylab.title('clock/framecount comparison')
    pylab.show()
