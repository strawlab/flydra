import tables
import tables.flavor
tables.flavor.restrict_flavors(keep=['numpy']) # ensure pytables 2.x
import numpy
import pylab
import flydra.analysis.result_utils

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
    time_model, full_output = flydra.analysis.result_utils.get_time_model_from_data(kresults,
                                                                                    debug=True,
                                                                                    full_output=True)
    framestamp = full_output['framestamp']
    mb_timestamp = full_output['mb_timestamp']
    gain = full_output['gain']
    offset = full_output['offset']
  
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
