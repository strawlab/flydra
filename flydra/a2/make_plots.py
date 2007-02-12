import pylab
import detect_saccades
import numpy

if 1:
    ca = detect_saccades.CachingAnalyzer()
    obj_id = 1082
    filename='DATA20061208_181556.kalmanized.h5'
    
    RAD2DEG = 180.0/numpy.pi
    results = ca.calculate_trajectory_metrics(obj_id,
                                              filename,
                                              frames_per_second=100.0,
                                              method='position based',
                                              method_params={'downsample':1})
    horiz_saccades = ca.detect_saccades(obj_id,filename,
                                        frames_per_second=100.0,
                                        method='position based',
                                        method_params={'downsample':1,
                                                       'horizontal only':True})
    
##    saccades = ca.detect_saccades(obj_id,filename,
##                                  frames_per_second=100.0,
##                                  method='position based',
##                                  method_params={'downsample':1,
##                                                 'horizontal only':False})
    
    if 1:
        # like tammero 2002 basic plot
        pylab.figure(figsize=(8,4))
        ax=pylab.subplot(4,1,1)
        pylab.plot(results['time_t'],results['heading_t']*RAD2DEG,'.-')
        for t in horiz_saccades['times']:
            pylab.axvline(t,color='k')
            
        ax = pylab.subplot(4,1,2,sharex=ax)
        pylab.plot(results['time_dt'],results['h_ang_vel_dt']*RAD2DEG,'k-',lw=2,label='heading')
        pylab.plot(results['time_dt'],results['ang_vel_dt']*RAD2DEG,'b-',lw=2, label='overall')
        pylab.legend()
        pylab.ylabel('angular velocity\n(deg/s)')
        ax.set_ylim([-2500,2500])
        for t in horiz_saccades['times']:
            pylab.axvline(t,color='k')
##        for t in saccades['times']:
##            pylab.axvline(t,color='b')

        ax = pylab.subplot(4,1,3,sharex=ax)
        pylab.plot(results['time_raw'], results['X_raw'][:,0],'.-') # x position
        for t in horiz_saccades['times']:
            pylab.axvline(t,color='k')
            
        ax = pylab.subplot(4,1,4,sharex=ax)
        pylab.plot(results['time_raw'], results['X_raw'][:,1],'.-') # y position
        for t in horiz_saccades['times']:
            pylab.axvline(t,color='k')
            
##        pylab.plot(times2,h_speeds2*100,'c-',lw=2,label='horizontal')
##        pylab.plot(times2,v_speeds2*100,'r-',lw=2,label='vertical')
##        pylab.ylabel('speed\n(cm/s)')
##        pylab.xlabel('time (s)')
##        pylab.legend(loc='upper left')
##        ax.set_yticks([-20,20,60])
##        ax.set_xlim((0,times.max()))
        pylab.figtext(0,0,'id %d, %s'%(obj_id,filename))
    pylab.show()
