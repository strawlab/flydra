import numpy

import params
import kalman

A=params.A
C=params.C
Q=params.Q
R=params.R

ss = A.shape[0]
os = C.shape[0]

# initial state error covariance guess
P_k1=numpy.eye(ss)

from result_utils import get_results, get_f_xyz_L_err

try:
    results
except NameError:
    results = get_results('DATA20060719_180955.h5')

# get "original data" from flydra's hypothesis testing algorithm
try:
    frames_orig3d,y_mm,L,err
except NameError:
    max_err=10.0
    typ='best'
    frames_orig3d,y_mm,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)
    del max_err
    del typ

y = y_mm/1000.0 # put in meters (from millimeters, mm)

frame_range = range(858535,859340)

xhat_k1 = numpy.hstack((y[0,:],(0,0,0, 0,0,0)))

kalman_state = kalman.KalmanFilter(A,C,Q,R,xhat_k1,P_k1)

xhats = []
Ps = []
y_show = []
for last_k,frame in enumerate(frame_range):
    
    
    test_cond = frames_orig3d==frame
    test_cond_nz = numpy.nonzero(test_cond)
    
    if len(test_cond_nz):
        # we have observation
        k = test_cond_nz[0]
        this_y = y[k,:]
    else:
        # no observation
        this_y = None
        print 'no data for frame',frame

    
    xhat,P = kalman_state.step(this_y,return_error_estimate=True)
    xhats.append(xhat)
    Ps.append(P)
    
    if this_y is None:
        this_y = numpy.nan*numpy.ones((os,))
    y_show.append(this_y)


#############################
    
xhats = numpy.asarray(xhats)
Ps = numpy.asarray(Ps)
y_show = numpy.asarray(y_show)

import pylab

pylab.figure()
ax = None
varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
for i in range(ss):
    ax = pylab.subplot(ss,1,i+1,sharex=ax)
    var = varnames[i]
    if i<os:
        ax.plot(frame_range,y_show[:,i],'k+',label='noisy measurements of %s'%var)
    ax.plot(frame_range,xhats[:,i],'b-',label='a posteri estimate of %s'%var)
    pylab.ylabel(var)
    
pylab.figure()
ax = None
for i in range(ss):
    ax = pylab.subplot(ss,1,i+1,sharex=ax)
    var = varnames[i]
    ax.plot(frame_range,Ps[:,i,i],'b-',label='a posteri estimate of variance of %s'%var)


    ax.legend()
pylab.xlabel('frame')

pylab.show()
