import numpy

import params
import kalman
import dataset2

A=params.A
C=params.C
Q=params.Q
R=params.R

ss = A.shape[0]
os = C.shape[0]

# initial state error covariance guess
P_k1=numpy.eye(ss)


y = dataset2.z
y = y/1000.0 # put in meters (from millimeters, mm)
frames = dataset2.frames
frame_range = range(frames[0],frames[-1]+1)

xhat_k1 = numpy.hstack((y[0,:],(0,0,0, 0,0,0)))

kalman_state = kalman.KalmanFilter(A,C,Q,R,xhat_k1,P_k1)

xhats = []
Ps = []
for last_k,frame in enumerate(frame_range):
    test_cond = frames==frame
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


#############################
    
xhats = numpy.asarray(xhats)
Ps = numpy.asarray(Ps)

import pylab

pylab.figure()
ax = None
varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
for i in range(ss):
    ax = pylab.subplot(ss,1,i+1,sharex=ax)
    var = varnames[i]
    if i<os:
        ax.plot(frames,y[:,i],'k+',label='noisy measurements of %s'%var)
    ax.plot(frame_range[:last_k+1],xhats[:,i],'b-',label='a posteri estimate of %s'%var)
    pylab.ylabel(var)
    
pylab.figure()
ax = None
for i in range(ss):
    ax = pylab.subplot(ss,1,i+1,sharex=ax)
    var = varnames[i]
    ax.plot(frame_range[:last_k+1],Ps[:,i,i],'b-',label='a posteri estimate of variance of %s'%var)


    ax.legend()
pylab.xlabel('frame')

pylab.show()
