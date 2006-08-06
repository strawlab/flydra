# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw

import numpy
#import numpy.matlib
import pylab
numpy.set_printoptions(linewidth = 150)

# intial parameters
ss = 9 # length of state vector (state size)
os = 3 # length of observation vector (observation size)

if 1:
    from dataset2 import z,time_msec,frames

if 0:
    zm = z[1:,:]
    dzdt = (z[1:,:]-z[:-1,:]) / (time_msec[1:]-time_msec[:-1])[:,numpy.newaxis]

    stateobs = numpy.hstack((zm,dzdt))
    stateobs_mean = numpy.mean(stateobs,axis=0)
    stateobs_var = numpy.std(stateobs,axis=0)
    #stateobs_covar = numpy.dot(stateobs_var[:,numpy.newaxis],stateobs_var[numpy.newaxis,:])
    stateobs_covar = numpy.cov(stateobs,rowvar=0)
    if 0:
        print stateobs
        print stateobs_mean
        print stateobs_var
        print stateobs_covar
        sys.exit(1)

dt = 10.0 # msec

# process update matrix (time evolution update matrix)
F = numpy.eye(ss)
### velocity affects position:
F[0,3]=dt
F[1,4]=dt
F[2,5]=dt

# acceleration on position
F[0,6]=dt**2
F[1,7]=dt**2
F[2,8]=dt**2

# acceleration on velocity
F[3,6]=dt
F[4,7]=dt
F[5,8]=dt

# acceleration decays
F[6,6]=.1
F[7,7]=.1
F[8,8]=.1
print 'F',F

# measurement prediction matrix
H = numpy.zeros((os,ss))
H[:os,:os] = numpy.eye(os)
print 'H',H

# parameters

# process covariance
Q = numpy.zeros((ss,ss))
for i in range(6,9):
    Q[i,i] = 1e-5
print 'Q',Q

# measurement noise covariance matrix
R = numpy.eye(os)
print 'R',R


# initial state error covariance guess
P_k1=numpy.eye(ss)
print 'P_k1',P_k1

R_real = R
#R_nodata = R*1e10#+1e10*numpy.ones_like(R)
R_nodata = R

xhats = [] # list for saving xhat
Ps = []

xhat_k1 = None
frame_range = range(frames[0],frames[-1]+1)
#frame_range = range(frames[0]+20,frames[-1]+1)
for last_k,frame in enumerate(frame_range):
    test_cond = frames==frame
    test_cond_nz = numpy.nonzero(test_cond)
    
    if len(test_cond_nz):
        # we have observation
        k = test_cond_nz[0]
        this_z = z[k,:]
        R = R_real
    else:
        # no observation
        this_z = None
        R = R_nodata
        print 'no data for frame',frame

    if xhat_k1 is None:
        # set first expected state vector to observation and zero velocity, zero accel
        xhat_k1 = numpy.hstack( (this_z,(0,0,0, 0,0,0)))
        
        
#for k in range(len(z)):
##    if k==10:
##        import sys; sys.exit()
    # time update
    xhatminus = numpy.dot(F,xhat_k1)
    Pminus = numpy.dot(numpy.dot(F,P_k1),numpy.transpose(F))+Q

    print '-'*20,last_k,frame

    print 'xhatminus',xhatminus
    print 'Pminus',Pminus
#    print 'H.T.shape',H.T.shape
    
    # measurement update
    #K = Pminus/( Pminus+R ) # gain on residuals
#    print 'H.T',H.T

    Knum = numpy.dot(Pminus,numpy.transpose(H))
    #Knum = numpy.dot(Pminus,H.T)
    Kdenom=  numpy.dot(numpy.dot(H,Pminus),numpy.transpose(H))+R
    print 'Knum',Knum
    print 'Kdenom',Kdenom

    print 'Knum.shape',Knum.shape
    print 'Kdenom.shape',Kdenom.shape
    #K = Knum/Kdenom
    #K = Knum[:m,:m]/Kdenom
    invKdenom = numpy.linalg.inv(Kdenom)

    K = numpy.dot(Knum,invKdenom)
    print 'K',K
    if this_z is not None:
        # update based on observation
        xhat = xhatminus+numpy.dot(K, this_z-numpy.dot(H,xhatminus))
    else:
        # no observation, a priori estimate becomes posteri estimate
        xhat = xhatminus

    print 'xhat',xhat
    one_minus_KH = numpy.eye(ss)-numpy.dot(K,H)
    P = numpy.dot(one_minus_KH,Pminus)
    print 'P',P

    # Save values for plotting
    xhats.append(xhat)
    Ps.append(P)

    # Update for next step.
    # (This step's a posteri estimates become next steps a priori.)
    xhat_k1 = xhat
    P_k1 = P

xhats = numpy.asarray(xhats)
Ps = numpy.asarray(Ps)

print 'last_k',last_k
print 'xhats.shape',xhats.shape

pylab.figure()
ax = None
varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
for i in range(ss):
    ax = pylab.subplot(ss,1,i+1,sharex=ax)
    var = varnames[i]
    if i<os:
        ax.plot(frames,z[:,i],'k+',label='noisy measurements of %s'%var)
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
