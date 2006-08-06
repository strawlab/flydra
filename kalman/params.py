import numpy
numpy.set_printoptions(linewidth = 150,precision=1)

# distance units are in m
# time units are in sec
# thus, velocity is (mm/msec) = (m/sec)

ss = 9 # length of state vector (state size)
os = 3 # length of observation vector (observation size)

dt = 0.01 # sec
dt2 = dt**2

# process update matrix (time evolution update matrix)
F = numpy.array([[   1. ,    0. ,    0. ,   dt  ,    0. ,    0. ,  dt2  ,    0. ,    0. ],
                 [   0. ,    1. ,    0. ,    0. ,   dt  ,    0. ,    0. ,  dt2  ,    0. ],
                 [   0. ,    0. ,    1. ,    0. ,    0. ,   dt  ,    0. ,    0. ,  dt2  ],
                 [   0. ,    0. ,    0. ,    1. ,    0. ,    0. ,   dt  ,    0. ,    0. ],
                 [   0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,   dt  ,    0. ],
                 [   0. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,   dt  ],
                 [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0.1,    0. ,    0. ],
                 [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0.1,    0. ],
                 [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0.1]])

# measurement prediction matrix
H = numpy.zeros((os,ss))
H[:os,:os] = numpy.eye(os) # directly measure x,y,z positions

# process covariance
Q = numpy.zeros((ss,ss))
for i in range(6,9):
    Q[i,i] = 1e-5 # acceleration noise

# measurement noise covariance matrix
R = numpy.eye(os) # 1mm

# initial state error covariance guess
P_k1=numpy.eye(ss)
print 'P_k1',repr(P_k1)
