#!/usr/bin/env python
from numarray import *
import fit_params

print "==========================================================="
print "==========================================================="
i=0

##### 1
X = zeros((8,8),type=UInt8)
X[4:7,2:7]=200
X[5,1]=193
X[5,7] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print '1 x0,y0',x0,y0
print 'orientation',orientation

##### 2
X = zeros((8,8),type=UInt8)
X[2:7,0:3]=200
X[1,1]=193
X[7,1] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print '2 x0,y0',x0,y0
print 'orientation',orientation

##### 3
X = zeros((8,8),type=UInt8)
X[2:7,0:3]=200
X[1,1]=193
X[7,0] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 4
X = zeros((8,8),type=UInt8)
X[1:4,6]=200
X[2:5,5]=200
X[3:6,4]=200
X[4:7,3]=200

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 5
X = zeros((8,8),type=UInt8)
X[1:4,6]=200
X[2:5,5]=200
X[3:6,4]=200
X[4:7,3]=200
X[7,3]=193
X[0,6] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 6
X = zeros((8,8),type=UInt8)
X[1:5,5]=200
X[3:7,4]=200
X[7,4]=193
X[0,5] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 7
X = zeros((8,8),type=UInt8)
X[0:2,6]=200
X[1:5,5]=200
X[3:7,4]=200
X[6:8,3]=200
X[7,4]=193
X[0,5] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 8
X = zeros((8,8),type=UInt8)
X[0:4,6]=200
X[1:5,5]=200
X[3:7,4]=200
X[4:8,3]=200
X[7,4]=193
X[0,5] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 9
X = zeros((8,8),type=UInt8)
X[0:4,6]=200
X[1:5,5]=200
X[2:6,4]=200
X[3:7,3]=200
X[1,7]=193
X[5,2] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 10
X = zeros((8,8),type=UInt8)
X[3:5,7]=200
X[1:5,6]=200
X[2:5,5]=200
X[3:6,4]=200
X[4:7,3]=200
X[4:6,2]=200
X[4,1]=193
X[5,7] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation

##### 11
X = zeros((8,8),type=UInt8)
X[2:4,7]=200
X[2:6,6]=200
X[2:5,5]=200
X[1:4,4]=200
X[0:3,3]=200
X[1:3,2]=200
X[2,1]=193
X[1,7] = 255

i=i+1
print X,i

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation
