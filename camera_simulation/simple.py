#!/usr/bin/env python
from numarray import *
import fit_params

X = zeros((8,8),type=UInt8)
X[2:7,3:6]=200
X[1,5]=193
X[7,3] = 255

print X

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation
