#!/usr/bin/env python
from numarray import *
import fit_params

X = zeros((20,10),type=UInt8)
X[6:8,3:6]=200
X[8,6]=255

print X

x0,y0,orientation=fit_params.fit_params(X)
print 'x0,y0',x0,y0
print 'orientation',orientation
