#!/usr/bin/env python
##import Numeric as na
##from Numeric import *
##from LinearAlgebra import *
import numarray as na
from numarray import *
from numarray.linear_algebra import *
import sys

# Extract (linear) camera parameters.

def cross(a,b):
    cross = []
    cross.append( a[1]*b[2]-a[2]*b[1] )
    cross.append( a[2]*b[0]-a[0]*b[2] )
    cross.append( a[0]*b[1]-a[1]*b[0] )
    return asarray(cross)

def norm(a):
    return sqrt(sum(a**2))

def rq(X):
    Qt, Rt = qr_decomposition(transpose(X))
    Rt = transpose(Rt)
    Qt = transpose(Qt)

    Qu = []

    Qu.append( cross(Rt[1,:], Rt[2,:] ) )
    Qu[0] = Qu[0]/norm(Qu[0])

    Qu.append( cross(Qu[0], Rt[2,:] ) )
    Qu[1] = Qu[1]/norm(Qu[1])

    Qu.append( cross(Qu[0], Qu[1] ) )

    R = matrixmultiply( Rt, transpose(Qu))
    Q = matrixmultiply( Qu, Qt )

    return R, Q

####################################################################

# sample data from "Multiple View Geometry in Computer Vision" Hartley
# and Zisserman, example 6.2, p. 163

if len( sys.argv ) < 2:
    P = array( [[ 3.53553e2,   3.39645e2,  2.77744e2,  -1.44946e6 ],
                [-1.03528e2,   2.33212e1,  4.59607e2,  -6.32525e5 ],
                [ 7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e2 ]] )
else:
    def load_ascii_matrix(filename):
        fd=open(filename,mode='rb')
        buf = fd.read()
        lines = buf.split('\n')[:-1]
        return na.array([map(float,line.split()) for line in lines])
    P=load_ascii_matrix( sys.argv[1] )

orig_determinant = determinant
def determinant( A ):
    return orig_determinant( asarray( A ) )

# camera center
X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

C_ = transpose(array( [[ X/T, Y/T, Z/T ]] ))

M = P[:,:3]

# do the work:
K,R = rq(M)
Knorm = K/K[2,2]

t = matrixmultiply( -R, C_ )

# reconstruct P via eqn 6.8 (p. 156)
P_ = matrixmultiply( K, concatenate( (R, t), axis=1 ) )

show_results = True
if show_results:
    print 'P (original):'
    print P
    print

    print 'C~ (center):'
    print C_
    print

    print 'K (calibration):'
    print K
    print

    print 'normalized K (calibration):'
    print Knorm
    print

    print 'R (orientation):'
    print R
    print

    print 't (translation in world coordinates):'
    print t
    print

    print 'P (reconstructed):'
    print P_
    print
