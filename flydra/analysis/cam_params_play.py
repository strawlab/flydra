#!/usr/bin/env python
import numpy
import sys

# Extract (linear) camera parameters.

def cross(a,b):
    cross = []
    cross.append( a[1]*b[2]-a[2]*b[1] )
    cross.append( a[2]*b[0]-a[0]*b[2] )
    cross.append( a[0]*b[1]-a[1]*b[0] )
    return numpy.asarray(cross)

def norm(a):
    return numpy.sqrt(numpy.sum(a**2))

def rq(X):
    Qt, Rt = numpy.linalg.qr(numpy.transpose(X))
    Rt = numpy.transpose(Rt)
    Qt = numpy.transpose(Qt)

    Qu = []

    Qu.append( cross(Rt[1,:], Rt[2,:] ) )
    Qu[0] = Qu[0]/norm(Qu[0])

    Qu.append( cross(Qu[0], Rt[2,:] ) )
    Qu[1] = Qu[1]/norm(Qu[1])

    Qu.append( cross(Qu[0], Qu[1] ) )

    R = numpy.dot( Rt, numpy.transpose(Qu))
    Q = numpy.dot( Qu, Qt )

    return R, Q

####################################################################

# sample data from "Multiple View Geometry in Computer Vision" Hartley
# and Zisserman, example 6.2, p. 163

if len( sys.argv ) < 2:
    P = numpy.array( [[ 3.53553e2,   3.39645e2,  2.77744e2,  -1.44946e6 ],
                      [-1.03528e2,   2.33212e1,  4.59607e2,  -6.32525e5 ],
                      [ 7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e2 ]] )
else:
    def load_ascii_matrix(filename):
        fd=open(filename,mode='rb')
        buf = fd.read()
        lines = buf.split('\n')[:-1]
        return numpy.array([map(float,line.split()) for line in lines])
    P=load_ascii_matrix( sys.argv[1] )

orig_determinant = numpy.linalg.det
def determinant( A ):
    return orig_determinant( numpy.asarray( A ) )

# camera center
X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

C_ = numpy.transpose(numpy.array( [[ X/T, Y/T, Z/T ]] ))

M = P[:,:3]

# do the work:
K,R = rq(M)
Knorm = K/K[2,2]

t = numpy.dot( -R, C_ )

# reconstruct P via eqn 6.8 (p. 156)
P_ = numpy.dot( K, numpy.concatenate( (R, t), axis=1 ) )

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
