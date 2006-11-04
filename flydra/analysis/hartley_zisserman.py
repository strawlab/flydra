#!/usr/bin/env python

import numarray as nx
import numarray.linear_algebra
svd = numarray.linear_algebra.singular_value_decomposition

# define plane by 3 points: (p. 66)
X1 = [ 2.0,   0.0, 0.0, 1.0 ]
X2 = [ 10.0,  0.0, 0.0, 1.0 ]
X3 = [  0.0, 20.0, 0.0, 1.0 ]

A = nx.array( [ X1, X2, X3] )
u,d,vt=svd(A,full_matrices=True)
Pt = vt[3,:]

print nx.dot( Pt, X1 ) # =0, eqn (3.2)
print nx.dot( A, nx.transpose(Pt) ) # = [0, 0, 0], eqn (3.3)

# define point by 3 planes: (p. 67)
X4 = [ 12, 43, 12, 1]
X5 = [ 13, 43, 13, 1]

X6 = [ 120, 431, -1, 1]
X7 = [ 12, 432, -1, 1]

A2 = nx.array( [ X1, X4, X5] )
u,d,vt2=svd(A2,full_matrices=True)
Pt2 = vt2[3,:]
A3 = nx.array( [ X1, X6, X7] )
u,d,vt3=svd(A3,full_matrices=True)
Pt3 = vt3[3,:]

PA = nx.array( [Pt, Pt2, Pt3] )

u,d,vtP=svd(PA,full_matrices=True)
Xi = vtP[3,:]
Xi = Xi/Xi[3] # normalize
print Xi,'?=',X1

print nx.dot( PA, Xi ) # eqn (3.5)

print

# ----------------------------------
#
# intersect >2 planes: (pg. 323)
#
# ----------------------------------

# 2 points defining line:
X1 = [ 100, 1, 23, 1]
X2 = [ 10,0, 0, 1]

# 4 random points to make 4 planes:
rX1 = [10, 20,30, 1]
rX2 = [2, 3, 4, 1]
rX3 = [1, 3, 1, 1]
rX4 = [20, 30, 1, 1]

# 4 planes, each defined by 3 points:
A = nx.array( [ X1, X2, rX1] )
u,d,vt=svd(A,full_matrices=True)
P1 = vt[3,:]

A = nx.array( [ X1, X2, rX2] )
u,d,vt=svd(A,full_matrices=True)
P2 = vt[3,:]

A = nx.array( [ X1, X2, rX3] )
u,d,vt=svd(A,full_matrices=True)
P3 = vt[3,:]

A = nx.array( [ X1, X2, rX4] )
u,d,vt=svd(A,full_matrices=True)
P4 = vt[3,:]

# Make extra plane for testing recovery of point X1
A = nx.array( [ X1, rX1, rX4] )
u,d,vt=svd(A,full_matrices=True)
Ptest = vt[3,:]

# compose A using all planes with line defined by points X1 and X2
A = nx.array([P1,P2,P3,P4])

# do SVD
u,d,vt=svd(A,full_matrices=True)
V = nx.transpose(vt)

# "two columns of V corresponding to the two largest singular values
# span the best rank 2 approximation to A and may be used to define
# the line of intersection of the planes"

Vcol0 = V[:,0]
Vcol1 = V[:,1]

# Check if we can recover X1.

# Vcol1 and Vcol1 should intersect to form a line defined by points X1
# and X2.  So, if we use another plane that includes X1 (but not X2),
# we should recover X1 when intersecting all three planes.

PA = nx.array( [Vcol0, Vcol1, Ptest] )

u,d,vt=svd(PA,full_matrices=True)
Xi = vt[3,:]
Xi = Xi/Xi[3] # normalize
print Xi,'?=',X1
