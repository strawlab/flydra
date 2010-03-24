#!/usr/bin/env python

import os
opj=os.path.join
import numpy as na
import numpy.linalg as la

data_dir = '/home/astraw/mcsc_data'

IdMat_name = 'IdMat.dat'
points_name = 'points.dat'
Pmat_name = 'camera%d.Pmat.cal'
points4cal_name='cam%d.points4cal.dat'

NTUPLE=2

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    lines = buf.split('\n')[:-1]
    return na.array([map(float,line.split()) for line in lines])

def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def uP2X(Umat,Ps):
    """

    arguments
    ---------
    Umat: 3*N x n matrix of n homogenous points
    Ps:   3 x 4*N matrix of projection matrices

    returns
    -------
    X:    4 x n matrix of homogenous 3D points
    """
    N=Umat.shape[0]/3
    n=Umat.shape[1]
    
    # reshuffle the Ps matrix
    Pmat = []
    for i in range(N):
        Pmat.append( Ps[0,i*4:i*4+4] )
        Pmat.append( Ps[1,i*4:i*4+4] )
        Pmat.append( Ps[2,i*4:i*4+4] )
    Pmat = na.array(Pmat)

    X = []
    for i in range(n): # for all points
        A = []
        for j in range(N): # for all cameras
            # create the data matrix
            A.append( ( Umat[j*3  ,i]*Pmat[j*3+2,:] - Pmat[j*3  ,:]) )
            A.append( ( Umat[j*3+1,i]*Pmat[j*3+2,:] - Pmat[j*3+1,:]) )
        A=na.asarray(A)
        u,s,v=la.singular_value_decomposition(A)
        X.append( v[-1,:] )
    
    #normalize reconstructed points
    X=na.array(X)
    X.transpose()
    w=X[3,:]
    X=X/na.resize(w,(4,X.shape[1]))
    return X

def tripleIdx(a):
    return ( na.repeat(na.asarray(a)*3,[3]*len(a)) +
             na.resize( [0,1,2],(3*len(a),)) )

if __name__=='__main__':
    IdMat=load_ascii_matrix(opj(data_dir,IdMat_name))
    CAMS=IdMat.shape[0]
    points=load_ascii_matrix(opj(data_dir,points_name))
    Ws=points

    Pmat=[load_ascii_matrix(opj(data_dir,Pmat_name%(i+1))) for i in range(CAMS)]
    npts=0
    fd = open('X.recalc.dat',mode='wb')
    for j in range(IdMat.shape[1]):
        cam_idx=na.nonzero(IdMat[:,j])[0]
        if len(cam_idx)<NTUPLE:
            fd.write('NaN NaN NaN\n')
            continue
        npts=npts+1
        Wsx = Ws[tripleIdx(cam_idx),j]
        Wsx = Wsx[:,na.newaxis]
        Pmatx = na.concatenate([Pmat[i] for i in cam_idx],axis=1)
        X=uP2X(Wsx,Pmatx)
        if 0:
            save_ascii_matrix('Wsx%d.dat'%npts,Wsx)
            save_ascii_matrix('Pmatx%d.dat'%npts,Pmatx)
            save_ascii_matrix('X%d.dat'%npts,X)
        fd.write('%f %f %f\n'%(X[0,0],X[1,0],X[2,0]))
    print npts,'points drawn'
