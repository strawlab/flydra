import os
opj=os.path.join
import numarray as nx
import sys
from numarray.linear_algebra import singular_value_decomposition

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    fd.close()
    lines = buf.split('\n')[:-1]
    return nx.array([map(float,line.split()) for line in lines])

def save_ascii_matrix(M,fd):
    def fmt(f):
        return '% 8e'%f
    A = nx.asarray(M)
    if len(A.shape) == 1:
        A=nx.reshape(A, (1,A.shape[0]) )

    close_file = False
    if type(fd) == str:
        fd = open(fd,mode='wb')
        close_file = True
        
    for i in range(A.shape[0]):
        buf = ' '.join( map( fmt, A[i,:] ) )
        fd.write( buf )
        fd.write( '\n' )
    if close_file:
        fd.close()

class Reconstructor:
    def __init__(self,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 debug = False,
                 ):
        self._debug = debug
        fd = open(os.path.join(calibration_dir,'camera_order.txt'),'r')
        cam_ids = fd.read().split('\n')
        if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        for i, cam_id in enumerate(cam_ids):
            fname = Pmat_name%(i+1)
            self.Pmat[cam_id] = load_ascii_matrix(opj(calibration_dir,fname))
        self.cam_order = cam_ids

    def find3d(self,
               cam_ids_and_points2d):
        
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        
        M=len(cam_ids_and_points2d) # number of views of single point
        A=nx.zeros((2*M,4),nx.Float64)
        
        for m, (cam_id,(xm,ym,slope,eccentricity)) in enumerate(cam_ids_and_points2d):
            Pmat = self.Pmat[cam_id]
            row3 = Pmat[2,:]
            A[2*m  ,:]=xm*row3 - Pmat[0,:]
            A[2*m+1,:]=ym*row3 - Pmat[1,:]

        if self._debug:
            print 'A', A.shape
            save_ascii_matrix(A,sys.stdout)
            print
        
        u,s,v=singular_value_decomposition(A)

        if self._debug:
            print 'u',u.shape
            save_ascii_matrix(u,sys.stdout)
            print

            print 's',s.shape
            save_ascii_matrix(s,sys.stdout)
            print

            print 'v',v.shape
            save_ascii_matrix(v,sys.stdout)
            print
        
        X = v[-1,0:3]/v[-1,3] # normalize
        return X

    def find2d(self,cam_id,X):
        # see Hartley & Zisserman (2003) p. 449
        if len(X) == 3:
            X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
        Pmat = self.Pmat[cam_id]
        x=nx.matrixmultiply(Pmat,X)
        return x[0:2,0]/x[2,0]
