import os
opj=os.path.join
import numarray as nx
from numarray.linear_algebra import singular_value_decomposition

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    fd.close()
    lines = buf.split('\n')[:-1]
    return nx.array([map(float,line.split()) for line in lines])

def save_ascii_matrix(M,filename):
    def fmt(f):
        return '% 8e'%f
    A = nx.asarray(M)
    if len(A.shape) == 1:
        A=nx.reshape(A, (1,A.shape[0]) )
    fd = open(filename,mode='wb')
    for i in range(A.shape[0]):
        buf = ' '.join( map( fmt, A[i,:] ) )
        fd.write( buf )
        fd.write( '\n' )
    fd.close()

class Reconstructor:
    def __init__(self,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 ):
        fd = open(os.path.join(calibration_dir,'camera_order.txt'),'r')
        cam_ids = fd.read().split('\n')
        if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        N = len(cam_ids)
        print 'cam_ids',cam_ids
        # load calibration matrices
        self.Pmat = {}
        for i, cam_id in enumerate(cam_ids):
            fname = Pmat_name%(i+1)
            print "Loading %s from %s"%(cam_id,fname)
            self.Pmat[cam_id] = load_ascii_matrix(opj(calibration_dir,fname))

    def find3d(self,
               cam_ids_and_points2d):
        # see Hartley & Zisserman (2003) p. 593 (see also p. 587)
        M=len(cam_ids_and_points2d) # number of views of single point
        A=nx.zeros((2*M,4),nx.Float64)
        
        for m, (cam_id,(xm,ym)) in enumerate(cam_ids_and_points2d):
            Pmat = self.Pmat[cam_id]
            row3 = Pmat[2,:]
            A[2*m  ,:]=xm*row3 - Pmat[0,:]
            A[2*m+1,:]=ym*row3 - Pmat[1,:]

        u,s,v=singular_value_decomposition(A)
        X = v[-1,0:3]/v[-1,3] # normalize
        return X

    def find2d(self,cam_id,X):
        if len(X) == 3:
            X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
        Pmat = self.Pmat[cam_id]
        x=nx.matrixmultiply(Pmat,X)
        return x[0:2,0]/x[2,0]
