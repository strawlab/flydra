import os
opj=os.path.join
import numarray as nx
import sys
import numarray.linear_algebra
svd = numarray.linear_algebra.singular_value_decomposition
Lstar_i = nx.array([2,3,1,0,0,0]) # Lstar to Pluecker line coords (i index)
Lstar_j = nx.array([3,1,2,3,2,1]) # Lstar to Pluecker line coords (j index)
L_i = nx.array([0,0,0,1,3,2])
L_j = nx.array([1,2,3,2,1,3])
    
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

def Lcoords2Lmatrix(Lcoords):
    Lcoords = nx.asarray(Lcoords)
    Lmatrix = nx.zeros((4,4),nx.Float64)
    Lmatrix[L_i,L_j]=Lcoords
    Lmatrix[L_j,L_i]=-Lcoords
    return Lmatrix
    
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

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def find3d(self,
               cam_ids_and_points2d):
        
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        
        M=len(cam_ids_and_points2d) # number of views of single point

        # Fill matrices
        A=nx.zeros((2*M,4),nx.Float64) # for best point
        P=nx.zeros((M,4),nx.Float64) # for best line
        for m, (cam_id,value_tuple) in enumerate(cam_ids_and_points2d):
            x,y,area,slope,eccentricity, p1,p2,p3,p4 = value_tuple
            Pmat = self.Pmat[cam_id]
            row3 = Pmat[2,:]
            A[2*m  ,:]=x*row3 - Pmat[0,:]
            A[2*m+1,:]=y*row3 - Pmat[1,:]

            P[m,:] = p1,p2,p3,p4
            
        # Calculate best point
        u,d,vt=svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize

        # Calculate best line
        u,d,vt=svd(P,full_matrices=True)
        # "two columns of V corresponding to the two largest singular
        # values span the best rank 2 approximation to A and may be
        # used to define the line of intersection of the planes"
        # (Hartley & Zisserman, p. 323)
        P = vt[0,:] # P,Q are planes (take row because this is transpose(V))
        Q = vt[1,:]

        # dual Pluecker representation: (Hartley & Zisserman, p. 71)
        Lstar = nx.outerproduct(P,Q) - nx.outerproduct(Q,P)
        # convert to Pluecker line coordinates
        Lcoords = Lstar[Lstar_i,Lstar_j]
        return X, Lcoords    

    def find2d(self,cam_id,X,Lcoords):
        # see Hartley & Zisserman (2003) p. 449
        if len(X) == 3:
            X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
        Pmat = self.Pmat[cam_id]
        x=nx.dot(Pmat,X)

        # see Hartley & Zisserman (2003) p. 198, eqn 8.2
        L = Lcoords2Lmatrix(Lcoords)
        # XXX could be made faster by pre-computing line projection matrix
        lx = nx.dot(Pmat, nx.dot(L,nx.transpose(Pmat)))
        l3 = lx[2,1], lx[0,2], lx[1,0] #(p. 581)
        
        return x[0:2,0]/x[2,0], l3
