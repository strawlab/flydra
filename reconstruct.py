# $Id$
import os
opj=os.path.join
import numarray as nx
import sys
import numarray.linear_algebra
import math
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

def as_column(x):
    x = nx.asarray(x)
    if len(x.shape) == 1:
        x = nx.reshape(x, (x.shape[0],1) )
    return x

def as_vec(x):
    x = nx.asarray(x)
    if len(x.shape) == 1:
        return x
    elif len(x.shape) == 2:
        long_dim = x.shape[0]+x.shape[1]-1
        if (x.shape[0]*x.shape[1]) != long_dim:
            # more than 1 rows or columns
            raise ValueError("cannot convert to vector")
    else:
        raise ValueError("cannot convert to vector")
    return nx.reshape(x,(longdim,))

def Lcoords2Lmatrix(Lcoords):
    Lcoords = nx.asarray(Lcoords)
    Lmatrix = nx.zeros((4,4),nx.Float64)
    Lmatrix[L_i,L_j]=Lcoords
    Lmatrix[L_j,L_i]=-Lcoords
    return Lmatrix

def Lmatrix2Lcoords(Lmatrix):
    return Lmatrix[L_i,L_j]

def pts2Lmatrix(A,B):
    A = as_column(A)
    B = as_column(B)
    L = nx.dot(A,nx.transpose(B)) - nx.dot(B,nx.transpose(A))
    return L

def pts2Lcoords(A,B):
    return Lmatrix2Lcoords(pts2Lmatrix(A,B))

def line_direction(Lcoords):
    #U = nx.array((-Lcoords[2], -Lcoords[4], Lcoords[5]))
    U = nx.array((-Lcoords[2], Lcoords[4], -Lcoords[5]))
    U = U/math.sqrt(U[0]**2 + U[1]**2 + U[2]**2) # normalize
    return U

def pmat2cam_center(P):
    assert P.shape == (3,4)
    determinant = numarray.linear_algebra.determinant
    
    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    C_ = nx.transpose(nx.array( [[ X/T, Y/T, Z/T ]] ))
    return C_
    
class Reconstructor:
    def __init__(self,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 debug = False,
                 ):
        self._debug = debug
        fd = open(os.path.join(calibration_dir,'camera_order.txt'),'r')
        res_fd = open(os.path.join(calibration_dir,'Res.dat'),'r')
        cam_ids = fd.read().split('\n')
        fd.close()
        if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Res = {}
        self.pmat_inv = {}
        
        for i, cam_id in enumerate(cam_ids):
            fname = Pmat_name%(i+1)
            pmat = load_ascii_matrix(opj(calibration_dir,fname))
            self.Pmat[cam_id] = pmat
            self.Res[cam_id] = map(int,res_fd.readline().split())
            self.pmat_inv[cam_id] = numarray.linear_algebra.generalized_inverse(pmat)
            
        res_fd.close()
        self.cam_order = cam_ids

    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def find3d(self,
               cam_ids_and_points2d):
        
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        
        M=len(cam_ids_and_points2d) # number of views of single point

        # Fill matrices
        A=nx.zeros((2*M,4),nx.Float64) # for best point
        P=[]
        for m, (cam_id,value_tuple) in enumerate(cam_ids_and_points2d):
            x,y,area,slope,eccentricity, p1,p2,p3,p4 = value_tuple
            Pmat = self.Pmat[cam_id]
            row3 = Pmat[2,:]
            A[2*m  ,:]=x*row3 - Pmat[0,:]
            A[2*m+1,:]=y*row3 - Pmat[1,:]

            if eccentricity > 2.0: # require a bit of elongation
                P.append( (p1,p2,p3,p4) )
        
        # Calculate best point
        A=A.copy() # force to be contiguous (XXX hack -- find out why it gets non-contiguous)
        u,d,vt=svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize

        if len(P) < 2:
            Lcoords = None
        else:
            P = nx.asarray(P)
            # Calculate best line
            u,d,vt=svd(P,full_matrices=True)
            # "two columns of V corresponding to the two largest singular
            # values span the best rank 2 approximation to A and may be
            # used to define the line of intersection of the planes"
            # (Hartley & Zisserman, p. 323)
            P = vt[0,:] # P,Q are planes (take row because this is transpose(V))
            Q = vt[1,:]

    ##        # dual Pluecker representation: (Hartley & Zisserman, p. 71)
    ##        Lstar = nx.outerproduct(P,Q) - nx.outerproduct(Q,P)
    ##        # convert to Pluecker line coordinates
    ##        Lcoords = Lstar[Lstar_i,Lstar_j]

            # directly to Pluecker line coordinates
            Lcoords = ( -(P[3]*Q[2]) + P[2]*Q[3],
                          P[3]*Q[1]  - P[1]*Q[3],
                        -(P[2]*Q[1]) + P[1]*Q[2],
                        -(P[3]*Q[0]) + P[0]*Q[3],
                        -(P[2]*Q[0]) + P[0]*Q[2],
                        -(P[1]*Q[0]) + P[0]*Q[1] )
        return X, Lcoords    

    def find2d(self,cam_id,X,Lcoords=None):
        # see Hartley & Zisserman (2003) p. 449
        if len(X) == 3:
            X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
        Pmat = self.Pmat[cam_id]
        x=nx.dot(Pmat,X)

        x = x[0:2,0]/x[2,0] # normalize
        
        if Lcoords is not None:
            # see Hartley & Zisserman (2003) p. 198, eqn 8.2
            L = Lcoords2Lmatrix(Lcoords)
            # XXX could be made faster by pre-computing line projection matrix
            lx = nx.dot(Pmat, nx.dot(L,nx.transpose(Pmat)))
            l3 = lx[2,1], lx[0,2], lx[1,0] #(p. 581)
            return x, l3
        else:
            return x

    def find3d_single_cam(self,cam_id,x):
        return nx.dot(self.pmat_inv[cam_id], as_column(x))
