# $Id$
import os, glob, sys, math
opj=os.path.join
import numarray as nx
import numarray.linear_algebra
svd = numarray.linear_algebra.singular_value_decomposition
import reconstruct_utils

L_i = nx.array([0,0,0,1,3,2])
L_j = nx.array([1,2,3,2,1,3])

MINIMUM_ECCENTRICITY = 2.0 # threshold to fit line
    
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


def setOfSubsets(L):
    """find all subsets of L

    from Alex Martelli:
    http://mail.python.org/pipermail/python-list/2001-January/027238.html
    """
    N = len(L)
    return [ [ L[i] for i in range(N)
                if X & (1L<<i) ]
        for X in range(2**N) ]

class Reconstructor:
    def __init__(self,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 debug = False,
                 ):
        self._debug = debug

        if type(calibration_dir) in [str,unicode]:
            calibration_data_source = 'normal files'
        else:
            calibration_data_source = 'pytables'

        if calibration_data_source == 'normal files':
            fd = open(os.path.join(calibration_dir,'camera_order.txt'),'r')
            cam_ids = fd.read().split('\n')
            fd.close()
            if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        elif calibration_data_source == 'pytables':
            import tables as PT # PyTables
            assert type(calibration_dir)==PT.File
            results = calibration_dir
            nodes = results.root.calibration.pmat._f_listNodes()
            cam_ids = []
            for node in nodes:
                cam_ids.append( node.name )
            
        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Res = {}
        self.pmat_inv = {}
        self._helper = {}
        
        if calibration_data_source == 'normal files':
            res_fd = open(os.path.join(calibration_dir,'Res.dat'),'r')
            for i, cam_id in enumerate(cam_ids):
                fname = Pmat_name%(i+1)
                pmat = load_ascii_matrix(opj(calibration_dir,fname)) # 3 rows x 4 columns
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = map(int,res_fd.readline().split())
                self.pmat_inv[cam_id] = numarray.linear_algebra.generalized_inverse(pmat)
            res_fd.close()

            # load non linear parameters
            rad_files = glob.glob(os.path.join(calibration_dir,'*.rad'))
            assert len(rad_files) < 10
            rad_files.sort() # cheap trick to associate camera number with file
            for cam_id, filename in zip( cam_ids, rad_files ):
                params = {}
                execfile(filename,params)
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    params['K11'], params['K22'], params['K13'], params['K23'],
                    params['kc1'], params['kc2'], params['kc3'], params['kc4'])
                
        elif calibration_data_source == 'pytables':
            for cam_id in cam_ids:
                pmat = nx.array(results.root.calibration.pmat.__getattr__(cam_id))
                res = tuple(results.root.calibration.resolution.__getattr__(cam_id))
                K = nx.array(results.root.calibration.intrinsic_linear.__getattr__(cam_id))
                nlparams = tuple(results.root.calibration.intrinsic_nonlinear.__getattr__(cam_id))
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = res
                self.pmat_inv[cam_id] = numarray.linear_algebra.generalized_inverse(pmat)
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    K[0,0], K[1,1], K[0,2], K[1,2],
                    nlparams[0], nlparams[1], nlparams[2], nlparams[3])
            
        self.cam_combinations = [s for s in setOfSubsets(cam_ids) if len(s) >=2]
        def cmpfunc(a,b):
            if len(a) > len(b):
                return -1
            else:
                return 0
        # order camera combinations from most cameras to least
        self.cam_combinations.sort(cmpfunc)

    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def get_intrinsic_linear(self, cam_id):
        self._helper[cam_id].get_K()
        
    def get_intrinsic_nonlinear(self, cam_id):
        self._helper[cam_id].get_nlparams()

    def undistort(self, cam_id, x_kk):
        return self._helper[cam_id].undistort(x_kk[0],x_kk[1])

    def distort(self, cam_id, xl):
        return self._helper[cam_id].distort(xl[0],xl[1])

    def find3d(self, cam_ids_and_points2d ):
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        
        # Construct matrices
        A=[]
        P=[]
        return_line_coords = True
        for m, (cam_id,value_tuple) in enumerate(cam_ids_and_points2d):
            if len(value_tuple)==2:
                # only point information ( no line )
                x,y = value_tuple
                return_line_coords = False
            else:
                # get shape information from each view of a blob:
                x,y,area,slope,eccentricity, p1,p2,p3,p4 = value_tuple
            Pmat = self.Pmat[cam_id] # Pmat is 3 rows x 4 columns
            row3 = Pmat[2,:]
            A.append( x*row3 - Pmat[0,:] )
            A.append( y*row3 - Pmat[1,:] )

            if return_line_coords:
                if eccentricity > MINIMUM_ECCENTRICITY: # require a bit of elongation
                    P.append( (p1,p2,p3,p4) )
        
        # Calculate best point
        A=nx.array(A)
        u,d,vt=svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize

        if not return_line_coords or len(P) < 2:
            Lcoords = None
        else:
            P = nx.asarray(P)
            # Calculate best line
            try:
                u,d,vt=svd(P,full_matrices=True)
                # "two columns of V corresponding to the two largest singular
                # values span the best rank 2 approximation to A and may be
                # used to define the line of intersection of the planes"
                # (Hartley & Zisserman, p. 323)
                P = vt[0,:] # P,Q are planes (take row because this is transpose(V))
                Q = vt[1,:]

                # directly to Pluecker line coordinates
                Lcoords = ( -(P[3]*Q[2]) + P[2]*Q[3],
                              P[3]*Q[1]  - P[1]*Q[3],
                            -(P[2]*Q[1]) + P[1]*Q[2],
                            -(P[3]*Q[0]) + P[0]*Q[3],
                            -(P[2]*Q[0]) + P[0]*Q[2],
                            -(P[1]*Q[0]) + P[0]*Q[1] )
            except:
                #print 'WARNING: %s %s'%(x.__class__, str(x))
                print 'WARNING: unknown error in reconstruct.py'
                Lcoords = None
        if return_line_coords:
            return X, Lcoords
        else:
            return X

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
