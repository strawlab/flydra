# $Id$
import os, glob, sys, math
opj=os.path.join
import numarray as nx
import numarray.linear_algebra
svd = numarray.linear_algebra.singular_value_decomposition
from numarray.ieeespecial import getnan # XXX could I use something faster?
# Numeric (for speed)
import Numeric as fast_nx
import LinearAlgebra
fast_svd = LinearAlgebra.singular_value_decomposition
import reconstruct_utils # in pyrex/C for speed
import time

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

def norm_vec(V):
    Va = nx.asarray(V)
    if len(Va.shape)==1:
        # vector
        U = Va/math.sqrt(Va[0]**2 + Va[1]**2 + Va[2]**2) # normalize
    else:
        assert Va.shape[1] == 3
        Vamags = nx.sqrt(Va[:,0]**2 + Va[:,1]**2 + Va[:,2]**2)
        U = Va/Vamags[:,nx.NewAxis]
    return U
        
def line_direction(Lcoords):
    L = nx.asarray(Lcoords)
    if len(L.shape)==1:
        # single line coord
        U = nx.array((-L[2], L[4], -L[5]))
    else:
        assert L.shape[1] == 6
        # XXX could speed up with concatenate:
        U = nx.transpose(nx.array((-L[:,2], L[:,4], -L[:,5]))) 
    return norm_vec(U)

def pluecker_from_verts(A,B):
    def cross(vec1,vec2):
        return ( vec1[1]*vec2[2] - vec1[2]*vec2[1],
                 vec1[2]*vec2[0] - vec1[0]*vec2[2],
                 vec1[0]*vec2[1] - vec1[1]*vec2[0] )
    if len(A)==3:
        A = A[0], A[1], A[2], 1.0
    if len(B)==3:
        B = B[0], B[1], B[2], 1.0
    A=nx.reshape(A,(4,1))
    B=nx.reshape(B,(4,1))
    L = nx.matrixmultiply(A,nx.transpose(B)) - nx.matrixmultiply(B,nx.transpose(A))
    return Lmatrix2Lcoords(L)

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
                 cal_source = '/home/astraw/mcsc_data',
                 ):
        self.cal_source = cal_source

        if type(self.cal_source) in [str,unicode]:
            self.cal_source_type = 'normal files'
        else:
            self.cal_source_type = 'pytables'

        if self.cal_source_type == 'normal files':
            fd = open(os.path.join(self.cal_source,'camera_order.txt'),'r')
            cam_ids = fd.read().split('\n')
            fd.close()
            if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        elif self.cal_source_type == 'pytables':
            import tables as PT # PyTables
            assert type(self.cal_source)==PT.File
            results = self.cal_source
            nodes = results.root.calibration.pmat._f_listNodes()
            cam_ids = []
            for node in nodes:
                cam_ids.append( node.name )
            
        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Pmat_fastnx = {}
        self.Res = {}
        self.pmat_inv = {}
        self._helper = {}
        
        if self.cal_source_type == 'normal files':
            res_fd = open(os.path.join(self.cal_source,'Res.dat'),'r')
            for i, cam_id in enumerate(cam_ids):
                fname = 'camera%d.Pmat.cal'%(i+1)
                pmat = load_ascii_matrix(opj(self.cal_source,fname)) # 3 rows x 4 columns
                self.Pmat[cam_id] = pmat
                self.Pmat_fastnx[cam_id] = fast_nx.array(pmat)
                self.Res[cam_id] = map(int,res_fd.readline().split())
                self.pmat_inv[cam_id] = numarray.linear_algebra.generalized_inverse(pmat)
            res_fd.close()

            # load non linear parameters
            rad_files = glob.glob(os.path.join(self.cal_source,'*.rad'))
            assert len(rad_files) < 10
            rad_files.sort() # cheap trick to associate camera number with file
            for cam_id, filename in zip( cam_ids, rad_files ):
                params = {}
                execfile(filename,params)
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    params['K11'], params['K22'], params['K13'], params['K23'],
                    params['kc1'], params['kc2'], params['kc3'], params['kc4'])
                
        elif self.cal_source_type == 'pytables':
            for cam_id in cam_ids:
                pmat = nx.array(results.root.calibration.pmat.__getattr__(cam_id))
                res = tuple(results.root.calibration.resolution.__getattr__(cam_id))
                K = nx.array(results.root.calibration.intrinsic_linear.__getattr__(cam_id))
                nlparams = tuple(results.root.calibration.intrinsic_nonlinear.__getattr__(cam_id))
                self.Pmat[cam_id] = pmat
                self.Pmat_fastnx[cam_id] = fast_nx.array(pmat)
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
        self.cam_combinations_by_size = {}
        for cc in self.cam_combinations:
            self.cam_combinations_by_size.setdefault(len(cc),[]).append(cc)
        self.cam_ids = cam_ids

    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def get_intrinsic_linear(self, cam_id):
        return self._helper[cam_id].get_K()
        
    def get_intrinsic_nonlinear(self, cam_id):
        return self._helper[cam_id].get_nlparams()

    def undistort(self, cam_id, x_kk):
        return self._helper[cam_id].undistort(x_kk[0],x_kk[1])

    def distort(self, cam_id, xl):
        return self._helper[cam_id].distort(xl[0],xl[1])

    def find_best_3d(self, d2,
                          acceptable_distance_pixels = 10.0,
                          minimum_eccentricity = 2.0):
        """

        Finds combination of cameras which uses the most number of
        cameras while minimizing mean reprojection error. Algorithm
        used accepts any camera combination with reprojection error
        less than acceptable_distance_pixels.

        """
        
        least_err_by_n_cameras = {}
        cam_ids = self.cam_ids # shorthand
        max_n_cams = len(cam_ids)
        allA = fast_nx.zeros( (2*max_n_cams,4),'d')
        bad_cam_ids = []
        cam_id2idx = {}
        all2d = {}
        Pmat_fastnx = {}
        Pmat_fastnx = self.Pmat_fastnx # shorthand
        for i,cam_id in enumerate(cam_ids):
            cam_id2idx[cam_id] = i

            # do we have incoming data?
            try:
                value_tuple = d2[cam_id]
            except KeyError:
                bad_cam_ids.append( cam_id )
                continue # don't build this row

            # was a 2d point found?
            values = value_tuple[:2]
            if len( getnan(values)[0] ):
                bad_cam_ids.append( cam_id )
                continue # don't build this row
            x,y = values

            Pmat = Pmat_fastnx[cam_id] # Pmat is 3 rows x 4 columns
            row3 = Pmat[2,:]
            allA[ i*2, : ] = x*row3 - Pmat[0,:]
            allA[ i*2+1, :]= y*row3 - Pmat[1,:]

            all2d[cam_id] = values

        #print allA
        least_err_by_n_cameras = {}
        cam_ids_for_least_err = {}
        X_for_least_err = {}
        for cam_ids_used in self.cam_combinations:
            missing_cam_data = False
            good_A_idx = []
            for cam_id in cam_ids_used:
                if cam_id in bad_cam_ids:
                    missing_cam_data = True
                    break
                else:
                    i = cam_id2idx[cam_id]
                    good_A_idx.extend( (i*2, i*2+1) )
            if missing_cam_data:
                continue
            A = fast_nx.take(allA,good_A_idx)
            #print cam_ids_used
            #print A
            #print
            n_cams = len(cam_ids_used)
            u,d,vt=fast_svd(A)
            X = vt[-1,:]/vt[-1,3] # normalize
            
            mean_dist = 0.0
            n_cams = len(cam_ids_used)
            alpha = 1.0/n_cams
            for cam_id in cam_ids_used:
                orig_x,orig_y = all2d[cam_id]
                Pmat = Pmat_fastnx[cam_id]
                new_xyw = fast_nx.matrixmultiply( Pmat, X )
                new_x, new_y = new_xyw[0:2]/new_xyw[2]
                
                dist = math.sqrt((orig_x-new_x)**2 + (orig_y-new_y)**2)
                mean_dist += dist*alpha

            least_err = least_err_by_n_cameras.get(n_cams,1e16)
            if mean_dist < least_err:
                least_err_by_n_cameras[n_cams] = mean_dist
                cam_ids_for_least_err[n_cams] = cam_ids_used
                X_for_least_err[n_cams] = X[:3]

        # now we have the best estimate for 2 views, 3 views, ...
        best_n_cams = 2
        least_err = least_err_by_n_cameras[2]
        for n_cams in range(3,max_n_cams+1):
            try:
                least_err = least_err_by_n_cameras[n_cams]
            except KeyError:
                break # if we don't have 4, we won't have 5
            if least_err < acceptable_distance_pixels:
                best_n_cams = n_cams

        # now calculate final values
        cam_ids_used = cam_ids_for_least_err[best_n_cams]
        X = X_for_least_err[best_n_cams]
        
        # calculate line3d
        P = []
        for cam_id in cam_ids_used:
            x,y,area,slope,eccentricity, p1,p2,p3,p4 = d2[cam_id]
            if eccentricity > minimum_eccentricity:
                P.append( (p1,p2,p3,p4) )
        if len(P) < 2:
            Lcoords = None
        else:
            P = fast_nx.array(P)
            try:
                u,d,vt=fast_svd(P,full_matrices=True)
                
                P = vt[0,:] # P,Q are planes (take row because this is transpose(V))
                Q = vt[1,:]
                
                # directly to Pluecker line coordinates
                Lcoords = ( -(P[3]*Q[2]) + P[2]*Q[3],
                              P[3]*Q[1]  - P[1]*Q[3],
                            -(P[2]*Q[1]) + P[1]*Q[2],
                            -(P[3]*Q[0]) + P[0]*Q[3],
                            -(P[2]*Q[0]) + P[0]*Q[2],
                            -(P[1]*Q[0]) + P[0]*Q[1] )
            except numarray.linear_algebra.LinearAlgebra2.LinearAlgebraError, exc:
                print 'WARNING:',str(exc)
                Lcoords = None
        return X, Lcoords, cam_ids_used

    def find3d(self, cam_ids_and_points2d, return_X_coords = True, return_line_coords = True ):
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        
        # Construct matrices
        A=[]
        P=[]
        for m, (cam_id,value_tuple) in enumerate(cam_ids_and_points2d):
            if len(value_tuple)==2:
                # only point information ( no line )
                x,y = value_tuple
                return_line_coords = False
            else:
                # get shape information from each view of a blob:
                x,y,area,slope,eccentricity, p1,p2,p3,p4 = value_tuple
            if return_X_coords:
                Pmat = self.Pmat[cam_id] # Pmat is 3 rows x 4 columns
                row3 = Pmat[2,:]
                A.append( x*row3 - Pmat[0,:] )
                A.append( y*row3 - Pmat[1,:] )

            if return_line_coords:
                if eccentricity > MINIMUM_ECCENTRICITY: # require a bit of elongation
                    P.append( (p1,p2,p3,p4) )
        
        # Calculate best point
        if return_X_coords:
            A=nx.array(A)
            u,d,vt=svd(A)
            X = vt[-1,0:3]/vt[-1,3] # normalize
            if not return_line_coords:
                return X

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
            except numarray.linear_algebra.LinearAlgebra2.LinearAlgebraError, exc:
                print 'WARNING:',str(exc)
                Lcoords = None
            except:
                print 'WARNING: unknown error in reconstruct.py'
                print '(you probably have an old version of numarray'
                print 'and SVD did not converge'
                Lcoords = None
        if return_line_coords:
            if return_X_coords:
                return X, Lcoords
            else:
                return Lcoords

    def find2d(self,cam_id,X,Lcoords=None,distorted=False):
        # see Hartley & Zisserman (2003) p. 449
        if len(X) == 3:
            X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
        Pmat = self.Pmat[cam_id]
        x=nx.dot(Pmat,X)

        x = x[0:2,0]/x[2,0] # normalize
        if distorted:
            xd, yd = self.distort(cam_id, x)
            x[0] = xd
            x[1] = yd

        # XXX Impossible to distort Lcoords. The image of the line
        # should be distorted downstream, but this is not usually
        # possible.
        
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
