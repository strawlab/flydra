# $Id$
import os, glob, sys, math
opj=os.path.join
import numpy as nx
import numpy
svd = numpy.linalg.svd
import numpy as fast_nx
fast_svd = numpy.linalg.svd
import flydra.reconstruct_utils as reconstruct_utils # in pyrex/C for speed
import time
from flydra.common_variables import MINIMUM_ECCENTRICITY

L_i = nx.array([0,0,0,1,3,2])
L_j = nx.array([1,2,3,2,1,3])

def rq(X):
    def cross(a,b):
        return numpy.array( (( a[1]*b[2]-a[2]*b[1] ),
                             ( a[2]*b[0]-a[0]*b[2] ),
                             ( a[0]*b[1]-a[1]*b[0] )) )

    def norm(a):
        return numpy.sqrt(numpy.sum(a**2))

    import scipy.linalg
    
    Qt, Rt = scipy.linalg.qr(X.transpose())
    Rt = Rt.transpose()
    Qt = Qt.transpose()

    Qu = []

    Qu.append( cross(Rt[1,:], Rt[2,:] ) )
    Qu[0] = Qu[0]/norm(Qu[0])

    Qu.append( cross(Qu[0], Rt[2,:] ) )
    Qu[1] = Qu[1]/norm(Qu[1])

    Qu.append( cross(Qu[0], Qu[1] ) )

    Qu = numpy.asarray(Qu)

    R = numpy.matrixmultiply( Rt, Qu.transpose())
    Q = numpy.matrixmultiply( Qu, Qt )

    return R, Q

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
    Lmatrix = nx.zeros((4,4),nx.float64)
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
        U = Va/Vamags[:,nx.newaxis]
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
    """

    See Hartley & Zisserman (2003) p. 70
    """
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
    determinant = numpy.linalg.det
    
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
        
class SingleCameraCalibration:
    def __init__(self,
                 cam_id=None,
                 Pmat=None,
                 res=None,
                 pp=None,
                 helper=None):
        """

        Required arguments
        ------------------
        cam_id - string identifying camera
        Pmat - camera calibration matrix (3 x 4)
        res - resolution (width,height)
        pp - pricipal point (point on image plane on optical axis) (x,y)

        Optional arguments
        ------------------
        helper - has camera distortion parameters
        """
        if type(cam_id) != str:
            raise TypeError('cam_id must be string')
        pm = numpy.asarray(Pmat)
        if pm.shape != (3,4):
            raise ValueError('Pmat must have shape (3,4)')
        if len(res) != 2:
            raise ValueError('len(res) must be 2')
        
        self.cam_id=cam_id
        self.Pmat=Pmat
        self.res=res

        if pp is None:
            pp = self.res[0]/2.0,self.res[1]/2.0
        if len(pp) != 2:
            raise ValueError('len(pp) must be 2')
        self.pp = pp

        if helper is None:
            helper = reconstruct_utils.ReconstructHelper(1,1, # focal length
                                                         0,0, # image center
                                                         0,0, # radial distortion
                                                         0,0) # tangential distortion
        if not isinstance(helper,reconstruct_utils.ReconstructHelper):
            raise TypeError('helper must be reconstruct_utils.ReconstructHelper instance')
        self.helper = helper

        self.pmat_inv = numpy.linalg.pinv(self.Pmat)
        
    def get_cam_center(self):
        return pmat2cam_center(self.Pmat)
    def get_M(self):
        """return parameters except extrinsic translation params"""
        return self.Pmat[:,:3]
    def get_KR(self):
        """return intrinsic params (K) and extrinsic rotation/scale params (R)"""
        M = self.get_M()
        K,R = rq(M)
        if K[2,2] != 0.0:
            # normalize K
            K = K/K[2,2]
        return K,R
    def get_extrinsic_parameter_matrix(self):
        """contains rotation and translation information"""
        C_ = self.get_cam_center()
        K,R = self.get_KR()
        t = numpy.matrixmultiply( -R, C_ )
        ext = numpy.concatenate( (R, t), axis=1 )
        return ext
    def get_optical_axis(self):
        # project back through principal point to get 3D line
        import flydra.geom
        c1 = self.get_cam_center()[:,0]
        
        x2d = (self.pp[0],self.pp[1],1.0)
        c2 = numpy.matrixmultiply(self.pmat_inv, as_column(x2d))[:,0]
        c2 = c2[:3]/c2[3]
        c1 = flydra.geom.ThreeTuple(c1)
        c2 = flydra.geom.ThreeTuple(c2)
        return flydra.geom.line_from_points( c1, c2 )
        
    def get_up_vector(self):
        # create up vector from image plane
        x2d_a = (self.pp[0],self.pp[1],1.0)
        c2_a = numpy.matrixmultiply(self.pmat_inv, as_column(x2d_a))[:,0]
        c2_a = c2_a[:3]/c2_a[3]
        
        x2d_b = (self.pp[0],self.pp[1]+1,1.0)
        c2_b = numpy.matrixmultiply(self.pmat_inv, as_column(x2d_b))[:,0]
        c2_b = c2_b[:3]/c2_b[3]

        up_dir = c2_b-c2_a
        return norm_vec(up_dir)

    def to_file(self,filename):
        fd = open(filename,'wb')
        fd.write(    'cam_id = "%s"\n'%self.cam_id)
        
        fd.write(    'pmat = [\n')
        for row in self.Pmat:
            fd.write('        [%s, %s, %s, %s],\n'%tuple([repr(x) for x in row]))
        fd.write(    '       ]\n')
        
        fd.write(    'res = (%d,%d)\n'%(self.res[0],self.res[1]))
        fd.write(    'pp = (%s,%s)\n'%(repr(self.pp[0]),repr(self.pp[1])))
                     
        fd.write(    'K = [\n')
        for row in self.helper.get_K():
            fd.write('     [%s, %s, %s],\n'%tuple([repr(x) for x in row]))
        fd.write(    '    ]\n')

        k1,k2,p1,p2 = self.helper.get_nlparams()
        fd.write(    'radial_params = %s, %s\n'%(repr(k1),repr(k2)))
        fd.write(    'tangential_params = %s, %s\n'%(repr(p1),repr(p2)))
        
def SingleCameraCalibration_fromfile(filename):
    params={}
    execfile(filename,params)
    pmat = numpy.asarray(params['pmat'])
    K = numpy.asarray(params['K'])
    cam_id = params['cam_id']
    res = params['res']
    pp = params['pp']
    k1,k2 = params['radial_params']
    p1,p2 = params['tangential_params']
    
    fc1 = K[0,0]
    cc1 = K[0,2]
    fc2 = K[1,1]
    cc2 = K[1,2]

    helper = reconstruct_utils.ReconstructHelper(fc1,fc2, # focal length
                                                 cc1,cc2, # image center
                                                 k1, k2, # radial distortion
                                                 p1, p2) # tangential distortion
    return SingleCameraCalibration(cam_id=cam_id,
                                   Pmat=pmat,
                                   res=res,
                                   pp=pp,
                                   helper=helper)
    
class Reconstructor:
    def __init__(self,
                 cal_source = None,
                 ):
        self.cal_source = cal_source

        if type(self.cal_source) in [str,unicode]:
            self.cal_source_type = 'normal files'
        elif hasattr(self.cal_source,'__len__'): # is sequence
            for i in range(len(self.cal_source)):
                if not isinstance(self.cal_source[i],SingleCameraCalibration):
                    raise TypeError('If calsource is a sequence, it must '
                                    'be a string specifying calibration '
                                    'directory or a sequence of '
                                    'SingleCameraCalibration instances.')
            self.cal_source_type = 'SingleCameraCalibration instances'
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
        elif self.cal_source_type=='SingleCameraCalibration instances':
            cam_ids = [scci.cam_id for scci in self.cal_source]
            
        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Res = {}
        self._helper = {}
        
        if self.cal_source_type == 'normal files':
            res_fd = open(os.path.join(self.cal_source,'Res.dat'),'r')
            for i, cam_id in enumerate(cam_ids):
                fname = 'camera%d.Pmat.cal'%(i+1)
                pmat = load_ascii_matrix(opj(self.cal_source,fname)) # 3 rows x 4 columns
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = map(int,res_fd.readline().split())
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
                self.Res[cam_id] = res
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    K[0,0], K[1,1], K[0,2], K[1,2],
                    nlparams[0], nlparams[1], nlparams[2], nlparams[3])
        elif self.cal_source_type=='SingleCameraCalibration instances':
            # find instance
            for cam_id in cam_ids:
                for scci in self.cal_source:
                    if scci.cam_id==cam_id:
                        break
                self.Pmat[cam_id] = scci.Pmat
                self.Res[cam_id] = scci.res
                self._helper[cam_id] = scci.helper

        self.pmat_inv = {}
        for cam_id in cam_ids:
            # For speed reasons, make sure self.Pmat has only numpy arrays.
            self.Pmat[cam_id] = numpy.array(self.Pmat[cam_id])
            
            self.pmat_inv[cam_id] = numpy.linalg.pinv(self.Pmat[cam_id])

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

    def get_cam_ids(self):
        return self.cam_ids

    def save_to_h5file(self, h5file, OK_to_delete_old_calibration=False):
        """create groups with calibration information"""

        import tables as PT # pytables
        class AdditionalInfo(PT.IsDescription):
            cal_source_type      = PT.StringCol(20)
            cal_source           = PT.StringCol(80)
            minimum_eccentricity = PT.Float32Col() # record what parameter was used during reconstruction
        pytables_filt = numpy.asarray
        ct = h5file.createTable # shorthand
        root = h5file.root # shorthand
        
        cam_ids = self.Pmat.keys()
        cam_ids.sort()

        if hasattr(root,'calibration'):
            if OK_to_delete_old_calibration:
                h5file.removeNode( root.calibration, recursive=True )
            else:
                raise RuntimeError('not deleting old calibration.')
            
        cal_group = h5file.createGroup(root,'calibration')
            
        pmat_group = h5file.createGroup(cal_group,'pmat')
        for cam_id in cam_ids:
            h5file.createArray(pmat_group, cam_id,
                               pytables_filt(self.get_pmat(cam_id)))
        res_group = h5file.createGroup(cal_group,'resolution')
        for cam_id in cam_ids:
            res = self.get_resolution(cam_id)
            h5file.createArray(res_group, cam_id, pytables_filt(res))

        intlin_group = h5file.createGroup(cal_group,'intrinsic_linear')
        for cam_id in cam_ids:
            intlin = self.get_intrinsic_linear(cam_id)
            h5file.createArray(intlin_group, cam_id, pytables_filt(intlin))

        intnonlin_group = h5file.createGroup(cal_group,'intrinsic_nonlinear')
        for cam_id in cam_ids:
            h5file.createArray(intnonlin_group, cam_id,
                               pytables_filt(self.get_intrinsic_nonlinear(cam_id)))

        h5additional_info = ct(cal_group,'additional_info', AdditionalInfo,
                               '')
        row = h5additional_info.row
        row['cal_source_type'] = self.cal_source_type
        row['cal_source'] = self.cal_source
        row['minimum_eccentricity'] = MINIMUM_ECCENTRICITY
        row.append()
        h5additional_info.flush()
                
    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]
    
    def get_camera_center(self, cam_id):
        return pmat2cam_center(self.Pmat[cam_id])

    def get_intrinsic_linear(self, cam_id):
        return self._helper[cam_id].get_K()
        
    def get_intrinsic_nonlinear(self, cam_id):
        return self._helper[cam_id].get_nlparams()

    def undistort(self, cam_id, x_kk):
        return self._helper[cam_id].undistort(x_kk[0],x_kk[1])

    def distort(self, cam_id, xl):
        return self._helper[cam_id].distort(xl[0],xl[1])

    def get_reconstruct_helper_dict(self):
        return self._helper

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
            except Exception, exc:
                print 'WARNING svd exception:',str(exc)
                Lcoords = None
            except:
                print 'WARNING: unknown error in reconstruct.py'
                print '(you probably have an old version of numarray'
                print 'and SVD did not converge)'
                Lcoords = None
        if return_line_coords:
            if return_X_coords:
                return X, Lcoords
            else:
                return Lcoords

    def find2d(self,cam_id,X,Lcoords=None,distorted=False):
        # see Hartley & Zisserman (2003) p. 449
        if type(X)==tuple or type(X)==list:
            rank1=True
        else:
            rank1 = len(X.shape)==1 # assuming array type
        if rank1:
            # make homogenous coords, rank2
            if len(X) == 3:
                X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
            else:
                X = X[:,nx.newaxis] # 4 rows, 1 column
        else:
            X = nx.transpose(X) # 4 rows, N columns
        Pmat = self.Pmat[cam_id]
        x=nx.dot(Pmat,X)
        
        x = x[0:2,:]/x[2,:] # normalize
        
        if distorted:
            if rank1:
                xd, yd = self.distort(cam_id, x)
                x[0] = xd
                x[1] = yd
            else:
                N_pts = x.shape[1]
                for i in range(N_pts):
                    xpt = x[:,i]
                    xd, yd = self.distort(cam_id, xpt)
                    x[0,i]=xd
                    x[1,i]=yd

        # XXX The rest of this function hasn't been (recently) checked
        # for >1 points. (i.e. not rank1)
        
        if Lcoords is not None:
            if distorted:
                
                # Impossible to distort Lcoords. The image of the line
                # could be distorted downstream.
                
                raise RuntimeError('cannot (easily) distort line')
            
            if not rank1:
                raise NotImplementedError('Line reconstruction not yet implemented for rank-2 data')
                
            # see Hartley & Zisserman (2003) p. 198, eqn 8.2
            L = Lcoords2Lmatrix(Lcoords)
            # XXX could be made faster by pre-computing line projection matrix
            lx = nx.dot(Pmat, nx.dot(L,nx.transpose(Pmat)))
            l3 = lx[2,1], lx[0,2], lx[1,0] #(p. 581)
            return x, l3
        else:
            if rank1:
                # convert back to rank1
                return x[:,0]
            return x

    def find3d_single_cam(self,cam_id,x):
        return nx.dot(self.pmat_inv[cam_id], as_column(x))

    def get_SingleCameraCalibration(self,cam_id):
        return SingleCameraCalibration(cam_id=cam_id,
                                       Pmat=self.Pmat[cam_id],
                                       res=self.Res[cam_id],
                                       helper=self._helper[cam_id])
