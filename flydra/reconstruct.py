import os, glob, sys, math
opj=os.path.join
import numpy as np
import numpy as nx
import numpy
import flydra.reconstruct_utils as reconstruct_utils # in pyrex/C for speed
import time
from flydra.common_variables import MINIMUM_ECCENTRICITY as DEFAULT_MINIMUM_ECCENTRICITY
import scipy.linalg
import traceback
import flydra.pmat_jacobian
import flydra.talign
import xml.etree.ElementTree as ET
import StringIO, warnings
import cgtypes
from optparse import OptionParser

R2D = 180.0/np.pi
D2R = np.pi/180.0

WARN_CALIB_DIFF = False

L_i = nx.array([0,0,0,1,3,2])
L_j = nx.array([1,2,3,2,1,3])


def mat2quat(m):
    """Initialize q from either a mat3 or mat4 and returns self."""
    # XXX derived from cgkit-1.2.0/cgtypes.pyx

    class PseudoQuat(object):
        pass

    q=PseudoQuat()

    d1 = m[0,0]
    d2 = m[1,1]
    d3 = m[2,2]
    t = d1+d2+d3+1.0
    if t>0.0:
        s = 0.5/np.sqrt(t)
        q.w = 0.25/s
        q.x = (m[2,1]-m[1,2])*s
        q.y = (m[0,2]-m[2,0])*s
        q.z = (m[1,0]-m[0,1])*s
    else:
        ad1 = abs(d1)
        ad2 = abs(d2)
        ad3 = abs(d3)
        if ad1>=ad2 and ad1>=ad3:
            s = np.sqrt(1.0+d1-d2-d3)*2.0
            q.x = 0.5/s
            q.y = (m[0,1]+m[1,0])/s
            q.z = (m[0,2]+m[2,0])/s
            q.w = (m[1,2]+m[2,1])/s
        elif ad2>=ad1 and ad2>=ad3:
            s = np.sqrt(1.0+d2-d1-d3)*2.0
            q.x = (m[0,1]+m[1,0])/s
            q.y = 0.5/s
            q.z = (m[1,2]+m[2,1])/s
            q.w = (m[0,2]+m[2,0])/s
        else:
            s = np.sqrt(1.0+d3-d1-d2)*2.0
            q.x = (m[0,2]+m[2,0])/s
            q.y = (m[1,2]+m[2,1])/s
            q.z = 0.5/s
            q.w = (m[0,1]+m[1,0])/s

    return q

def my_rq(M):
    """RQ decomposition, ensures diagonal of R is positive"""
    R,K = scipy.linalg.rq(M)
    n = R.shape[0]
    for i in range(n):
        if R[i,i]<0:
            # I checked this with Mathematica. Works if R is upper-triangular.
            R[:,i] = -R[:,i]
            K[i,:] = -K[i,:]
##    if R[0,0]<0:
##        # I checked this with Mathematica. Works if R is upper-triangular.
##        R[0,0] = -R[0,0]
##        K[0,:] = -K[0,:]
    return R,K

def filter_comments(lines_tmp):
    lines = []
    for line in lines_tmp:
        try:
            comment_idx = line.index('#')
            no_comment_line = line[:comment_idx]
            no_comment_line.strip()
            if len(no_comment_line):
                line = no_comment_line
            else:
                continue # nothing on this line
        except ValueError:
            pass
        lines.append(line)
    return lines

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    fd.close()
    lines = buf.split('\n')[:-1]
    lines = filter_comments( lines )
    return nx.array([map(float,line.split()) for line in lines])

def test_load_ascii_matrix():
    contents = '''   5.1468544e+00  -1.8892933e-01  -3.7855949e+00  -6.3683613e+02
  -4.1691492e-01  -6.2570417e+00  -7.0782063e-01   2.4157199e+03
  -1.7169043e-03  -2.2409402e-04  -4.2079099e-03   6.6955409e+00
'''
    import tempfile
    fname = tempfile.mktemp()
    fd = open(fname,mode='w')
    fd.write(contents)
    fd.close()
    result = load_ascii_matrix(fname)
    assert result.shape == (3,4)

def save_ascii_matrix(M,fd,isint=False):
    def fmt(f):
        if isint:
            return '%d'%f
        else:
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

def line_direction_distance(Lcoords_A, Lcoords_B):
    """find distance between 2 line directions, A and B, ignore sign"""
    dir_vecA = cgtypes.vec3(Lcoords_A[2], -Lcoords_A[4], Lcoords_A[5])
    dir_vecB = cgtypes.vec3(Lcoords_B[2], -Lcoords_B[4], Lcoords_B[5])
    ## cos_angle = dir_vecA.dot( dir_vecB )
    ## angle = np.arccos( cos_angle )
    angle = dir_vecA.angle( dir_vecB )
    flipped_angle = np.pi-angle
    return min(angle,flipped_angle)

def norm_vec(V):
    Va = np.asarray(V,dtype=np.float64) # force double precision floats
    if len(Va.shape)==1:
        # vector
        U = Va/math.sqrt(Va[0]**2 + Va[1]**2 + Va[2]**2) # normalize
    else:
        assert Va.shape[1] == 3
        Vamags = nx.sqrt(Va[:,0]**2 + Va[:,1]**2 + Va[:,2]**2)
        U = Va/Vamags[:,nx.newaxis]
    return U

def line_direction(Lcoords):
    """convert from Pluecker coordinates to a direction"""
    # should maybe be PQmath.pluecker_to_orient
    L = nx.asanyarray(Lcoords)
    if L.ndim==1:
        # single line coord
        U = nx.array((-L[2], L[4], -L[5]))
    else:
        assert L.ndim==2
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
    L = nx.dot(A,nx.transpose(B)) - nx.dot(B,nx.transpose(A))
    return Lmatrix2Lcoords(L)

def pmat2cam_center(P):
    """

    See Hartley & Zisserman (2003) p. 163
    """
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
    http://mail.python.org/pipermail/python-list/2001-January/067815.html

    This is also called the power set:
    http://en.wikipedia.org/wiki/Power_set
    """
    N = len(L)
    return [ [ L[i] for i in range(N)
                if X & (1L<<i) ]
        for X in range(2**N) ]

intrinsic_normalized_eps = 1e-6
def normalize_pmat(pmat):
    pmat_orig = pmat
    M = pmat[:,:3]
    t = pmat[:,3,numpy.newaxis]
    K,R = my_rq(M)
    if abs(K[2,2]-1.0)>intrinsic_normalized_eps:
        pmat = pmat/K[2,2]
    assert numpy.allclose(pmat2cam_center(pmat_orig),pmat2cam_center(pmat))
    return pmat

def intersect_planes_to_find_line(P):
    # Is there a faster way when len(P)==2?
    try:
        u,d,vt=scipy.linalg.svd(P,full_matrices=True)
        # "two columns of V corresponding to the two largest singular
        # values span the best rank 2 approximation to A and may be
        # used to define the line of intersection of the planes"
        # (Hartley & Zisserman, p. 323)

        # get planes (take row because this is transpose(V))
        planeP = vt[0,:]
        planeQ = vt[1,:]

        # directly to Pluecker line coordinates
        Lcoords = ( -(planeP[3]*planeQ[2]) + planeP[2]*planeQ[3],
                      planeP[3]*planeQ[1]  - planeP[1]*planeQ[3],
                    -(planeP[2]*planeQ[1]) + planeP[1]*planeQ[2],
                    -(planeP[3]*planeQ[0]) + planeP[0]*planeQ[3],
                    -(planeP[2]*planeQ[0]) + planeP[0]*planeQ[2],
                    -(planeP[1]*planeQ[0]) + planeP[0]*planeQ[1] )
    except Exception, exc:
        print 'WARNING svd exception:',str(exc)
        Lcoords = None
    except:
        print 'WARNING: unknown error in reconstruct.py'
        print '(you probably have an old version of numarray'
        print 'and SVD did not converge)'
        Lcoords = None
    return Lcoords

def do_3d_operations_on_2d_point(helper,
                                 x0u, y0u, # undistorted coords
                                 pmat_inv, pmat_meters_inv,
                                 camera_center, camera_center_meters,
                                 x0_abs, y0_abs, # distorted coords
                                 rise, run):
    """this function is a hack"""

    matrixmultiply = numpy.dot
    svd = numpy.dual.svd # use fastest ATLAS/fortran libraries

    found_point_image_plane = [x0u,y0u,1.0]

    if not numpy.isnan(rise):
        # calculate plane containing camera origin and found line
        # in 3D world coords

        # Step 1) Find world coordinates points defining plane:
        #    A) found point
        X0=matrixmultiply(pmat_inv,found_point_image_plane)

        #    B) another point on found line
        if 1:
            # The right way - convert the slope in distorted coords to
            # slope in undistorted coords.
            dirvec = numpy.array([run,rise])
            dirvec = dirvec/numpy.sqrt( numpy.sum( dirvec**2 )) # normalize
            dirvec *= 0.1 # make really small
            x1u, y1u = helper.undistort(x0_abs+dirvec[0],y0_abs+dirvec[1])
            X1=matrixmultiply(pmat_inv,[x1u,y1u,1.0])
        else:
            # The wrong way - assume slope is the same in distorted and undistorted coords
            x1u, y1u = x0u+run, y0u+rise
            X1=matrixmultiply(pmat_inv,[x1u,y1u,1.0])

        #    C) world coordinates of camera center already known

        # Step 2) Find world coordinates of plane
        A = nx.array( [ X0, X1, camera_center] ) # 3 points define plane
        try:
            u,d,vt=svd(A,full_matrices=True)
        except:
            print 'rise,run',rise,run
            print 'pmat_inv',pmat_inv
            print 'X0, X1, camera_center',X0, X1, camera_center
            raise
        Pt = vt[3,:] # plane parameters

        p1,p2,p3,p4 = Pt[0:4]
        if numpy.isnan(p1):
            print 'ERROR: SVD returned nan'
    else:
        p1,p2,p3,p4 = numpy.nan,numpy.nan, numpy.nan,numpy.nan

    # calculate pluecker coords of 3D ray from camera center to point
    # calculate 3D coords of point on image plane
    X0meters = numpy.dot(pmat_meters_inv, found_point_image_plane )
    X0meters = X0meters[:3]/X0meters[3] # convert to shape = (3,)
    # project line
    pluecker_meters = pluecker_from_verts(X0meters,camera_center_meters)
    (ray0, ray1, ray2, ray3, ray4, ray5) = pluecker_meters # unpack

    return (p1, p2, p3, p4,
            ray0, ray1, ray2, ray3, ray4, ray5)

def angles_near(a,b, eps=None, mod_pi=False,debug=False):
    """compare if angles a and b are within eps of each other. assumes radians"""

    if mod_pi:
        r1 = angles_near(a, b, eps=eps, mod_pi=False,debug=debug)
        r2 = angles_near(a+np.pi, b, eps=eps, mod_pi=False,debug=debug)
        return r1 or r2

    diff = abs(a-b)

    diff = diff%(2*numpy.pi) # 0 <= diff <= 2pi

    result = abs(diff) < eps
    if debug:
        print 'a',a*R2D
        print 'b',b*R2D
        print 'diff',diff*R2D
    if abs(diff-2*numpy.pi) < eps:
        result = result or True
    return result

def test_angles_near():
    pi = np.pi
    for A in [-2*pi, -1*pi, -pi/2, 0, pi/2, pi, 2*pi]:
        assert angles_near(A,A,eps=1e-15)==True
        assert angles_near(A,A+np.pi/8,eps=np.pi/4)==True
        assert angles_near(A,A-np.pi/8,eps=np.pi/4)==True

        assert angles_near(A,A+1.1*np.pi,eps=np.pi/4)==False
        assert angles_near(A,A+1.1*np.pi,eps=np.pi/4,mod_pi=True)==True

def test_angles_near2():
    a = 1.51026430701
    b = 2.92753197003
    assert angles_near(a,b,eps=10.0*D2R,mod_pi=True) == False

class SingleCameraCalibration:
    """Complete per-camera calibration information.

    Parameters
    ----------
    cam_id : string
      identifying camera
    Pmat : ndarray (3x4)
      camera calibration matrix
    res : sequence of length 2
      resolution (width,height)
    helper : :class:`CamParamsHelper` instance
      (optional) specifies camera distortion parameters
    scale_factor : None or float
      specifies how to convert your units to meters
    """
    def __init__(self,
                 cam_id=None, # non-optional
                 Pmat=None,   # non-optional
                 res=None,    # non-optional
                 helper=None,
                 scale_factor=None, # scale_factor is for conversion to meters (e.g. should be 1e-3 if your units are mm)
                 no_error_on_intrinsic_parameter_problem = False,
                 ):
        if type(cam_id) != str:
            raise TypeError('cam_id must be string')
        pm = numpy.asarray(Pmat)
        if pm.shape != (3,4):
            raise ValueError('Pmat must have shape (3,4)')
        if len(res) != 2:
            raise ValueError('len(res) must be 2 (res = %s)'%repr(res))

        self._scaled_cache = {}

        self.cam_id=cam_id
        self.Pmat=Pmat
        self.res=res

        if helper is None:
            M = numpy.asarray(Pmat)
            cam_center = pmat2cam_center(M)

            intrinsic_parameters, cam_rotation = my_rq(M[:,:3])
            #intrinsic_parameters = intrinsic_parameters/intrinsic_parameters[2,2] # normalize
            eps = 1e-6
            if abs(intrinsic_parameters[2,2]-1.0)>eps:
                if no_error_on_intrinsic_parameter_problem:
                    warnings.warn('expected last row/col of intrinsic '
                                  'parameter matrix to be unity. It is %s'%
                                  intrinsic_parameters[2,2])
                    intrinsic_parameters = \
                                 intrinsic_parameters/intrinsic_parameters[2,2]
                    print intrinsic_parameters
                else:
                    print ('WARNING: expected last row/col of intrinsic '
                           'parameter matrix to be unity')
                    print 'intrinsic_parameters[2,2]',intrinsic_parameters[2,2]
                    raise ValueError('expected last row/col of intrinsic '
                                     'parameter matrix to be unity')

            fc1 = intrinsic_parameters[0,0]
            cc1 = intrinsic_parameters[0,2]
            fc2 = intrinsic_parameters[1,1]
            cc2 = intrinsic_parameters[1,2]

            helper = reconstruct_utils.ReconstructHelper(fc1,fc2, # focal length
                                                         cc1,cc2, # image center
                                                         0,0, # radial distortion
                                                         0,0) # tangential distortion
        if not isinstance(helper,reconstruct_utils.ReconstructHelper):
            raise TypeError('helper must be reconstruct_utils.ReconstructHelper instance')
        self.helper = helper

        self.pmat_inv = numpy.linalg.pinv(self.Pmat)
        self.scale_factor = scale_factor

    def __ne__(self,other):
        return not (self==other)

    def __eq__(self,other):
        return ((self.cam_id == other.cam_id) and
                numpy.allclose(self.Pmat,other.Pmat) and
                numpy.allclose(self.res,other.res) and
                self.helper == other.helper)

    def get_pmat(self):
        return self.Pmat

    def get_aligned_copy(self, M):
        if self.scale_factor != 1.0:
            warnings.warn('aligning calibration without unity scale')
        aligned_Pmat = flydra.talign.align_pmat(M,self.Pmat)
        aligned = SingleCameraCalibration(cam_id=self.cam_id,
                                          Pmat=aligned_Pmat,
                                          res=self.res,
                                          helper=self.helper,
                                          scale_factor=self.scale_factor)
        return aligned

    def get_scaled(self,scale_factor):
        """change units (e.g. from mm to meters)

        Note: some of the data structures are shared with the unscaled original
        """
        if scale_factor in self._scaled_cache:
            return self._scaled_cache[scale_factor]

        scale_array = numpy.ones((3,4))
        scale_array[:,3] = scale_factor # mulitply last column by scale_factor
        scaled_Pmat = scale_array*self.Pmat # element-wise multiplication

        if self.scale_factor is not None:
            new_scale_factor = self.scale_factor/scale_factor
        else:
            new_scale_factor = None

        scaled = SingleCameraCalibration(cam_id=self.cam_id,
                                         Pmat=scaled_Pmat,
                                         res=self.res,
                                         helper=self.helper,
                                         scale_factor=new_scale_factor)
        self._scaled_cache[scale_factor] = scaled
        return scaled
    def get_res(self):
        return self.res

    def get_cam_center(self):
        """get the 3D location of the camera center in world coordinates"""
        # should be called get_camera_center?
        return pmat2cam_center(self.Pmat)
    def get_M(self):
        """return parameters except extrinsic translation params"""
        return self.Pmat[:,:3]
    def get_t(self):
        """return extrinsic translation parameters"""
        return self.Pmat[:,3,numpy.newaxis]
    def get_KR(self):
        """return intrinsic params (K) and extrinsic rotation/scale params (R)"""
        M = self.get_M()
        K,R = my_rq(M)
##        if K[2,2] != 0.0:
##            # normalize K
##            K = K/K[2,2]
        return K,R
    def get_mean_focal_length(self):
        K,R = self.get_KR()
        return (K[0,0]+K[1,1])/2.0
    def get_image_center(self):
        K,R = self.get_KR()
        return K[0,2], K[1,2]
    def get_extrinsic_parameter_matrix(self):
        """contains rotation and translation information"""
        C_ = self.get_cam_center()
        K,R = self.get_KR()
        t = numpy.dot( -R, C_ )
        ext = numpy.concatenate( (R, t), axis=1 )
        return ext

    def get_example_3d_point_creating_image_point(self,image_point,w_val=1.0):
        # project back through principal point to get 3D line
        c1 = self.get_cam_center()[:,0]

        x2d = (image_point[0],image_point[1],1.0)
        c2 = numpy.dot(self.pmat_inv, as_column(x2d))[:,0]
        c2 = c2[:3]/c2[3]

        direction = c2-c1
        direction = direction/numpy.sqrt(numpy.sum(direction**2))
        c3 = c1+direction*w_val
        return c3

    def get_optical_axis(self):
        # project back through principal point to get 3D line
        #import flydra.geom as geom
        import flydra.fastgeom as geom
        c1 = self.get_cam_center()[:,0]
        pp = self.get_image_center()

        x2d = (pp[0],pp[1],1.0)
        c2 = numpy.dot(self.pmat_inv, as_column(x2d))[:,0]
        c2 = c2[:3]/c2[3]
        c1 = geom.ThreeTuple(c1)
        c2 = geom.ThreeTuple(c2)
        return geom.line_from_points( c1, c2 )

    def get_up_vector(self):
        # create up vector from image plane
        pp = self.get_image_center()
        x2d_a = (pp[0],pp[1],1.0)
        c2_a = numpy.dot(self.pmat_inv, as_column(x2d_a))[:,0]
        c2_a = c2_a[:3]/c2_a[3]

        x2d_b = (pp[0],pp[1]+1,1.0)
        c2_b = numpy.dot(self.pmat_inv, as_column(x2d_b))[:,0]
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

        fd.write(    'K = [\n')
        for row in self.helper.get_K():
            fd.write('     [%s, %s, %s],\n'%tuple([repr(x) for x in row]))
        fd.write(    '    ]\n')

        k1,k2,p1,p2 = self.helper.get_nlparams()
        fd.write(    'radial_params = %s, %s\n'%(repr(k1),repr(k2)))
        fd.write(    'tangential_params = %s, %s\n'%(repr(p1),repr(p2)))

    def get_sba_format_line(self):
        # From the sba demo/README.txt::
        #
        #   The file for the camera motion parameters has a separate
        #   line for every camera, each line containing 12 parameters
        #   (the five intrinsic parameters in the order focal length
        #   in x pixels, principal point coordinates in pixels, aspect
        #   ratio [i.e. focalY/focalX] and skew factor, plus a 4
        #   element quaternion for rotation and a 3 element vector for
        #   translation).
        K,R = self.get_KR()
        t = self.get_t()[:,0] # drop a dimension

        # extract sba parameters from intrinsic parameter matrix
        focal = K[0,0]
        ppx = K[0,2]
        ppy = K[1,2]
        aspect = K[1,1]/K[0,0]
        skew = K[0,1]

        # extract rotation quaternion from rotation matrix
        q=mat2quat(R)

        qw=q.w; qx=q.x; qy=q.y; qz=q.z
        t0=t[0]; t1=t[1]; t2=t[2]
        result = ('%(focal)s %(ppx)s %(ppy)s %(aspect)s %(skew)s '
                  '%(qw)s %(qx)s %(qy)s %(qz)s '
                  '%(t0)s %(t1)s %(t2)s'%locals())
        return result

    def get_3D_plane_and_ray(self, x0d, y0d, slope):
        """return a ray and plane defined by a point and a slope at that point

        The 2D distorted point (x0d, y0d) is transfored to a 3D
        ray. Together with the slope at that point, a 3D plane that
        passes through the 2D point in the direction of the slope is
        returned.

        """
        # undistort
        x0u,y0u = self.helper.undistort(x0d,y0d)
        rise=slope
        run=1.0
        meter_me=self.get_scaled(self.scale_factor)
        pmat_meters_inv = meter_me.pmat_inv
        # homogeneous coords for camera centers
        camera_center = np.ones((4,))
        camera_center[:3] = self.get_cam_center()[:,0]
        camera_center_meters = np.ones((4,))
        camera_center_meters[:3] = meter_me.get_cam_center()[:,0]
        try:
            tmp = do_3d_operations_on_2d_point(self.helper, x0u, y0u,
                                               self.pmat_inv, pmat_meters_inv,
                                               camera_center, camera_center_meters,
                                               x0d, y0d, rise, run)
        except:
            print 'camera_center',camera_center
            print 'camera_center_meters',camera_center_meters
            raise
        (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4, ray5) = tmp
        plane = (p1, p2, p3, p4)
        ray = (ray0, ray1, ray2, ray3, ray4, ray5)
        return plane,ray

    def add_element(self,parent):
        """add self as XML element to parent"""
        assert ET.iselement(parent)
        elem = ET.SubElement(parent,"single_camera_calibration")

        cam_id = ET.SubElement(elem, "cam_id")
        cam_id.text = self.cam_id

        pmat = ET.SubElement(elem, "calibration_matrix")
        fd = StringIO.StringIO()
        save_ascii_matrix(self.Pmat,fd)
        mystr = fd.getvalue()
        mystr = mystr.strip()
        mystr = mystr.replace('\n','; ')
        pmat.text = mystr
        fd.close()

        res = ET.SubElement(elem, "resolution")
        res.text = ' '.join(map(str,self.res))

        scale_factor = ET.SubElement(elem, "scale_factor")
        scale_factor.text = str(self.scale_factor)

        self.helper.add_element( elem )

    def as_obj_for_json(self):
        result = dict(cam_id=self.cam_id,
                      calibration_matrix= [ row.tolist() for row in self.Pmat ],
                      resolution=list(self.res),
                      scale_factor=self.scale_factor,
                      non_linear_parameters=self.helper.as_obj_for_json(),
                      )
        return result

def SingleCameraCalibration_fromfile(filename):
    params={}
    execfile(filename,params)
    pmat = numpy.asarray(params['pmat']) # XXX redundant information in pmat and K
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
                                   helper=helper)

def SingleCameraCalibration_from_xml(elem):
    assert ET.iselement(elem)
    assert elem.tag == "single_camera_calibration"
    cam_id = elem.find("cam_id").text
    pmat = numpy.array(numpy.mat(elem.find("calibration_matrix").text))
    res = numpy.array(numpy.mat(elem.find("resolution").text))[0,:]
    scale_factor = float(elem.find("scale_factor").text)
    helper_elem = elem.find("non_linear_parameters")
    helper = reconstruct_utils.ReconstructHelper_from_xml(helper_elem)

    return SingleCameraCalibration(cam_id=cam_id,
                                   Pmat=pmat,
                                   res=res,
                                   scale_factor=scale_factor,
                                   helper=helper)

def SingleCameraCalibration_from_basic_pmat(pmat,**kw):
    M = numpy.asarray(pmat)
    cam_center = pmat2cam_center(M)

    intrinsic_parameters, cam_rotation = my_rq(M[:,:3])
    #intrinsic_parameters = intrinsic_parameters/intrinsic_parameters[2,2] # normalize
    if abs(intrinsic_parameters[2,2]-1.0)>intrinsic_normalized_eps:
        raise ValueError('expected last row/col of intrinsic parameter matrix to be unity')

    # (K = intrinsic parameters)

    #cam_translation = numpy.dot( -cam_rotation, cam_center )
    #extrinsic_parameters = numpy.concatenate( (cam_rotation, cam_translation), axis=1 )

    #mean_focal_length = (intrinsic_parameters[0,0]+intrinsic_parameters[1,1])/2.0
    #center = intrinsic_parameters[0,2], intrinsic_parameters[1,2]

    #focalLength, center = compute_stuff_from_cal_matrix(cal)

    fc1 = intrinsic_parameters[0,0]
    cc1 = intrinsic_parameters[0,2]
    fc2 = intrinsic_parameters[1,1]
    cc2 = intrinsic_parameters[1,2]

    helper = reconstruct_utils.ReconstructHelper(fc1,fc2, # focal length
                                                 cc1,cc2, # image center
                                                 0,0, # radial distortion
                                                 0,0) # tangential distortion
    return SingleCameraCalibration(Pmat=M,
                                   helper=helper,
                                   **kw)

def pretty_dump(e, ind=''):
    # from http://www.devx.com/opensource/Article/33153/0/page/4

    # start with indentation
    s = ind
    # put tag (don't close it just yet)
    s += '<' + e.tag
    # add all attributes
    for (name, value) in e.items():
        s += ' ' + name + '=' + "'%s'" % value
    # if there is text close start tag, add the text and add an end tag
    if e.text and e.text.strip():
        s += '>' + e.text + '</' + e.tag + '>'
    else:
        # if there are children...
        if len(e) > 0:
            # close start tag
            s += '>'
            # add every child in its own line indented
            for child in e:
                s += '\n' + pretty_dump(child, ind + '  ')
            # add closing tag in a new line
            s += '\n' + ind + '</' + e.tag + '>'
        else:
            # no text and no children, just close the starting tag
            s += ' />'
    return s

def Reconstructor_from_xml(elem):
    assert ET.iselement(elem)
    assert elem.tag == "multi_camera_reconstructor"
    sccs = []
    minimum_eccentricity = None
    for child in elem:
        if child.tag == "single_camera_calibration":
            scc = SingleCameraCalibration_from_xml(child)
            sccs.append( scc )
        elif child.tag == 'minimum_eccentricity':
            minimum_eccentricity = float(child.text)
        else:
            raise ValueError('unknown tag: %s'%child.tag)
    return Reconstructor(sccs,minimum_eccentricity=minimum_eccentricity)

class Reconstructor:
    """A complete calibration for all cameras in a flydra setup

    Parameters
    ==========
    cal_source : {string, list, open pytables file instance}

      The source of the calibration. Can be a string specifying the
      path of a directory output by MultiCamSelfCal, a string
      specifying an .xml calibration file path, a string specifying a
      pytables file, a list of :class:`SingleCameraCalibration`
      instances, or an open pytables file object.

    do_normalize_pmat : boolean
      Whether the pmat is normalized such that the intrinsic parameters
      are in the expected form
    minimum_eccentricity : float, optional
      Minimum eccentricity (ratio of long to short axis of an
      ellipse) of 2D detected object required to use detected
      orientation for performing 3D body orientation estimation.

    """
    def __init__(self,
                 cal_source = None,
                 do_normalize_pmat=True,
                 minimum_eccentricity=None,
                 ):
        self.cal_source = cal_source

        if isinstance(self.cal_source,str) or isinstance(self.cal_source,unicode):
            if not self.cal_source.endswith('h5'):
                if os.path.isdir(self.cal_source):
                    self.cal_source_type = 'normal files'
                else:
                    self.cal_source_type = 'xml file'
            else:
                self.cal_source_type = 'pytables filename'
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

        close_cal_source = False
        if self.cal_source_type == 'pytables filename':
            import tables as PT # PyTables
            use_cal_source = PT.openFile(self.cal_source,mode='r')
            close_cal_source = True
            self.cal_source_type = 'pytables'
        else:
            use_cal_source = self.cal_source

        if self.cal_source_type == 'normal files':
            fd = open(os.path.join(use_cal_source,'camera_order.txt'),'r')
            cam_ids = fd.read().split('\n')
            fd.close()
            if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
        elif self.cal_source_type == 'xml file':
            root = ET.parse(use_cal_source).getroot()
            if root.tag == "multi_camera_reconstructor":
                next_self = Reconstructor_from_xml(root)
            else:
                r_node = root.find("multi_camera_reconstructor")
                if r_node is None:
                    raise ValueError(
                        'XML file does not contain reconstructor node')
            cam_ids = next_self.cam_ids
        elif self.cal_source_type == 'pytables':
            import tables as PT # PyTables
            assert type(use_cal_source)==PT.File
            results = use_cal_source
            nodes = results.root.calibration.pmat._f_listNodes()
            cam_ids = []
            for node in nodes:
                cam_ids.append( node.name )
        elif self.cal_source_type=='SingleCameraCalibration instances':
            cam_ids = [scci.cam_id for scci in use_cal_source]
        else:
            raise ValueError(
                "unknown cal_source_type '%s'"%self.cal_source_type)

        if minimum_eccentricity is None:
            self.minimum_eccentricity = None

            # not specified in call to constructor
            if self.cal_source_type == 'pytables':
                # load from file
                mi_col = results.root.calibration.additional_info[:]['minimum_eccentricity']
                assert len(mi_col)==1
                self.minimum_eccentricity = mi_col[0]
            elif self.cal_source_type == 'normal files':
                min_e_fname = os.path.join(use_cal_source,'minimum_eccentricity.txt')
                if os.path.exists(min_e_fname):
                    fd = open(min_e_fname,'r')
                    self.minimum_eccentricity = float( fd.read().strip() )
                    fd.close()
            elif self.cal_source_type == 'xml file':
                self.minimum_eccentricity = next_self.minimum_eccentricity
            if self.minimum_eccentricity is None:
                # use default
                if int(os.environ.get('FORCE_MINIMUM_ECCENTRICITY','0')):
                    raise ValueError('minimum_eccentricity cannot be default')
                else:
                    warnings.warn('No minimum eccentricity specified, using default')
                    self.minimum_eccentricity=DEFAULT_MINIMUM_ECCENTRICITY
        else:
            # use the value that was passed in to constructor
            self.minimum_eccentricity=minimum_eccentricity

        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Res = {}
        self._helper = {}

        # values for converting to meters
        self._known_units2scale_factor = {
            'millimeters':1e-3,
            'meters':1.0,
            }

        self._scale_factor2known_units = {}
        for tmp_unit, tmp_scale_factor in self._known_units2scale_factor.iteritems():
            self._scale_factor2known_units[tmp_scale_factor] = tmp_unit

        if self.cal_source_type == 'normal files':
            res_fd = open(os.path.join(use_cal_source,'Res.dat'),'r')
            for i, cam_id in enumerate(cam_ids):
                fname = 'camera%d.Pmat.cal'%(i+1)
                pmat = load_ascii_matrix(opj(use_cal_source,fname)) # 3 rows x 4 columns
                if do_normalize_pmat:
                    pmat_orig = pmat
                    pmat = normalize_pmat(pmat)
##                    if not numpy.allclose(pmat_orig,pmat):
##                        assert numpy.allclose(pmat2cam_center(pmat_orig),pmat2cam_center(pmat))
##                        #print 'normalized pmat, but camera center should  changed for %s'%cam_id
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = map(int,res_fd.readline().split())
            res_fd.close()

            # load non linear parameters
            rad_files = os.path.join(use_cal_source,'*.rad')
            for cam_id_enum, cam_id in enumerate(cam_ids):
                filename = os.path.join(use_cal_source,
                                        'basename%d.rad'%(cam_id_enum+1,))
                if not os.path.exists(filename):
                    if len(rad_files):
                        raise RuntimeError(
                            '.rad files present but none named "%s"'%filename)
                    warnings.warn('no non-linear data (e.g. radial distortion) '
                                  'in calibration for %s'%cam_id)
                    self._helper[cam_id] = SingleCameraCalibration_from_basic_pmat(
                        self.Pmat[cam_id],
                        cam_id=cam_id,
                        res=self.Res[cam_id]).helper
                    continue

                self._helper[cam_id] = reconstruct_utils.make_ReconstructHelper_from_rad_file(filename)

            filename = os.path.join(use_cal_source,'calibration_units.txt')
            if os.path.exists(filename):
                fd = file(filename,'r')
                value = fd.read()
                fd.close()
                value = value.strip()
            else:
                warnings.warn('Assuming scale_factor units are millimeters in '
                              '%s: file %s does not exist'%(
                    __file__,filename))
                value = 'millimeters'

            if value in self._known_units2scale_factor:
                self.scale_factor = self._known_units2scale_factor[value]
            else:
                raise ValueError('Unknown unit "%s"'%value)


        elif self.cal_source_type == 'pytables':
            scale_factors = []
            for cam_id in cam_ids:
                pmat = nx.array(results.root.calibration.pmat.__getattr__(cam_id))
                res = tuple(results.root.calibration.resolution.__getattr__(cam_id))
                K = nx.array(results.root.calibration.intrinsic_linear.__getattr__(cam_id))
                nlparams = tuple(results.root.calibration.intrinsic_nonlinear.__getattr__(cam_id))
                if hasattr(results.root.calibration,'scale_factor2meters'):
                    sf_array = numpy.array(results.root.calibration.scale_factor2meters.__getattr__(cam_id))
                    scale_factors.append(sf_array[0])
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = res
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    K[0,0], K[1,1], K[0,2], K[1,2],
                    nlparams[0], nlparams[1], nlparams[2], nlparams[3])

            unique_scale_factors = list(set(scale_factors))
            if len(unique_scale_factors)==0:
                print 'Assuming scale_factor units are millimeters in pytables',__file__
                self.scale_factor = self._known_units2scale_factor['millimeters']
            elif len(unique_scale_factors)==1:
                self.scale_factor = unique_scale_factors[0]
            else:
                raise NotImplementedError('cannot handle case where each camera has a different scale factor')

        elif self.cal_source_type=='SingleCameraCalibration instances':
            # find instance
            scale_factors = []
            for cam_id in cam_ids:
                for scci in use_cal_source:
                    if scci.cam_id==cam_id:
                        break
                self.Pmat[cam_id] = scci.Pmat
                self.Res[cam_id] = scci.res
                self._helper[cam_id] = scci.helper
                if scci.scale_factor is not None:
                    scale_factors.append( scci.scale_factor )
            unique_scale_factors = list(set(scale_factors))
            if len(unique_scale_factors)==0:
                print 'Assuming scale_factor units are millimeters in SingleCameraCalibration instances (%s)'%(
                    __file__,)
                self.scale_factor = self._known_units2scale_factor['millimeters']
            elif len(unique_scale_factors)==1:
                self.scale_factor = unique_scale_factors[0]
            else:
                raise NotImplementedError('cannot handle case where each camera has a different scale factor')
        elif self.cal_source_type=='xml file':
            self.scale_factor = next_self.scale_factor
            self.Pmat = next_self.Pmat
            self.Res = next_self.Res
            self._helper =  next_self._helper

        self.pmat_inv = {}
        self.pinhole_model_with_jacobian = {}
        for cam_id in cam_ids:
            # For speed reasons, make sure self.Pmat has only numpy arrays.
            self.Pmat[cam_id] = numpy.array(self.Pmat[cam_id])

            self.pmat_inv[cam_id] = numpy.linalg.pinv(self.Pmat[cam_id])
            self.pinhole_model_with_jacobian[cam_id] = flydra.pmat_jacobian.PinholeCameraModelWithJacobian(self.Pmat[cam_id])

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
        # fill self._cam_centers_cache
        self._cam_centers_cache = {}
        for cam_id in self.cam_ids:
            self._cam_centers_cache[cam_id] = self.get_camera_center(cam_id)[:,0] # make rank-1

        if close_cal_source:
            use_cal_source.close()

    def get_aligned_copy(self, M):
        orig_sccs = [self.get_SingleCameraCalibration(cam_id)
                     for cam_id in self.cam_ids]
        aligned_sccs = [scc.get_aligned_copy(M) for scc in orig_sccs]
        return Reconstructor(aligned_sccs,
                             minimum_eccentricity=self.minimum_eccentricity)

    def set_minimum_eccentricity(self,minimum_eccentricity):
        self.minimum_eccentricity = minimum_eccentricity

    def __ne__(self,other):
        return not (self==other)

    def __eq__(self,other):
        orig_sccs = [self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids]
        other_sccs = [other.get_SingleCameraCalibration(cam_id) for cam_id in other.cam_ids]
        eq = True
        for my_scc,other_scc in zip(orig_sccs,other_sccs):
            if my_scc != other_scc:
                if int(os.environ.get('DEBUG_EQ','0')):
                    for attr in ['cam_id','Pmat','res','helper']:
                        print
                        print 'attr',attr
                        print 'my_scc',getattr(my_scc,attr)
                        print 'other_scc',getattr(other_scc,attr)
                eq = False
                break
        return eq

    def get_3D_plane_and_ray(self, cam_id, *args, **kwargs):
        s = self.get_SingleCameraCalibration(cam_id)
        return s.get_3D_plane_and_ray(*args,**kwargs)

    def get_extrinsic_parameter_matrix(self,cam_id):
        scc = self.get_SingleCameraCalibration(cam_id)
        return scc.get_extrinsic_parameter_matrix()

    def get_scaled(self,scale_factor=None):
        """change units (e.g. from mm to meters)

        Note: some of the data structures are shared with the unscaled original
        """
        if scale_factor is None:
            scale_factor = self.get_scale_factor()

        # get original calibration
        orig_sccs = [self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids]
        scaled_sccs = [scc.get_scaled(scale_factor) for scc in orig_sccs]
        return Reconstructor(scaled_sccs,minimum_eccentricity=self.minimum_eccentricity)

    def get_cam_ids(self):
        return self.cam_ids

    def save_to_files_in_new_directory(self, new_dirname):
        if os.path.exists(new_dirname):
            raise RuntimeError('directory "%s" already exists'%new_dirname)
        os.mkdir(new_dirname)

        fd = open(os.path.join(new_dirname,'camera_order.txt'),'w')
        for cam_id in self.cam_ids:
            fd.write(cam_id+'\n')
        fd.close()

        fd = open(os.path.join(new_dirname,'minimum_eccentricity.txt'),'w')
        fd.write(repr(self.minimum_eccentricity)+'\n')
        fd.close()

        res_fd = open(os.path.join(new_dirname,'Res.dat'),'w')
        for cam_id in self.cam_ids:
            res_fd.write( ' '.join(map(str,self.Res[cam_id])) + '\n' )
        res_fd.close()

        for i, cam_id in enumerate(self.cam_ids):
            fname = 'camera%d.Pmat.cal'%(i+1)
            pmat_fd = open(os.path.join(new_dirname,fname),'w')
            save_ascii_matrix(self.Pmat[cam_id],pmat_fd)
            pmat_fd.close()

        # non linear parameters
        for i, cam_id in enumerate(self.cam_ids):
            fname = 'basename%d.rad'%(i+1)
            self._helper[cam_id].save_to_rad_file( os.path.join(new_dirname,fname) )

        fd = open(os.path.join(new_dirname,'calibration_units.txt'),mode='w')
        fd.write(self.get_calibration_unit()+'\n')
        fd.close()

    def save_to_xml_filename(self, xml_filename):
        root = ET.Element("root")
        self.add_element(root)
        child = root[0]
        result = pretty_dump(child,ind='  ')
        fd = open(xml_filename,mode='w')
        fd.write(result)
        fd.close()

    def as_obj_for_json(self):
        result = dict(cameras=[],
                      minimum_eccentricity=self.minimum_eccentricity)
        for cam_id in self.cam_ids:
            scc = self.get_SingleCameraCalibration(cam_id)
            result['cameras'].append(scc.as_obj_for_json())
        return result

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

        scale_group = h5file.createGroup(cal_group,'scale_factor2meters')
        for cam_id in cam_ids:
            h5file.createArray(scale_group, cam_id,
                               pytables_filt([self.get_scale_factor()]))

        h5additional_info = ct(cal_group,'additional_info', AdditionalInfo,
                               '')
        row = h5additional_info.row
        row['cal_source_type'] = self.cal_source_type
        if isinstance(self.cal_source,list):
            row['cal_source'] = '(originally was list - not saved here)'
        else:
            if not isinstance( self.cal_source,PT.File):
                row['cal_source'] = self.cal_source
            else:
                row['cal_source'] = self.cal_source.filename
        row['minimum_eccentricity'] = self.minimum_eccentricity
        row.append()
        h5additional_info.flush()

    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def get_pmat_inv(self, cam_id):
        return self.pmat_inv[cam_id]

    def get_pinhole_model_with_jacobian(self, cam_id):
        return self.pinhole_model_with_jacobian[cam_id]

    def get_camera_center(self, cam_id):
        # should be called get_cam_center?
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

    def find3d(self, cam_ids_and_points2d, return_X_coords = True,
               return_line_coords = True, orientation_consensus = 0):
        """Find 3D coordinate using all data given

        Implements a linear triangulation method to find a 3D
        point. For example, see Hartley & Zisserman section 12.2
        (p.312). Also, 12.8 for intersecting lines.

        Note: For a function which does hyptothesis testing to
        selectively choose 2D to incorporate, see
        hypothesis_testing_algorithm__find_best_3d() in
        reconstruct_utils.

        The data should already be undistorted before passing to this
        function.

        """
        svd = scipy.linalg.svd
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)

        # Construct matrices
        A=[]
        P=[]
        for m, (cam_id,value_tuple) in enumerate(cam_ids_and_points2d):
            if len(value_tuple)==2:
                # only point information ( no line )
                x,y = value_tuple
                have_line_coords = False
                if return_line_coords:
                    raise ValueError('requesting 3D line coordinates, but no 2D line coordinates given')
            else:
                # get shape information from each view of a blob:
                x,y,area,slope,eccentricity, p1,p2,p3,p4 = value_tuple
                have_line_coords = True
            if return_X_coords:
                Pmat = self.Pmat[cam_id] # Pmat is 3 rows x 4 columns
                row2 = Pmat[2,:]
                A.append( x*row2 - Pmat[0,:] )
                A.append( y*row2 - Pmat[1,:] )

            if return_line_coords and have_line_coords:
                if eccentricity > self.minimum_eccentricity: # require a bit of elongation
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
            if orientation_consensus < 0:
                orientation_consensus = (len(cam_ids_and_points2d) +
                                         orientation_consensus)

            if orientation_consensus != 0 and len(P) > orientation_consensus:
                # test for better fit with fewer orientations
                allps = [np.array(ps) for ps in
                         setOfSubsets(P) if len(ps)==orientation_consensus]
                all_Lcoords = []
                for ps in allps:
                    all_Lcoords.append( intersect_planes_to_find_line(ps) )
                N = len(all_Lcoords)
                distance_matrix = np.zeros((N,N))
                for i in range(N):
                    for j in range(i+1,N):
                        d = line_direction_distance(
                            all_Lcoords[i], all_Lcoords[j] )
                        # distances are symmetric
                        distance_matrix[i,j] = d
                        distance_matrix[j,i] = d
                sum_distances = np.sum( distance_matrix, axis=0 )
                closest_to_neighbors_idx = np.argmin( sum_distances )
                Lcoords = all_Lcoords[closest_to_neighbors_idx]
            else:
                P = nx.asarray(P)
                # Calculate best line
                Lcoords = intersect_planes_to_find_line(P)

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
        """see also SingleCameraCalibration.get_example_3d_point_creating_image_point()"""
        return nx.dot(self.pmat_inv[cam_id], as_column(x))

    def get_projected_line_from_2d(self,cam_id,xy):
        """project undistorted points into pluecker line

        see also SingleCameraCalibration.get_example_3d_point_creating_image_point()"""
        # XXX Could use nullspace method of back-projection of lines (that pass through camera center, HZ section 8.1)
        # image of 2d point in 3d space (on optical ray)
        XY = self.find3d_single_cam(cam_id,(xy[0],xy[1],1.0))
        XY = XY[:3,0]/XY[3] # convert to rank1
        C = self._cam_centers_cache[cam_id]
        return pluecker_from_verts(XY,C)

    def get_scale_factor(self):
        return self.scale_factor

    def get_calibration_unit(self):
        return self._scale_factor2known_units[self.scale_factor]

    def get_SingleCameraCalibration(self,cam_id):
        return SingleCameraCalibration(cam_id=cam_id,
                                       Pmat=self.Pmat[cam_id],
                                       res=self.Res[cam_id],
                                       helper=self._helper[cam_id],
                                       scale_factor=self.scale_factor,
                                       )

    def get_distorted_line_segments(self, cam_id, line3d ):
        dummy = [0,0,0] # dummy 3D coordinate

        # project 3d line into projected 2d line
        dummy2d, proj = self.find2d(cam_id, dummy, line3d)

        # now distort 2d line into 2d line segments

        # calculate undistorted 2d line segments

        # line at x = -100
        l = numpy.array([1,0,100])

        # line at x = 1000
        r = numpy.array([1,0,-1000])


        lh = numpy.cross(proj,l)
        rh = numpy.cross(proj,r)

        if lh[2]==0 or rh[2]==0:
            if 1:
                raise NotImplementedError('cannot deal with exactly vertical lines')
            b = numpy.array([0,1,100])
            t = numpy.array([0,1,-1000])
            bh = numpy.cross(proj,b)
            th = numpy.cross(proj,t)

        x0 = lh[0]/lh[2]
        y0 = lh[1]/lh[2]

        x1 = rh[0]/rh[2]
        y1 = rh[1]/rh[2]

        dy = y1-y0
        dx = x1-x0
        n_pts = 10000
        frac = numpy.linspace(0,1,n_pts)
        xs = x0 + frac*dx
        ys = y0 + frac*dy

        # distort 2d segments
        xs_d = []
        ys_d = []
        for xy in zip(xs,ys):
            x_distorted, y_distorted = self.distort(cam_id,xy)
            if -100<=x_distorted<=800 and -100<=y_distorted<=800:
                xs_d.append( x_distorted )
                ys_d.append( y_distorted )
        return xs_d, ys_d

    def add_element(self,parent):
        """add self as XML element to parent"""
        assert ET.iselement(parent)
        elem = ET.SubElement(parent,"multi_camera_reconstructor")
        for cam_id in self.cam_ids:
            scc = self.get_SingleCameraCalibration(cam_id)
            scc.add_element(elem)
        if 1:
            me_elem = ET.SubElement(elem,"minimum_eccentricity")
            me_elem.text = repr(self.minimum_eccentricity)

def test_sba():
    import flydra.generate_fake_calibration as gfc
    recon = gfc.generate_calibration()
    for cam_id in recon.cam_ids:
        scc = recon.get_SingleCameraCalibration(cam_id)
        sba_line = scc.get_sba_format_line()

def test():
    import flydra.generate_fake_calibration as gfc
    recon = gfc.generate_calibration()
    Xs = [[0,0,0],
         [1,2,3],
         [0.5,-.2,.0004]]
    for X in Xs:
        print 'X',X
        Xh = numpy.concatenate((X,[1]))
        print 'Xh',Xh
        for cam_id in recon.cam_ids:
            pmat = recon.get_pmat(cam_id)
            print 'numpy.dot(pmat,Xh)',numpy.dot(pmat,Xh)
            x = recon.find2d(cam_id,X)
            print cam_id,'pmat ========='
            print pmat
            Rt = recon.get_extrinsic_parameter_matrix(cam_id)
            print 'Rt'
            print Rt
            print 'numpy.dot(Rt,Xh)'
            print numpy.dot(Rt,Xh)
            K,R = recon.get_SingleCameraCalibration(cam_id).get_KR()
            print 'K'
            print K
            print 'numpy.dot(K,numpy.dot(Rt,Xh))'
            print numpy.dot(K,numpy.dot(Rt,Xh))
            print 'numpy.dot(pmat,Xh)'
            print numpy.dot(pmat,Xh)
            print 'x',x
        print
        break
    print '-='*30
    for cam_id in recon.cam_ids:
        line = recon.get_projected_line_from_2d( cam_id, (800,600) )
        print cam_id,'line',line
        scc = recon.get_SingleCameraCalibration(cam_id)
        print 'center',scc.get_cam_center()
        oax = scc.get_optical_axis()
        print 'oax',oax
        print 'oax HZ',oax.to_hz()
        print 'closest',oax.closest()
        print

def align_calibration():
     usage = """%prog [options]

     Requires an alignment file. This can either be a raw file, or the
     list of camera locations.

     A raw alignment file (specified with --align-raw) should have
     contents like the following, where s is scale, R is a 3x3
     rotation matrix, and t is a 3 vector specifying translation::

       s=0.89999324180965479
       R=[[0.99793608705515335, 0.041419147873301365, -0.04907158385969549],
          [-0.044244064258451607, 0.99733841974169912, -0.057952905751338504],
          [-0.04654061592784884, -0.060004422308518594, -0.99711255150683809]]
       t=[-0.77477404833588825, 0.32807381821090476, 5.6937474990726518]

     A list of camera locations (specified with --align-cams) should
     look like the following for 2 cameras at (1,2,3) and (4,5,6)::

       1 2 3
       4 5 6

     """

     parser = OptionParser(usage)

     parser.add_option("--orig-reconstructor", type='string',
                       help="calibration/reconstructor path (required)")

     parser.add_option("--dest-dir", type='string',
                       help="output calibration/reconstructor path")

     parser.add_option("--align-raw", type='string',
                       help="raw alignment file path")

     parser.add_option("--align-cams", type='string',
                       help="new camera locations alignment file path")

     parser.add_option("--scaled", action='store_true',
                       default=False,
                       help=('Set if the alignment should be '
                             'applied to the scaled calibration'),
                       )

     (options, args) = parser.parse_args()

     if options.orig_reconstructor is None:
         raise ValueError('--orig-reconstructor must be specified')

     if options.align_raw is None and options.align_cams is None:
         raise ValueError(
             'either --align-raw or --align-cams must be specified')

     if options.align_raw is not None and options.align_cams is not None:
         raise ValueError(
             'only one of --align-raw and --align-cams can be specified')

     src=options.orig_reconstructor
     origR = Reconstructor(cal_source=src)

     if options.dest_dir is None:
         dst = src+'.aligned'
     else:
         dst = options.dest_dir
     if os.path.exists(dst):
         raise RuntimeError('destination %s exists'%dst)

     if options.scaled:
         srcR = origR.get_scaled()
     else:
         srcR = origR

     import flydra.align as align
     if options.align_raw is not None:
         mylocals = {}
         myglobals = {}
         execfile(options.align_raw,myglobals,mylocals)
         s = mylocals['s']
         R = np.array(mylocals['R'])
         t = np.array(mylocals['t'])
     else:
         assert options.align_cams is not None
         cam_ids = srcR.get_cam_ids()
         ccs = [srcR.get_camera_center(cam_id)[:,0] for cam_id in cam_ids]
         print 'ccs',ccs
         orig_cam_centers = np.array(ccs).T
         print 'orig_cam_centers',orig_cam_centers.T
         new_cam_centers = load_ascii_matrix(options.align_cams).T
         print 'new_cam_centers',new_cam_centers.T
         s,R,t = align.estsimt(orig_cam_centers,new_cam_centers)

     print 's',s
     print 'R',R
     print 't',t

     M = align.build_xfrom(s,R,t)

     alignedR = srcR.get_aligned_copy(M)
     alignedR.save_to_files_in_new_directory(dst)

if __name__=='__main__':
    test()
