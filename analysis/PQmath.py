import math
import cgtypes
from numarray.ieeespecial import nan
import numarray as nx

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

def rotate_velocity_by_orientation(vel,orient):
    # this is backwards from a normal quaternion rotation
    return orient.inverse()*vel*orient

def is_unit_vector(U):
    V = nx.array(U)
    if len(V.shape)==1:
        V = V[nx.NewAxis,:]
    V = V**2
    mag = nx.sqrt(nx.sum(V,axis=1))
    return nx.sum(nx.abs(mag-1.0)) < 1e-15

def world2body( U, roll_angle = 0 ):
    """convert world coordinates to body-relative coordinates

    inputs:
    
    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    M (3x3 matrix), get world coords via nx.dot(M,X)
    """
    assert is_unit_vector(U)
    # notation from Wagner, 1986a, Appendix 1
    Bxy = math.atan2( U[1], U[0] ) # heading angle
    Bxz = math.asin( U[2] ) # pitch (body) angle
    Byz = roll_angle # roll angle
    cos = math.cos
    sin = math.sin
    M = nx.array( (( cos(Bxy)*cos(-Bxz), sin(Bxy)*cos(-Bxz), -sin(-Bxz)),
                   ( cos(Bxy)*sin(-Bxz)*sin(Byz) - sin(Bxy)*cos(Byz), sin(Bxy)*sin(-Bxz)*sin(Byz) + cos(Bxy)*cos(Byz), cos(-Bxz)*sin(Byz)),
                   ( cos(Bxy)*sin(-Bxz)*cos(Byz) + sin(Bxy)*sin(Byz), sin(Bxy)*sin(-Bxz)*cos(Byz) - cos(Bxy)*sin(Byz), cos(-Bxz)*cos(Byz))))
    return M

def cross(vec1,vec2):
    return ( vec1[1]*vec2[2] - vec1[2]*vec2[1],
             vec1[2]*vec2[0] - vec1[0]*vec2[2],
             vec1[0]*vec2[1] - vec1[1]*vec2[0] )

def make_quat(angle_radians, axis_of_rotation):
    half_angle = angle_radians/2.0
    a = math.cos(half_angle)
    b,c,d = math.sin(half_angle)*norm_vec(axis_of_rotation)
    return cgtypes.quat(a,b,c,d)

def euler_to_orientation( yaw=0.0, pitch = 0.0 ):
    """
>>> euler_to_orientation(0,0)
(1.0, 0.0, 0.0)

>>> euler_to_orientation(math.pi,0)
(-1.0, 1.2246063538223773e-16, 0.0)

>>> euler_to_orientation(math.pi,math.pi/2)
(-6.1230317691118863e-17, 7.4983036091106872e-33, 1.0)

>>> euler_to_orientation(math.pi,-math.pi/2)
(-6.1230317691118863e-17, 7.4983036091106872e-33, -1.0)

>>> euler_to_orientation(math.pi,-math.pi/4)
(-0.70710678118654757, 8.6592745707193554e-17, -0.70710678118654746)

>>> euler_to_orientation(math.pi/2,-math.pi/4)
(4.3296372853596777e-17, 0.70710678118654757, -0.70710678118654746)
"""
    z = math.sin( pitch )
    xyr = math.cos( pitch )
    y = xyr*math.sin( yaw )
    x = xyr*math.cos( yaw )
    return x, y, z

def euler_to_quat(roll=0.0,pitch=0.0,yaw=0.0):
    heading = yaw
    attitude = pitch
    bank = roll
    c1 = math.cos(heading/2);
    s1 = math.sin(heading/2);
    c2 = math.cos(attitude/2);
    s2 = math.sin(attitude/2);
    c3 = math.cos(bank/2);
    s3 = math.sin(bank/2);
    c1c2 = c1*c2;
    s1s2 = s1*s2;
    w =c1c2*c3 - s1s2*s3;
    x =c1c2*s3 + s1s2*c3;
    y =s1*c2*c3 + c1*s2*s3;
    z =c1*s2*c3 - s1*c2*s3;
    z, y = y, -z
    return cgtypes.quat(w,x,y,z)
    
##    a,b,c=roll,-pitch,yaw
##    Qx = cgtypes.quat( math.cos(a/2.0), math.sin(a/2.0), 0, 0 )
##    Qy = cgtypes.quat( math.cos(b/2.0), 0, math.sin(b/2.0), 0 )
##    Qz = cgtypes.quat( math.cos(c/2.0), 0, 0, math.sin(c/2.0) )
##    return Qx*Qy*Qz

def orientation_to_euler( U ):
    """
convert orientation to euler angles (in radians)

results are yaw, pitch (no roll is provided)

>>> orientation_to_euler( (1, 0, 0) )
(0.0, 0.0)

>>> orientation_to_euler( (0, 1, 0) )
(1.5707963267948966, 0.0)

>>> orientation_to_euler( (-1, 0, 0) )
(3.1415926535897931, 0.0)

>>> orientation_to_euler( (0, -1, 0) )
(-1.5707963267948966, 0.0)

>>> orientation_to_euler( (0, 0, 1) )
(0.0, 1.5707963267948966)

>>> orientation_to_euler( (0, 0, -1) )
(0.0, -1.5707963267948966)

>>> orientation_to_euler( (0,0,0) ) # This is not a unit vector.
Traceback (most recent call last):
    ...
AssertionError

>>> r1=math.sqrt(2)/2

>>> math.pi/4
0.78539816339744828

>>> orientation_to_euler( (r1,0,r1) )
(0.0, 0.78539816339744828)

>>> orientation_to_euler( (r1,r1,0) )
(0.78539816339744828, 0.0)

"""

    if str(U[0]) == 'nan':
        return (nan,nan)
    assert is_unit_vector(U)
        
    yaw = math.atan2( U[1],U[0] )
    xy_r = math.sqrt(U[0]**2+U[1]**2)
    pitch = math.atan2( U[2], xy_r )
    return yaw, pitch
    
def orientation_to_quat( U, roll_angle=0 ):
    """convert world coordinates to body-relative coordinates

    inputs:
    
    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    quaternion (4 tuple)
    """
    if roll_angle != 0:
        # I presume this has an effect on the b component of the quaternion
        raise NotImplementedError('')
    if str(U[0]) == 'nan':
        return cgtypes.quat((nan,nan,nan,nan))
    yaw, pitch = orientation_to_euler( U )
    return euler_to_quat(yaw=yaw, pitch=pitch, roll=roll_angle)

def quat_to_orient(S3):
    """returns x, y, z for unit quaternions"""
    u = cgtypes.quat(0,1,0,0)
    if type(S3)()==QuatSeq(): # XXX isinstance(S3,QuatSeq)
        V = [q*u*q.inverse() for q in S3]
        return nx.array([(v.x, v.y, v.z) for v in V])
    else:
        V=S3*u*S3.inverse()
        return V.x, V.y, V.z

def quat_to_euler(q):
    """returns yaw, pitch, roll

    at singularities (north and south pole), assumes roll = 0
    """
    eps=1e-14
    qw = q.w; qx = q.x; qy = q.z; qz = -q.y
    pitch_y = 2*qx*qy + 2*qz*qw
    pitch = math.asin(pitch_y)
    if pitch_y > (1.0-eps): # north pole
        yaw = 2*math.atan2( qx, qw)
        roll = 0.0
    elif pitch_y < -(1.0-eps): # south pole
        yaw = -2*math.atan2( qx, qw)
        roll = 0.0
    else:
        yaw = math.atan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
        roll = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    return yaw, pitch, roll

def quat_to_absroll(q):
    qw = q.w; qx = q.x; qy = q.z; qz = -q.y
    roll = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    return abs(roll)

class ObjectiveFunctionPosition:
    """methods from Kim, Hsieh, Wang, Wang, Fang, Woo"""
    def __init__(self, p, h, alpha, no_distance_penalty_idxs=None):
        """

        no_distance_penalty_idxs -- indexes into p where fit data
            should not be penalized (probably because 'original data'
            was interpolated and is of no value)
        
        """
        self.p = p
        self.h = h
        self.alpha = alpha
        
        self.p_err_weights = nx.ones( p.shape )
        if no_distance_penalty_idxs is not None:
            for i in no_distance_penalty_idxs:
                self.p_err_weights[i] = 0
    def _getDistance(self, ps):
        return nx.sum(nx.sum(self.p_err_weights*((self.p - ps)**2), axis=1))
    def _getEnergy(self, ps):
        d2p = (ps[2:] - 2*ps[1:-1] + ps[:-2]) / (self.h**2)
        return  nx.sum( nx.sum(d2p**2,axis=1))
    def eval(self,ps):
        D = self._getDistance(ps)
        E = self._getEnergy(ps)
        return D + self.alpha*E # eqn. 22
    def get_del_F(self,P):
        class PDfinder:
            def __init__(self,objective_function,P):
                self.objective_function = objective_function
                self.P = P
                self.F_P = self.objective_function.eval(P)
                self.ndims = P.shape[1]
            def eval_pd(self,i):
                # evaluate 3 values (perturbations in x,y,z directions)
                PERTURB = 1e-6 # perturbation amount (in meters)

                dFdP = []
                for j in range(self.ndims):
                    P_i_j = self.P[i,j]
                    # temporarily perturb P_i
                    self.P[i,j] = P_i_j+PERTURB
                    F_Pj = self.objective_function.eval(P)
                    self.P[i,j] = P_i_j
                    
                    dFdP.append( (F_Pj-self.F_P)/PERTURB )
                return dFdP
        _pd_finder = PDfinder(self,P)
        PDs = nx.array([ _pd_finder.eval_pd(i) for i in range(len(P)) ])
        return PDs

def smooth_position( P, delta_t, alpha, lmbda, eps ):
    """smooth a sequence of positions

    see the following citation for details:
    
    "Noise Smoothing for VR Equipment in Quaternions," C.C. Hsieh,
    Y.C. Fang, M.E. Wang, C.K. Wang, M.J. Kim, S.Y. Shin, and
    T.C. Woo, IIE Transactions, vol. 30, no. 7, pp. 581-587, 1998
    
    P are the positions as in an n x m array, where n is the number of
    data points and m is the number of dimensions in which the data is
    measured.

    inputs
    ------

    P       positions (m by n array, m = number of samples, n = dimensionality)
    delta_t temporal interval, in seconds, between samples
    alpha   relative importance of acceleration versus position
    lmbda   step size when descending gradient
    eps     termination criterion

    output
    ------

    Pstar   smoothed positions (same format as P)
    """
    h = delta_t
    Pstar = P
    err = 2*eps # cycle at least once
    of = ObjectiveFunctionPosition(P,h,alpha)
    while err > eps:
        del_F = of.get_del_F(Pstar)
        err = nx.sum( nx.sum( del_F**2 ) )
        Pstar = Pstar - lmbda*del_F
    return Pstar

class QuatSeq(list):
    def __abs__(self):
        return nx.array([ abs(q) for q in self ])
    def __add__(self,other):
        if isinstance(other,QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([ p+q for p,q in zip(self,other) ])
        else:
            raise ValueError('only know how to add QuatSeq with QuatSeq')
    def __sub__(self,other):
        if isinstance(other,QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([ p-q for p,q in zip(self,other) ])
        else:
            raise ValueError('only know how to subtract QuatSeq with QuatSeq')
    def __mul__(self,other):
        if isinstance(other,QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([ p*q for p,q in zip(self,other) ])
        elif isinstance(other,nx.NumArray):
            assert len(other.shape)==2
            if other.shape[1] == 3:
                other = nx.concatenate( (nx.zeros((other.shape[0],1)),
                                         other), axis=1 )
            assert other.shape[1] == 4
            other = QuatSeq([cgtypes.quat(o) for o in other])
            return self*other
        else:
            try:
                other = float(other)
            except (ValueError, TypeError):
                raise TypeError('only know how to multiply QuatSeq with QuatSeq, n*3 array or float')
            return QuatSeq([ p*other for p in self])
    def __div__(self,other):
        try:
            other = float(other)
        except ValueError:
            raise ValueError('only know how to divide QuatSeq with floats')
        return QuatSeq([ p/other for p in self])
    def __pow__(self,n):
        return QuatSeq([ q**n for q in self ])
    def __neg__(self):
        return QuatSeq([ -q for q in self ])
    def __str__(self):
        return 'QuatSeq('+list.__str__(self)+')'
    def __repr__(self):
        return 'QuatSeq('+list.__repr__(self)+')'
    def inverse(self):
        return QuatSeq([ q.inverse() for q in self ])
    def exp(self):
        return QuatSeq([ q.exp() for q in self ])
    def log(self):
        return QuatSeq([ q.log() for q in self ])
    def __getslice__(self,*args,**kw):
        return QuatSeq( list.__getslice__(self,*args,**kw) )
    def get_w(self):
        return nx.array([ q.w for q in self])
    w = property(get_w)
    def get_x(self):
        return nx.array([ q.x for q in self])
    x = property(get_x)
    def get_y(self):
        return nx.array([ q.y for q in self])
    y = property(get_y)
    def get_z(self):
        return nx.array([ q.z for q in self])
    z = property(get_z)

class ObjectiveFunctionQuats:
    """methods from Kim, Hsieh, Wang, Wang, Fang, Woo"""
    def __init__(self, q, h, beta, gamma):
        self.q = q
        self.q_inverse = self.q.inverse()
        self.h = h
        self.h2 = self.h**2
        self.beta = beta
        self.gamma = gamma
    def _getDistance(self, qs):
        return sum( abs(    (self.q_inverse * qs).log() )**2)
    def _getRoll(self, qs):
        return sum([quat_to_absroll(q) for q in qs])
    def _getEnergy(self, qs):
        omega_dot = ((qs[1:-1].inverse()*qs[2:]).log() -
                     (qs[:-2].inverse()*qs[1:-1]).log()) / self.h2
        return sum( abs( omega_dot )**2 )
    def eval(self,qs):
        D = self._getDistance(qs)
        E = self._getEnergy(qs)
        if self.gamma == 0:
            return D + self.beta*E # eqn. 23
        R = self._getRoll(qs)
        return D + self.beta*E + self.gamma*R # eqn. 23
    
    def get_del_G(self,Q):
        class PDfinder:
            """partial derivative finder"""
            def __init__(self,objective_function,Q):
                self.objective_function = objective_function
                self.Q = Q
                self.G_Q = self.objective_function.eval(Q) # G evaluated at Q
            def eval_pd(self,i):
                # evaluate 3 values (perturbations in x,y,z directions)
                PERTURB = 1e-6 # perturbation amount (must be less than sqrt(pi))
                #PERTURB = 1e-10 # perturbation amount (must be less than sqrt(pi))

                q_i = self.Q[i]
                q_i_inverse = q_i.inverse()
                
                qx = q_i*cgtypes.quat(0,PERTURB,0,0).exp()
                self.Q[i] = qx
                G_Qx = self.objective_function.eval(self.Q)
                dist_x = abs((q_i_inverse*qx).log())

                qy = q_i*cgtypes.quat(0,0,PERTURB,0).exp()
                self.Q[i] = qy
                G_Qy = self.objective_function.eval(self.Q)
                dist_y = abs((q_i_inverse*qy).log())

                qz = q_i*cgtypes.quat(0,0,0,PERTURB).exp()
                self.Q[i] = qz
                G_Qz = self.objective_function.eval(self.Q)
                dist_z = abs((q_i_inverse*qz).log())

                self.Q[i] = q_i

                qdir = cgtypes.quat(0,
                                    (G_Qx-self.G_Q)/dist_x,
                                    (G_Qy-self.G_Q)/dist_y,
                                    (G_Qz-self.G_Q)/dist_z)
                return qdir
        pd_finder = PDfinder(self,Q)
        del_G_Q = QuatSeq([ pd_finder.eval_pd(i) for i in range(len(Q)) ])
        return del_G_Q

def _test():
    # test math
    eps=1e-7
    yaws = list( nx.arange( -math.pi, math.pi, math.pi/16.0 ) )
    yaws.append( math.pi )
    pitches = list( nx.arange( -math.pi/2, math.pi/2, math.pi/16.0 ) )
    pitches.append( math.pi/2 )
    err_count = 0
    total_count = 0
    for yaw in yaws:
        for pitch in pitches:
            had_err = False
            # forward and backward test 1
            yaw2,pitch2 = orientation_to_euler(euler_to_orientation(yaw,pitch))
            if abs(yaw-yaw2)>eps or abs(pitch-pitch2)>eps:
                print 'orientation problem at',repr((yaw,pitch))
                had_err = True

            # forward and backward test 2
            yaw3, pitch3, roll3 = quat_to_euler( euler_to_quat( yaw=yaw, pitch=pitch ))
            if abs(yaw-yaw3)>eps or abs(pitch-pitch3)>eps:
                print 'quat problem at',repr((yaw,pitch))
                #print ' ',yaw,pitch,0.0
                print ' ',repr((yaw3, pitch3, roll3))
                print '    ',abs(yaw-yaw3)
                print '    ',abs(pitch-pitch3)
                print
                had_err = True

            # triangle test 1
            xyz1=euler_to_orientation(yaw,pitch)
            xyz2=quat_to_orient( euler_to_quat( yaw=yaw, pitch=pitch ))
            l2dist = math.sqrt(nx.sum((nx.array(xyz1)-nx.array(xyz2))**2))
            if l2dist > eps:
                print 'other problem at',repr((yaw,pitch))
                print ' ',xyz1
                print ' ',xyz2
                print
                had_err = True

            # triangle test 2
            yaw4, pitch4, roll4 = quat_to_euler(orientation_to_quat( xyz1 ))
            if abs(yaw-yaw4)>eps or abs(pitch-pitch4)>eps:
                print 'yet another problem at',repr((yaw,pitch))
                print
                had_err = True
            total_count += 1
            if had_err:
                err_count += 1
    print 'Error count: (%d of %d)'%(err_count,total_count)
                
    # do doctest
    import doctest, PQmath
    return doctest.testmod(PQmath)


if __name__ == "__main__":
    _test()
