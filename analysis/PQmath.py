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

def assert_unit_vector(U):
    chk = norm_vec(U)
    assert nx.sum(nx.abs(chk-U)) < 1e-15

def world2body( U, roll_angle = 0 ):
    """convert world coordinates to body-relative coordinates

    inputs:
    
    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    M (3x3 matrix), get world coords via nx.dot(M,X)
    """
    assert_unit_vector(U)
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

def euler_to_quat(roll=0.0,pitch=0.0,yaw=0.0):
    a,b,c=roll,-pitch,yaw
    Qx = cgtypes.quat( math.cos(a/2.0), math.sin(a/2.0), 0, 0 )
    Qy = cgtypes.quat( math.cos(b/2.0), 0, math.sin(b/2.0), 0 )
    Qz = cgtypes.quat( math.cos(c/2.0), 0, 0, math.sin(c/2.0) )
    return Qx*Qy*Qz

def orientation_to_quat( U, roll_angle=0 ):
    """convert world coordinates to body-relative coordinates

    inputs:
    
    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    quaternion (4 tuple)
    """
    if str(U[0]) == 'nan':
        return cgtypes.quat((nan,nan,nan,nan))
    try:
        assert_unit_vector(U)
    except:
        print 'ERROR: U is not unit (mag = %f)'%( math.sqrt(U[0]**2 + U[1]**2 + U[2]**2), )
        raise

    if 1:
        yaw = math.atan2( U[1],U[0] )
        pitch = math.atan2( U[2], U[0] )
        return euler_to_quat(yaw=yaw,pitch=pitch)
    else:
        if roll_angle != 0:
            # I presume this has an effect on the b component of the quaternion
            raise NotImplementedError('')
        fast = True
        if not fast:
            X = (1,0,0)
            v = nx.array(cross(X,U))
            v = v/norm(v)
        else:
            v = (0,-U[2],U[1]) # faster way of same
            
        if nx.sum(nx.abs(v))<1e-16:
            return make_quat(0, (1,0,0))
        
        if not fast:
            cos_theta = nx.dot(U,X)
            theta = math.acos(cos_theta/(norm(U)*norm(X)))
        else:
            cos_theta = U[0] # faster way of same
            theta = math.acos(cos_theta)
            
        #return cgtypes.quat(theta, cgtypes.vec3(v))
        return make_quat(theta, v)

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
    """returns yaw, pitch, roll"""
    qw = q.w; qx = q.x; qy = q.y; qz = q.z

    # from Martin Baker's website:
    #heading = math.atan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
    #attitude = math.asin(2*qx*qy + 2*qz*qw)
    #bank = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    #return heading, attitude, bank

    pitch = -math.atan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
    yaw = math.asin(2*qx*qy + 2*qz*qw)
    roll = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    return yaw, pitch, roll

def quat_to_absroll(q):
    qw = q.w; qx = q.x; qy = q.y; qz = q.z

    # from Martin Baker's website:
    #heading = math.atan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
    #attitude = math.asin(2*qx*qy + 2*qz*qw)
    #bank = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    #return heading, attitude, bank

    roll = math.atan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    return abs(roll)

class ObjectiveFunctionPosition:
    """methods from Kim, Hsieh, Wang, Wang, Fang, Woo"""
    def __init__(self, p, h, alpha):
        self.p = p
        self.h = h
        self.alpha = alpha
    def _getDistance(self, ps):
        return nx.sum(nx.sum((self.p - ps)**2, axis=1))
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
            def eval_pd(self,i):
                # evaluate 3 values (perturbations in x,y,z directions)
                perturb = 1e-6 # perturbation amount (in meters)
                
                P_i_x = self.P[i,0]
                # temporarily perturb P_i
                self.P[i,0] = P_i_x+perturb
                F_Px = self.objective_function.eval(P)
                self.P[i,0] = P_i_x
                
                P_i_y = self.P[i,1]
                self.P[i,1] = P_i_y+perturb
                F_Py = self.objective_function.eval(P)
                self.P[i,1] = P_i_y
                
                P_i_z = self.P[i,2]
                self.P[i,2] = P_i_z+perturb
                F_Pz = self.objective_function.eval(P)
                self.P[i,2] = P_i_z
                
                dFdP = ((F_Px-self.F_P)/perturb,
                        (F_Py-self.F_P)/perturb,
                        (F_Pz-self.F_P)/perturb)
                return dFdP
        _pd_finder = PDfinder(self,P)
        PDs = nx.array([ _pd_finder.eval_pd(i) for i in range(len(P)) ])
        return PDs

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
                perturb = 1e-6 # perturbation amount (must be less than sqrt(pi))
                #perturb = 1e-10 # perturbation amount (must be less than sqrt(pi))

                q_i = self.Q[i]
                q_i_inverse = q_i.inverse()
                
                qx = q_i*cgtypes.quat(0,perturb,0,0).exp()
                self.Q[i] = qx
                G_Qx = self.objective_function.eval(self.Q)
                dist_x = abs((q_i_inverse*qx).log())

                qy = q_i*cgtypes.quat(0,0,perturb,0).exp()
                self.Q[i] = qy
                G_Qy = self.objective_function.eval(self.Q)
                dist_y = abs((q_i_inverse*qy).log())

                qz = q_i*cgtypes.quat(0,0,0,perturb).exp()
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
