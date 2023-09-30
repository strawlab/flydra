"""Position and Quaternion orientation math"""
from __future__ import print_function

import math
import cgtypes  # cgkit 1.x
import numpy as nx
import numpy as np
import numpy
import time

nan = nx.nan


class NoIndexChangedClass:
    pass


no_index_changed = NoIndexChangedClass()

L_i = nx.array([0, 0, 0, 1, 3, 2])
L_j = nx.array([1, 2, 3, 2, 1, 3])


def Lmatrix2Lcoords(Lmatrix):
    return Lmatrix[L_i, L_j]


# pluecker_to_orient is flydra_core.reconstruct.line_direction


def pluecker_to_quat(line3d):
    """cannot set roll angle"""
    import flydra_core.reconstruct

    orient = flydra_core.reconstruct.line_direction(line3d)
    return orientation_to_quat(orient)


def pluecker_from_verts(A, B):
    if len(A) == 3:
        A = A[0], A[1], A[2], 1.0
    if len(B) == 3:
        B = B[0], B[1], B[2], 1.0
    A = nx.reshape(A, (4, 1))
    B = nx.reshape(B, (4, 1))
    L = nx.matrixmultiply(A, nx.transpose(B)) - nx.matrixmultiply(B, nx.transpose(A))
    return Lmatrix2Lcoords(L)


def norm_vec(V):
    Va = nx.asarray(V)
    if len(Va.shape) == 1:
        # vector
        U = Va / math.sqrt(Va[0] ** 2 + Va[1] ** 2 + Va[2] ** 2)  # normalize
    else:
        assert Va.shape[1] == 3
        Vamags = nx.sqrt(Va[:, 0] ** 2 + Va[:, 1] ** 2 + Va[:, 2] ** 2)
        U = Va / Vamags[:, nx.newaxis]
    return U


def rotate_velocity_by_orientation(vel, orient):
    # this is backwards from a normal quaternion rotation
    return orient.inverse() * vel * orient


def is_unit_vector(U, eps=1e-10):
    V = nx.asarray(U)
    Visdim1 = False
    if len(V.shape) == 1:
        V = V[nx.newaxis, :]
        Visdim1 = True
    V = V ** 2
    mag = nx.sqrt(nx.sum(V, axis=1))
    result = abs(mag - 1.0) < eps

    # consider nans to be unit vectors
    if Visdim1:
        if np.isnan(mag[0]):
            result = True
    else:
        result[np.isnan(mag)] = True

    return result


def world2body(U, roll_angle=0):
    """convert world coordinates to body-relative coordinates

    inputs:

    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    M (3x3 matrix), get world coords via nx.dot(M,X)
    """
    assert is_unit_vector(U)
    # notation from Wagner, 1986a, Appendix 1
    Bxy = math.atan2(U[1], U[0])  # heading angle
    Bxz = math.asin(U[2])  # pitch (body) angle
    Byz = roll_angle  # roll angle
    cos = math.cos
    sin = math.sin
    M = nx.array(
        (
            (cos(Bxy) * cos(-Bxz), sin(Bxy) * cos(-Bxz), -sin(-Bxz)),
            (
                cos(Bxy) * sin(-Bxz) * sin(Byz) - sin(Bxy) * cos(Byz),
                sin(Bxy) * sin(-Bxz) * sin(Byz) + cos(Bxy) * cos(Byz),
                cos(-Bxz) * sin(Byz),
            ),
            (
                cos(Bxy) * sin(-Bxz) * cos(Byz) + sin(Bxy) * sin(Byz),
                sin(Bxy) * sin(-Bxz) * cos(Byz) - cos(Bxy) * sin(Byz),
                cos(-Bxz) * cos(Byz),
            ),
        )
    )
    return M


def cross(vec1, vec2):
    return (
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    )


def make_quat(angle_radians, axis_of_rotation):
    half_angle = angle_radians / 2.0
    a = math.cos(half_angle)
    b, c, d = math.sin(half_angle) * norm_vec(axis_of_rotation)
    return cgtypes.quat(a, b, c, d)


def euler_to_orientation(yaw=0.0, pitch=0.0):
    """convert euler angles into orientation vector"""
    z = math.sin(pitch)
    xyr = math.cos(pitch)
    y = xyr * math.sin(yaw)
    x = xyr * math.cos(yaw)
    return x, y, z


def test_euler_to_orientation():
    xyz = euler_to_orientation(0, 0)
    assert np.allclose(xyz, (1.0, 0.0, 0.0))

    xyz = euler_to_orientation(math.pi, 0)
    assert np.allclose(xyz, (-1.0, 1.2246063538223773e-16, 0.0))

    xyz = euler_to_orientation(math.pi, math.pi / 2)
    assert np.allclose(xyz, (-6.1230317691118863e-17, 7.4983036091106872e-33, 1.0))

    xyz = euler_to_orientation(math.pi, -math.pi / 2)
    assert np.allclose(xyz, (-6.1230317691118863e-17, 7.4983036091106872e-33, -1.0))

    xyz = euler_to_orientation(math.pi, -math.pi / 4)
    assert np.allclose(
        xyz, (-0.70710678118654757, 8.6592745707193554e-17, -0.70710678118654746)
    )

    xyz = euler_to_orientation(math.pi / 2, -math.pi / 4)
    assert np.allclose(
        xyz, (4.3296372853596777e-17, 0.70710678118654757, -0.70710678118654746)
    )


def euler_to_quat(roll=0.0, pitch=0.0, yaw=0.0):
    # see http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm

    # Supposedly the order is heading first, then attitude, the bank.

    heading = yaw
    attitude = pitch
    bank = roll
    c1 = math.cos(heading / 2)
    s1 = math.sin(heading / 2)
    c2 = math.cos(attitude / 2)
    s2 = math.sin(attitude / 2)
    c3 = math.cos(bank / 2)
    s3 = math.sin(bank / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    z, y = y, -z
    return cgtypes.quat(w, x, y, z)


##    a,b,c=roll,-pitch,yaw
##    Qx = cgtypes.quat( math.cos(a/2.0), math.sin(a/2.0), 0, 0 )
##    Qy = cgtypes.quat( math.cos(b/2.0), 0, math.sin(b/2.0), 0 )
##    Qz = cgtypes.quat( math.cos(c/2.0), 0, 0, math.sin(c/2.0) )
##    return Qx*Qy*Qz


def orientation_to_euler(U):
    """
convert orientation to euler angles (in radians)

results are yaw, pitch (no roll is provided)

>>> print(orientation_to_euler( (1, 0, 0) ))
(0.0, 0.0)

>>> print(orientation_to_euler( (0, 1, 0) ))
(1.5707963267948966, 0.0)

>>> print(orientation_to_euler( (-1, 0, 0) ))
(3.141592653589793, 0.0)

>>> print(orientation_to_euler( (0, -1, 0) ))
(-1.5707963267948966, 0.0)

>>> print(orientation_to_euler( (0, 0, 1) ))
(0.0, 1.5707963267948966)

>>> print(orientation_to_euler( (0, 0, -1) ))
(0.0, -1.5707963267948966)

>>> print(orientation_to_euler( (0,0,0) )) # This is not a unit vector. # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
AssertionError

>>> r1=math.sqrt(2)/2

>>> '%.5f' % (math.pi/4)
'0.78540'

>>> print(orientation_to_euler( (r1,0,r1) ))
(0.0, 0.7853981633974483)

>>> print(orientation_to_euler( (r1,r1,0) ))
(0.7853981633974483, 0.0)

"""

    if numpy.isnan(U[0]):
        return (nan, nan)
    assert is_unit_vector(U)

    yaw = math.atan2(U[1], U[0])
    xy_r = math.sqrt(U[0] ** 2 + U[1] ** 2)
    pitch = math.atan2(U[2], xy_r)
    return yaw, pitch


def orientation_to_quat(U, roll_angle=0, force_pitch_0=False):
    """convert world coordinates to body-relative coordinates

    inputs:

    U is a unit vector indicating directon of body long axis
    roll_angle is the roll angle (in radians)

    output:

    quaternion (4 tuple)
    """
    if roll_angle != 0:
        # I presume this has an effect on the b component of the quaternion
        raise NotImplementedError("")
    # if type(U) == nx.NumArray and len(U.shape)==2: # array of vectors
    if 0:
        result = QuatSeq()
        for u in U:
            if numpy.isnan(u[0]):
                result.append(cgtypes.quat((nan, nan, nan, nan)))
            else:
                yaw, pitch = orientation_to_euler(u)
                result.append(euler_to_quat(yaw=yaw, pitch=pitch, roll=roll_angle))
        return result
    else:
        if numpy.isnan(U[0]):
            return cgtypes.quat((nan, nan, nan, nan))
        yaw, pitch = orientation_to_euler(U)
        if force_pitch_0:
            pitch = 0
        return euler_to_quat(yaw=yaw, pitch=pitch, roll=roll_angle)


def quat_to_orient(S3):
    """returns x, y, z for unit quaternions

    Parameters
    ----------
    S3 : {quaternion, QuatSeq instance}
      The quaternion or sequence of quaternions to transform into an
      orientation vector.

    Returns
    -------
    orient : {tuple, ndarray}

      If the input was a single quaternion, the output is (x,y,z). If
      the input was a QuatSeq instance of length N, the output is an
      ndarray of shape (N,3).

    """
    u = cgtypes.quat(0, 1, 0, 0)
    if type(S3)() == QuatSeq():  # XXX isinstance(S3,QuatSeq)
        V = [q * u * q.inverse() for q in S3]
        return nx.array([(v.x, v.y, v.z) for v in V])
    else:
        V = S3 * u * S3.inverse()
        return V.x, V.y, V.z


def quat_to_euler(q):
    """returns yaw, pitch, roll

    at singularities (north and south pole), assumes roll = 0
    """

    # See http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/

    eps = 1e-14
    qw = q.w
    qx = q.x
    qy = q.z
    qz = -q.y
    pitch_y = 2 * qx * qy + 2 * qz * qw
    if pitch_y > (1.0 - eps):  # north pole]
        yaw = 2 * math.atan2(qx, qw)
        pitch = math.asin(1.0)
        roll = 0.0
    elif pitch_y < -(1.0 - eps):  # south pole
        yaw = -2 * math.atan2(qx, qw)
        pitch = math.asin(-1.0)
        roll = 0.0
    else:
        yaw = math.atan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy ** 2 - 2 * qz ** 2)
        pitch = math.asin(pitch_y)
        roll = math.atan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx ** 2 - 2 * qz ** 2)
    return yaw, pitch, roll


def quat_to_absroll(q):
    qw = q.w
    qx = q.x
    qy = q.z
    qz = -q.y
    roll = math.atan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx ** 2 - 2 * qz ** 2)
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

        self.p_err_weights = nx.ones((len(p),))
        if no_distance_penalty_idxs is not None:
            for i in no_distance_penalty_idxs:
                self.p_err_weights[i] = 0

    def _getDistance(self, ps):
        result = nx.sum(self.p_err_weights * nx.sum((self.p - ps) ** 2, axis=1))
        assert not np.isnan(result)
        return result

    def _getEnergy(self, ps):
        d2p = (ps[2:] - 2 * ps[1:-1] + ps[:-2]) / (self.h ** 2)
        return nx.sum(nx.sum(d2p ** 2, axis=1))

    def eval(self, ps):
        D = self._getDistance(ps)
        E = self._getEnergy(ps)
        return D + self.alpha * E  # eqn. 22

    def get_del_F(self, P):
        class PDfinder:
            def __init__(self, objective_function, P):
                self.objective_function = objective_function
                self.P = P
                self.F_P = self.objective_function.eval(P)
                self.ndims = P.shape[1]

            def eval_pd(self, i):
                # evaluate 3 values (perturbations in x,y,z directions)
                PERTURB = 1e-6  # perturbation amount (in meters)

                dFdP = []
                for j in range(self.ndims):
                    P_i_j = self.P[i, j]
                    # temporarily perturb P_i
                    self.P[i, j] = P_i_j + PERTURB
                    F_Pj = self.objective_function.eval(P)
                    self.P[i, j] = P_i_j

                    dFdP.append((F_Pj - self.F_P) / PERTURB)
                return dFdP

        _pd_finder = PDfinder(self, P)
        PDs = nx.array([_pd_finder.eval_pd(i) for i in range(len(P))])
        return PDs


def smooth_position(P, delta_t, alpha, lmbda, eps, verbose=False):
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
    err = 2 * eps  # cycle at least once
    of = ObjectiveFunctionPosition(P, h, alpha)
    while err > eps:
        del_F = of.get_del_F(Pstar)
        err = nx.sum(nx.sum(del_F ** 2))
        Pstar = Pstar - lmbda * del_F
        if verbose:
            print("err %g (eps %g)" % (err, eps))
    return Pstar


class QuatSeq(list):
    def __abs__(self):
        return nx.array([abs(q) for q in self])

    def __add__(self, other):
        if isinstance(other, QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([p + q for p, q in zip(self, other)])
        else:
            raise ValueError("only know how to add QuatSeq with QuatSeq")

    def __sub__(self, other):
        if isinstance(other, QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([p - q for p, q in zip(self, other)])
        else:
            raise ValueError("only know how to subtract QuatSeq with QuatSeq")

    def __mul__(self, other):
        if isinstance(other, QuatSeq):
            assert len(self) == len(other)
            return QuatSeq([p * q for p, q in zip(self, other)])
        elif hasattr(other, "shape"):  # ndarray
            assert len(other.shape) == 2
            if other.shape[1] == 3:
                other = nx.concatenate((nx.zeros((other.shape[0], 1)), other), axis=1)
            assert other.shape[1] == 4
            other = QuatSeq([cgtypes.quat(o) for o in other])
            return self * other
        else:
            try:
                other = float(other)
            except (ValueError, TypeError):
                raise TypeError(
                    "only know how to multiply QuatSeq with QuatSeq, n*3 array or float"
                )
            return QuatSeq([p * other for p in self])

    def copy(self):
        """return a deep copy of self"""
        return QuatSeq([cgtypes.quat(q) for q in self])

    def __div__(self, other):
        # Python 2 only
        try:
            other = float(other)
        except ValueError:
            raise ValueError("only know how to divide QuatSeq with floats")
        return QuatSeq([p / other for p in self])

    def __truediv__(self, other):
        try:
            other = float(other)
        except ValueError:
            raise ValueError("only know how to divide QuatSeq with floats")
        return QuatSeq([p / other for p in self])

    def __pow__(self, n):
        return QuatSeq([q ** n for q in self])

    def __neg__(self):
        return QuatSeq([-q for q in self])

    def __str__(self):
        return "QuatSeq(" + list.__str__(self) + ")"

    def __repr__(self):
        return "QuatSeq(" + list.__repr__(self) + ")"

    def inverse(self):
        return QuatSeq([q.inverse() for q in self])

    def exp(self):
        return QuatSeq([q.exp() for q in self])

    def log(self):
        return QuatSeq([q.log() for q in self])

    def __getitem__(self, expr):
        if type(expr) is slice:
            return QuatSeq(list.__getitem__(self, expr))
        else:
            return list.__getitem__(self, expr)

    def __getslice__(self, *args, **kw):
        # Python 2 only
        return QuatSeq(list.__getslice__(self, *args, **kw))

    def get_w(self):
        return nx.array([q.w for q in self])

    w = property(get_w)

    def get_x(self):
        return nx.array([q.x for q in self])

    x = property(get_x)

    def get_y(self):
        return nx.array([q.y for q in self])

    y = property(get_y)

    def get_z(self):
        return nx.array([q.z for q in self])

    z = property(get_z)

    def to_numpy(self):
        return numpy.vstack((self.w, self.x, self.y, self.z))


def _calc_idxs_and_fixup_q(no_distance_penalty_idxs, q):
    valid_cond = None
    if no_distance_penalty_idxs is not None:
        valid_cond = np.ones(len(q), dtype=np.bool_)
        for i in no_distance_penalty_idxs:
            valid_cond[i] = 0

            # Set missing quaternion to some arbitrary value. The
            # _getEnergy() cost term should push it around to
            # minimize accelerations and it won't get counted in
            # the distance penalty.
            q[i] = cgtypes.quat(1, 0, 0, 0)
    else:
        no_distance_penalty_idxs = []

    for i, qi in enumerate(q):
        if numpy.any(numpy.isnan([qi.w, qi.x, qi.y, qi.z])):
            if not i in no_distance_penalty_idxs:
                raise ValueError(
                    "you are passing data with 'nan' "
                    "values, but have not set the "
                    "no_distance_penalty_idxs argument"
                )
    return valid_cond, no_distance_penalty_idxs, q


class ObjectiveFunctionQuats(object):
    """methods from Kim, Hsieh, Wang, Wang, Fang, Woo"""

    def __init__(
        self,
        q,
        h,
        beta,
        gamma,
        ignore_direction_sign=False,
        no_distance_penalty_idxs=None,
    ):
        """
        parameters
        ==========
        ignore_direction_sign - on each timestep, the distance term
        for each orientation will be the smallest of 2 possibilities:
        co-directional and anti-directional with the original
        direction vector. Note that this may make the derivative
        non-continuous and hence the entire operation unstable.

        """
        q = q.copy()  # make copy so we don't screw up original below

        (self.valid_cond, no_distance_penalty_idxs, q) = _calc_idxs_and_fixup_q(
            no_distance_penalty_idxs, q
        )

        if 1:
            if 0:
                # XXX this hasn't been tested in a long time...
                # convert back and forth from orientation to eliminate roll
                # ori = quat_to_orient(self.q)
                # no_roll_quat = orientation_to_quat(quat_to_orient(self.q))
                self.q_inverse = orientation_to_quat(quat_to_orient(self.q)).inverse()
            else:
                self.qorients = quat_to_orient(q)
        else:
            self.q_inverse = q.inverse()
        self.h = h
        self.h2 = self.h ** 2
        self.beta = beta
        self.gamma = gamma

        self.no_distance_penalty_idxs = no_distance_penalty_idxs
        self.ignore_direction_sign = ignore_direction_sign

    def _getDistance(self, qs, changed_idx=None):
        # This distance function is independent of "roll" angle, and
        # thus doesn't penalize changes in roll.
        if 0:
            # XXX this hasn't been tested in a long time...
            if 0:
                # convert back and forth from orientation to eliminate roll
                qs = orientation_to_quat(quat_to_orient(qs))
            return sum((abs((self.q_inverse * qs).log()) ** 2))
        else:
            # test distance of orientations in R3 (L2 norm)
            qtest_orients = quat_to_orient(qs)
            if self.ignore_direction_sign:
                # test both directions, take the minimum distance
                L2dist1 = nx.sum((qtest_orients - self.qorients) ** 2, axis=1)
                L2dist2 = nx.sum((qtest_orients + self.qorients) ** 2, axis=1)
                L2dist = np.min(L2dist1, L2dist2, axis=1)
            else:
                L2dist = nx.sum((qtest_orients - self.qorients) ** 2, axis=1)
            result = nx.sum(L2dist[self.valid_cond])
            assert not np.isnan(result)
            return result

    def _getRoll(self, qs, changed_idx=None):
        return sum([quat_to_absroll(q) for q in qs])

    def _getEnergy(self, qs, changed_idx=None):

        # We do not hide values at no_distance_penalty_idxs here
        # because we want qs to be any value -- that's the point. (Not
        # a good idea to set to nan.)

        omega_dot = (
            (qs[1:-1].inverse() * qs[2:]).log() - (qs[:-2].inverse() * qs[1:-1]).log()
        ) / self.h2
        result = nx.sum(abs(omega_dot) ** 2)
        assert not np.isnan(result)
        return result

    def eval(self, qs, changed_idx=None):
        D = self._getDistance(qs, changed_idx=changed_idx)
        E = self._getEnergy(qs, changed_idx=changed_idx)
        if self.gamma == 0:
            return D + self.beta * E  # eqn. 23
        R = self._getRoll(qs, changed_idx=changed_idx)
        return D + self.beta * E + self.gamma * R  # eqn. 23

    def get_del_G(self, Q):
        class PDfinder:
            """partial derivative finder"""

            def __init__(self, objective_function, Q, no_distance_penalty_idxs=None):
                if no_distance_penalty_idxs is None:
                    self.no_distance_penalty_idxs = []
                else:
                    self.no_distance_penalty_idxs = no_distance_penalty_idxs
                self.objective_function = objective_function
                self.Q = Q
                # G evaluated at Q
                self.G_Q = self.objective_function.eval(
                    Q, changed_idx=no_index_changed,  # nothing has changed yet
                )

            def eval_pd(self, i):
                # evaluate 3 values (perturbations in x,y,z directions)
                PERTURB = 1e-6  # perturbation amount (must be less than sqrt(pi))
                # PERTURB = 1e-10 # perturbation amount (must be less than sqrt(pi))

                q_i = self.Q[i]
                q_i_inverse = q_i.inverse()

                qx = q_i * cgtypes.quat(0, PERTURB, 0, 0).exp()
                self.Q[i] = qx
                G_Qx = self.objective_function.eval(self.Q, changed_idx=i)
                dist_x = abs((q_i_inverse * qx).log())

                qy = q_i * cgtypes.quat(0, 0, PERTURB, 0).exp()
                self.Q[i] = qy
                G_Qy = self.objective_function.eval(self.Q, changed_idx=i)
                dist_y = abs((q_i_inverse * qy).log())

                qz = q_i * cgtypes.quat(0, 0, 0, PERTURB).exp()
                self.Q[i] = qz
                G_Qz = self.objective_function.eval(self.Q, changed_idx=i)
                dist_z = abs((q_i_inverse * qz).log())

                self.Q[i] = q_i

                qdir = cgtypes.quat(
                    0,
                    (G_Qx - self.G_Q) / dist_x,
                    (G_Qy - self.G_Q) / dist_y,
                    (G_Qz - self.G_Q) / dist_z,
                )
                return qdir

        ##print 'Q',Q,'='*200
        pd_finder = PDfinder(self, Q)
        del_G_Q = QuatSeq([pd_finder.eval_pd(i) for i in range(len(Q))])
        return del_G_Q

    def set_cache_qs(self, qs):
        # no-op
        pass


class CachingObjectiveFunctionQuats(ObjectiveFunctionQuats):
    """This class should is a fast version of ObjectiveFunctionQuats when used properly"""

    def __init__(self, *args, **kwargs):
        super(CachingObjectiveFunctionQuats, self).__init__(*args, **kwargs)
        self.cache_qs = None

    def set_cache_qs(self, qs):
        # initialize the cache
        self.cache_qs = qs.copy()  # copy
        self._getDistance(self.cache_qs, init_cache=True)
        self._getEnergy(self.cache_qs, init_cache=True)

    def _calc_changed(self, qs):
        1 / 0
        import warnings

        warnings.warn("maximum performance not acheived because search forced")
        changed_idx = None
        for i in range(len(qs)):
            if not qs[i] == self.cache_qs[i]:
                changed_idx = i
                break
        return changed_idx

    def _getDistance(self, qs, init_cache=False, changed_idx=None):
        # This distance function is independent of "roll" angle, and
        # thus doesn't penalize changes in roll.
        if init_cache:
            # test distance of orientations in R3 (L2 norm)
            qtest_orients = quat_to_orient(qs)
            self._cache_L2dist = nx.sum((qtest_orients - self.qorients) ** 2, axis=1)
            assert changed_idx is None
        elif changed_idx is None:
            changed_idx = self._calc_changed(qs)

        mixin_new_results_with_cache = (changed_idx is not None) and (
            not isinstance(changed_idx, NoIndexChangedClass)
        )
        if mixin_new_results_with_cache:
            qtest_orient = quat_to_orient(qs[changed_idx])

            if self.ignore_direction_sign:
                # test both directions, take the minimum distance
                new_val1 = nx.sum(
                    (qtest_orient - self.qorients[changed_idx]) ** 2
                )  # no axis, 1 dim
                new_val2 = nx.sum(
                    (qtest_orient + self.qorients[changed_idx]) ** 2
                )  # no axis, 1 dim
                new_val = np.min(new_val1, new_val2)
            else:
                new_val = nx.sum(
                    (qtest_orient - self.qorients[changed_idx]) ** 2
                )  # no axis, 1 dim

            orig_val = self._cache_L2dist[changed_idx]
            self._cache_L2dist[changed_idx] = new_val

        result = nx.sum(self._cache_L2dist[self.valid_cond])

        if mixin_new_results_with_cache:
            self._cache_L2dist[changed_idx] = orig_val  # restore cache

        assert not np.isnan(result)
        return result

    def _getEnergy(self, qs, init_cache=False, changed_idx=None):
        if init_cache:
            self._cache_omega_dot = (
                (qs[1:-1].inverse() * qs[2:]).log()
                - (qs[:-2].inverse() * qs[1:-1]).log()
            ) / self.h2
            assert changed_idx is None
        elif changed_idx is None:
            changed_idx = self._calc_changed(qs)

        mixin_new_results_with_cache = (changed_idx is not None) and (
            not isinstance(changed_idx, NoIndexChangedClass)
        )
        if mixin_new_results_with_cache:
            fix_idx_center = changed_idx - 1  # central index into cache_omega_dot
            context_info = []
            for offset in [-1, 0, 1]:  # results aren't entirely local
                idx = fix_idx_center + offset
                if (idx < 0) or (idx >= len(self._cache_omega_dot)):
                    continue

                orig_val = self._cache_omega_dot[idx]
                new_val = (
                    (qs[idx + 1].inverse() * qs[idx + 2]).log()
                    - (qs[idx].inverse() * qs[idx + 1]).log()
                ) / self.h2
                context_info.append((idx, orig_val))
                self._cache_omega_dot[idx] = new_val
        result = nx.sum(abs(self._cache_omega_dot) ** 2)

        if mixin_new_results_with_cache:
            for (idx, orig_val) in context_info:
                self._cache_omega_dot[idx] = orig_val

        assert not np.isnan(result)
        return result


class QuatSmoother(object):
    def __init__(
        self,
        frames_per_second=None,
        beta=1.0,
        gamma=0.0,
        lambda2=1e-11,
        percent_error_eps_quats=9,
        epsilon2=0,
        max_iter2=2000,
    ):

        self.delta_t = 1.0 / frames_per_second
        self.beta = beta
        self.gamma = gamma
        self.lambda2 = lambda2
        self.percent_error_eps_quats = percent_error_eps_quats
        self.epsilon2 = epsilon2
        self.max_iter2 = max_iter2

    def smooth_directions(self, direction_vec, **kwargs):
        assert len(direction_vec.shape) == 2
        assert direction_vec.shape[1] == 3

        Q = QuatSeq([orientation_to_quat(U) for U in direction_vec])
        Qsmooth = self.smooth_quats(Q, **kwargs)
        direction_vec_smooth = quat_to_orient(Qsmooth)
        return direction_vec_smooth

    def smooth_quats(
        self,
        Q,
        display_progress=False,
        objective_func_name="CachingObjectiveFunctionQuats",
        no_distance_penalty_idxs=None,
    ):
        if objective_func_name == "CachingObjectiveFunctionQuats":
            of_class = CachingObjectiveFunctionQuats
        elif objective_func_name == "ObjectiveFunctionQuats":
            of_class = ObjectiveFunctionQuats
        if display_progress:
            print("constructing objective function...")
        of = of_class(
            Q,
            self.delta_t,
            self.beta,
            self.gamma,
            no_distance_penalty_idxs=no_distance_penalty_idxs,
        )
        if display_progress:
            print("done constructing objective function.")

        # lambda2 = 2e-9
        # lambda2 = 1e-9
        # lambda2 = 1e-11
        Q_k = Q.copy()  # make copy

        # make all Q_k elements valid so that we can improve them.
        (valid_cond, no_distance_penalty_idxs, Q_k) = _calc_idxs_and_fixup_q(
            no_distance_penalty_idxs, Q_k
        )

        last_err = None
        count = 0
        while count < self.max_iter2:
            count += 1
            if display_progress:
                start = time.time()
            if display_progress:
                print("initializing cache")
            of.set_cache_qs(Q_k)  # set the cache (no-op on non-caching version)
            if display_progress:
                print("computing del_G")
            del_G = of.get_del_G(Q_k)
            if display_progress:
                print("del_G done")
            D = of._getDistance(
                Q_k, changed_idx=no_index_changed
            )  # no change since we set the cache a few lines ago
            if display_progress:
                print("D done")
            E = of._getEnergy(
                Q_k, changed_idx=no_index_changed
            )  # no change since we set the cache a few lines ago
            if display_progress:
                print("E done")
            R = of._getRoll(Q_k)
            if display_progress:
                print(
                    "  G = %s + %s*%s + %s*%s"
                    % (str(D), str(self.beta), str(E), str(self.gamma), str(R))
                )
            if display_progress:
                stop = time.time()
            err = np.sqrt(np.sum(np.array(abs(del_G)) ** 2))

            if err < self.epsilon2:
                if display_progress:
                    print("reached epsilon2")
                break
            elif last_err is not None:
                pct_err = (last_err - err) / last_err * 100.0
                if display_progress:
                    print("Q elapsed: % 6.2f secs," % (stop - start,), end=" ")
                if display_progress:
                    print("current gradient:", err, end=" ")
                if display_progress:
                    print("   (%4.2f%%)" % (pct_err,))

                if err > last_err:
                    if display_progress:
                        print("ERROR: error is increasing, aborting")
                    break
                if pct_err < self.percent_error_eps_quats:
                    if display_progress:
                        print("reached percent_error_eps_quats")
                    break
            else:
                if display_progress:
                    print("Q elapsed: % 6.2f secs," % (stop - start,), end=" ")
                if display_progress:
                    print("current gradient:", err)
                pass
            last_err = err
            Q_k = Q_k * (del_G * -self.lambda2).exp()
        if count >= self.max_iter2:
            if display_progress:
                print("reached max_iter2")
            pass
        Qsmooth = Q_k
        return Qsmooth


def _test():
    # test math
    eps = 1e-7
    yaws = list(nx.arange(-math.pi, math.pi, math.pi / 16.0))
    yaws.append(math.pi)
    pitches = list(nx.arange(-math.pi / 2, math.pi / 2, math.pi / 16.0))
    pitches.append(math.pi / 2)
    err_count = 0
    total_count = 0
    for yaw in yaws:
        for pitch in pitches:
            had_err = False
            # forward and backward test 1
            yaw2, pitch2 = orientation_to_euler(euler_to_orientation(yaw, pitch))
            if abs(yaw - yaw2) > eps or abs(pitch - pitch2) > eps:
                print("orientation problem at", repr((yaw, pitch)))
                had_err = True

            if 1:
                # forward and backward test 2
                rolls = list(nx.arange(-math.pi / 2, math.pi / 2, math.pi / 16.0))
                rolls.append(math.pi / 2)
                for roll in rolls:
                    if pitch == math.pi / 2 or pitch == -math.pi / 2:
                        at_singularity = True
                    else:
                        at_singularity = False
                    yaw3, pitch3, roll3 = quat_to_euler(
                        euler_to_quat(yaw=yaw, pitch=pitch, roll=roll)
                    )
                    # relax criteria at singularities
                    singularity_is_ok = not (
                        at_singularity and abs(pitch3) - abs(pitch) < eps
                    )
                    if abs(yaw - yaw3) > eps or abs(pitch - pitch3) > eps:
                        if not (abs(yaw) == math.pi and (abs(yaw) - abs(yaw3) < eps)):
                            if not (at_singularity or singularity_is_ok):
                                print("quat problem at", repr((yaw, pitch, roll)))
                                print("               ", repr((yaw3, pitch3, roll3)))
                                print("    ", abs(yaw - yaw3))
                                print("    ", abs(pitch - pitch3))
                                print()
                                had_err = True

            # triangle test 1
            xyz1 = euler_to_orientation(yaw, pitch)
            xyz2 = quat_to_orient(euler_to_quat(yaw=yaw, pitch=pitch))
            l2dist = math.sqrt(nx.sum((nx.array(xyz1) - nx.array(xyz2)) ** 2))
            if l2dist > eps:
                print("other problem at", repr((yaw, pitch)))
                print(" ", xyz1)
                print(" ", xyz2)
                print()
                had_err = True

            # triangle test 2
            yaw4, pitch4, roll4 = quat_to_euler(orientation_to_quat(xyz1))
            if abs(yaw - yaw4) > eps or abs(pitch - pitch4) > eps:
                print("yet another problem at", repr((yaw, pitch)))
                print()
                had_err = True

            total_count += 1
            if had_err:
                err_count += 1
    print("Error count: (%d of %d)" % (err_count, total_count))

    # do doctest
    import doctest, PQmath

    return doctest.testmod(PQmath)


if __name__ == "__main__":
    _test()
