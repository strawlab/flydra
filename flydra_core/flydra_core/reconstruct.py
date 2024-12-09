from __future__ import print_function
import os, glob, sys, math

opj = os.path.join
import numpy as np
import numpy as nx
import numpy

import flydra_core._reconstruct_utils as reconstruct_utils  # in pyrex/C for speed
import time
from flydra_core.common_variables import (
    MINIMUM_ECCENTRICITY as DEFAULT_MINIMUM_ECCENTRICITY,
)
from flydra_core.common_variables import WATER_ROOTS_EPS
import scipy.linalg
import traceback
import flydra_core._pmat_jacobian as _pmat_jacobian  # in pyrex/C for speed
import flydra_core._pmat_jacobian_water as _pmat_jacobian_water  # in pyrex/C for speed
import flydra_core._fastgeom as fastgeom
import flydra_core.geom as slowgeom
import flydra_core.water as water
import flydra_core.align
import xml.etree.ElementTree as ET
import sys
import six

if sys.version_info.major == 3:
    import functools

import warnings
import cgtypes  # ubuntu: python-cgkit1 or "pip install cgkit1"
from optparse import OptionParser

R2D = 180.0 / np.pi
D2R = np.pi / 180.0

Z0_plane = slowgeom.Plane(slowgeom.ThreeTuple((0, 0, 1)), 0)

DEFAULT_WATER_REFRACTIVE_INDEX = 1.3330

WARN_CALIB_DIFF = False

L_i = nx.array([0, 0, 0, 1, 3, 2])
L_j = nx.array([1, 2, 3, 2, 1, 3])

NO_BACKWARDS_COMPAT = int(os.environ.get("FLYDRA_NO_BACKWARDS_COMPAT", "0"))


def mat2quat(m):
    """Initialize q from either a mat3 or mat4 and returns self."""
    # XXX derived from cgkit-1.2.0/cgtypes.pyx

    class PseudoQuat(object):
        pass

    q = PseudoQuat()

    d1 = m[0, 0]
    d2 = m[1, 1]
    d3 = m[2, 2]
    t = d1 + d2 + d3 + 1.0
    if t > 0.0:
        s = 0.5 / np.sqrt(t)
        q.w = 0.25 / s
        q.x = (m[2, 1] - m[1, 2]) * s
        q.y = (m[0, 2] - m[2, 0]) * s
        q.z = (m[1, 0] - m[0, 1]) * s
    else:
        ad1 = abs(d1)
        ad2 = abs(d2)
        ad3 = abs(d3)
        if ad1 >= ad2 and ad1 >= ad3:
            s = np.sqrt(1.0 + d1 - d2 - d3) * 2.0
            q.x = 0.5 / s
            q.y = (m[0, 1] + m[1, 0]) / s
            q.z = (m[0, 2] + m[2, 0]) / s
            q.w = (m[1, 2] + m[2, 1]) / s
        elif ad2 >= ad1 and ad2 >= ad3:
            s = np.sqrt(1.0 + d2 - d1 - d3) * 2.0
            q.x = (m[0, 1] + m[1, 0]) / s
            q.y = 0.5 / s
            q.z = (m[1, 2] + m[2, 1]) / s
            q.w = (m[0, 2] + m[2, 0]) / s
        else:
            s = np.sqrt(1.0 + d3 - d1 - d2) * 2.0
            q.x = (m[0, 2] + m[2, 0]) / s
            q.y = (m[1, 2] + m[2, 1]) / s
            q.z = 0.5 / s
            q.w = (m[0, 1] + m[1, 0]) / s

    return q


def my_rq(M):
    """RQ decomposition, ensures diagonal of R is positive"""
    R, K = scipy.linalg.rq(M)
    n = R.shape[0]
    for i in range(n):
        if R[i, i] < 0:
            # I checked this with Mathematica. Works if R is upper-triangular.
            R[:, i] = -R[:, i]
            K[i, :] = -K[i, :]
    ##    if R[0,0]<0:
    ##        # I checked this with Mathematica. Works if R is upper-triangular.
    ##        R[0,0] = -R[0,0]
    ##        K[0,:] = -K[0,:]
    return R, K


def filter_comments(lines_tmp):
    lines = []
    for line in lines_tmp:
        try:
            comment_idx = line.index("#")
            no_comment_line = line[:comment_idx]
        except ValueError:
            no_comment_line = line
        no_comment_line.strip()
        if len(no_comment_line):
            line = no_comment_line
        else:
            continue  # nothing on this line
        lines.append(line)
    return lines


def save_ascii_matrix(M, fd, isint=False):
    def fmt(f):
        if isint:
            return "%d" % f
        else:
            return "% 8e" % f

    A = nx.asarray(M)
    if len(A.shape) == 1:
        A = nx.reshape(A, (1, A.shape[0]))

    close_file = False
    if type(fd) == str:
        fd = open(fd, mode="wb")
        close_file = True

    for i in range(A.shape[0]):
        buf = " ".join(map(fmt, A[i, :]))
        fd.write(buf)
        fd.write("\n")
    if close_file:
        fd.close()


def as_column(x):
    x = nx.asarray(x)
    if len(x.shape) == 1:
        x = nx.reshape(x, (x.shape[0], 1))
    return x


def as_vec(x):
    x = nx.asarray(x)
    if len(x.shape) == 1:
        return x
    elif len(x.shape) == 2:
        long_dim = x.shape[0] + x.shape[1] - 1
        if (x.shape[0] * x.shape[1]) != long_dim:
            # more than 1 rows or columns
            raise ValueError("cannot convert to vector")
    else:
        raise ValueError("cannot convert to vector")
    return nx.reshape(x, (longdim,))


def Lcoords2Lmatrix(Lcoords):
    Lcoords = nx.asarray(Lcoords)
    Lmatrix = nx.zeros((4, 4), nx.float64)
    Lmatrix[L_i, L_j] = Lcoords
    Lmatrix[L_j, L_i] = -Lcoords
    return Lmatrix


def Lmatrix2Lcoords(Lmatrix):
    return Lmatrix[L_i, L_j]


def pts2Lmatrix(A, B):
    A = as_column(A)
    B = as_column(B)
    L = nx.dot(A, nx.transpose(B)) - nx.dot(B, nx.transpose(A))
    return L


def pts2Lcoords(A, B):
    return Lmatrix2Lcoords(pts2Lmatrix(A, B))


def line_direction_distance(Lcoords_A, Lcoords_B):
    """find distance between 2 line directions, A and B, ignore sign"""
    dir_vecA = cgtypes.vec3(Lcoords_A[2], -Lcoords_A[4], Lcoords_A[5])
    dir_vecB = cgtypes.vec3(Lcoords_B[2], -Lcoords_B[4], Lcoords_B[5])
    ## cos_angle = dir_vecA.dot( dir_vecB )
    ## angle = np.arccos( cos_angle )
    angle = dir_vecA.angle(dir_vecB)
    flipped_angle = np.pi - angle
    return min(angle, flipped_angle)


def norm_vec(V):
    Va = np.asarray(V, dtype=np.float64)  # force double precision floats
    if len(Va.shape) == 1:
        # vector
        U = Va / math.sqrt(np.sum(Va ** 2))  # normalize
    else:
        assert Va.shape[1] == 3
        Vamags = nx.sqrt(np.sum(Va ** 2, axis=1))
        U = Va / Vamags[:, nx.newaxis]
    return U


def line_direction(Lcoords):
    """convert from Pluecker coordinates to a direction"""
    # should maybe be PQmath.pluecker_to_orient
    L = nx.asanyarray(Lcoords)
    if L.ndim == 1:
        # single line coord
        U = nx.array((-L[2], L[4], -L[5]))
    else:
        assert L.ndim == 2
        assert L.shape[1] == 6
        # XXX could speed up with concatenate:
        U = nx.transpose(nx.array((-L[:, 2], L[:, 4], -L[:, 5])))
    return norm_vec(U)


def pluecker_from_verts(A, B):
    """

    See Hartley & Zisserman (2003) p. 70
    """
    if len(A) == 3:
        A = A[0], A[1], A[2], 1.0
    if len(B) == 3:
        B = B[0], B[1], B[2], 1.0
    A = nx.reshape(A, (4, 1))
    B = nx.reshape(B, (4, 1))
    L = nx.dot(A, nx.transpose(B)) - nx.dot(B, nx.transpose(A))
    return Lmatrix2Lcoords(L)


def pmat2cam_center(P):
    """

    See Hartley & Zisserman (2003) p. 163
    """
    assert P.shape == (3, 4)
    determinant = numpy.linalg.det

    # camera center
    X = determinant([P[:, 1], P[:, 2], P[:, 3]])
    Y = -determinant([P[:, 0], P[:, 2], P[:, 3]])
    Z = determinant([P[:, 0], P[:, 1], P[:, 3]])
    T = -determinant([P[:, 0], P[:, 1], P[:, 2]])

    C_ = nx.transpose(nx.array([[X / T, Y / T, Z / T]]))
    return C_


def setOfSubsets(L):
    """find all subsets of L

    from Alex Martelli:
    http://mail.python.org/pipermail/python-list/2001-January/067815.html

    This is also called the power set:
    http://en.wikipedia.org/wiki/Power_set
    """
    N = len(L)
    return [[L[i] for i in range(N) if X & (1 << i)] for X in range(2 ** N)]


intrinsic_normalized_eps = 1e-6


def normalize_pmat(pmat):
    pmat_orig = pmat
    M = pmat[:, :3]
    t = pmat[:, 3, numpy.newaxis]
    K, R = my_rq(M)
    if abs(K[2, 2] - 1.0) > intrinsic_normalized_eps:
        pmat = pmat / K[2, 2]
    assert numpy.allclose(pmat2cam_center(pmat_orig), pmat2cam_center(pmat))
    return pmat


def intersect_planes_to_find_line(P):
    # Is there a faster way when len(P)==2?
    try:
        u, d, vt = scipy.linalg.svd(P, full_matrices=True)
        # "two columns of V corresponding to the two largest singular
        # values span the best rank 2 approximation to A and may be
        # used to define the line of intersection of the planes"
        # (Hartley & Zisserman, p. 323)

        # get planes (take row because this is transpose(V))
        planeP = vt[0, :]
        planeQ = vt[1, :]

        # directly to Pluecker line coordinates
        Lcoords = (
            -(planeP[3] * planeQ[2]) + planeP[2] * planeQ[3],
            planeP[3] * planeQ[1] - planeP[1] * planeQ[3],
            -(planeP[2] * planeQ[1]) + planeP[1] * planeQ[2],
            -(planeP[3] * planeQ[0]) + planeP[0] * planeQ[3],
            -(planeP[2] * planeQ[0]) + planeP[0] * planeQ[2],
            -(planeP[1] * planeQ[0]) + planeP[0] * planeQ[1],
        )
    except Exception as exc:
        print("WARNING svd exception:", str(exc))
        Lcoords = None
    except:
        print("WARNING: unknown error in reconstruct.py")
        print("(you probably have an old version of numarray")
        print("and SVD did not converge)")
        Lcoords = None
    return Lcoords


def do_3d_operations_on_2d_point(
    helper,
    x0u,
    y0u,  # undistorted coords
    pmat_inv,
    camera_center,
    x0_abs,
    y0_abs,  # distorted coords
    rise,
    run,
):
    """this function is a hack"""

    matrixmultiply = numpy.dot
    svd = numpy.linalg.svd  # use fastest ATLAS/fortran libraries

    found_point_image_plane = [x0u, y0u, 1.0]

    if not (numpy.isnan(rise) or numpy.isnan(run)):
        # calculate plane containing camera origin and found line
        # in 3D world coords

        # Step 1) Find world coordinates points defining plane:
        #    A) found point
        X0 = matrixmultiply(pmat_inv, found_point_image_plane)

        #    B) another point on found line
        if 1:
            # The right way - convert the slope in distorted coords to
            # slope in undistorted coords.
            dirvec = numpy.array([run, rise])
            dirvec = dirvec / numpy.sqrt(numpy.sum(dirvec ** 2))  # normalize
            dirvec *= 0.1  # make really small
            x1u, y1u = helper.undistort(x0_abs + dirvec[0], y0_abs + dirvec[1])
            X1 = matrixmultiply(pmat_inv, [x1u, y1u, 1.0])
        else:
            # The wrong way - assume slope is the same in distorted and undistorted coords
            x1u, y1u = x0u + run, y0u + rise
            X1 = matrixmultiply(pmat_inv, [x1u, y1u, 1.0])

        #    C) world coordinates of camera center already known

        # Step 2) Find world coordinates of plane
        A = nx.array([X0, X1, camera_center])  # 3 points define plane
        try:
            u, d, vt = svd(A, full_matrices=True)
        except:
            print("rise,run", rise, run)
            print("pmat_inv", pmat_inv)
            print("X0, X1, camera_center", X0, X1, camera_center)
            raise
        Pt = vt[3, :]  # plane parameters

        p1, p2, p3, p4 = Pt[0:4]
        if numpy.isnan(p1):
            print("ERROR: SVD returned nan")
    else:
        p1, p2, p3, p4 = numpy.nan, numpy.nan, numpy.nan, numpy.nan

    # calculate pluecker coords of 3D ray from camera center to point
    # calculate 3D coords of point on image plane
    X0 = numpy.dot(pmat_inv, found_point_image_plane)
    X0 = X0[:3] / X0[3]  # convert to shape = (3,)
    # project line
    pluecker = pluecker_from_verts(X0, camera_center)
    (ray0, ray1, ray2, ray3, ray4, ray5) = pluecker  # unpack

    return (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4, ray5)


def angles_near(a, b, eps=None, mod_pi=False, debug=False):
    """compare if angles a and b are within eps of each other. assumes radians"""

    if mod_pi:
        r1 = angles_near(a, b, eps=eps, mod_pi=False, debug=debug)
        r2 = angles_near(a + np.pi, b, eps=eps, mod_pi=False, debug=debug)
        return r1 or r2

    diff = abs(a - b)

    diff = diff % (2 * numpy.pi)  # 0 <= diff <= 2pi

    result = abs(diff) < eps
    if debug:
        print("a", a * R2D)
        print("b", b * R2D)
        print("diff", diff * R2D)
    if abs(diff - 2 * numpy.pi) < eps:
        result = result or True
    return result


def test_angles_near():
    pi = np.pi
    for A in [-2 * pi, -1 * pi, -pi / 2, 0, pi / 2, pi, 2 * pi]:
        assert angles_near(A, A, eps=1e-15) == True
        assert angles_near(A, A + np.pi / 8, eps=np.pi / 4) == True
        assert angles_near(A, A - np.pi / 8, eps=np.pi / 4) == True

        assert angles_near(A, A + 1.1 * np.pi, eps=np.pi / 4) == False
        assert angles_near(A, A + 1.1 * np.pi, eps=np.pi / 4, mod_pi=True) == True


def test_angles_near2():
    a = 1.51026430701
    b = 2.92753197003
    assert angles_near(a, b, eps=10.0 * D2R, mod_pi=True) == False


def lineIntersect3D(PA, PB):
    """Find intersection point of lines in 3D space, in the least squares sense.

PA :          Nx3-matrix containing starting point of N lines
PB :          Nx3-matrix containing end point of N lines
P_Intersect : Best intersection point of the N lines, in least squares sense.
distances   : Distances from intersection point to the input lines
Anders Eikenes, 2012
translated to Python by Andrew Straw 2015
"""
    # from http://www.mathworks.com/matlabcentral/fileexchange/37192
    Si = PB - PA  # N lines described as vectors
    ni = Si / np.sqrt(np.sum(Si ** 2, axis=1))[:, np.newaxis]  # Normalize vectors
    nx = ni[:, 0]
    ny = ni[:, 1]
    nz = ni[:, 2]
    SXX = np.sum(nx ** 2 - 1)
    SYY = np.sum(ny ** 2 - 1)
    SZZ = np.sum(nz ** 2 - 1)
    SXY = np.sum(nx * ny)
    SXZ = np.sum(nx * nz)
    SYZ = np.sum(ny * nz)
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    CX = np.sum(PA[:, 0] * (nx ** 2 - 1) + PA[:, 1] * (nx * ny) + PA[:, 2] * (nx * nz))
    CY = np.sum(PA[:, 0] * (nx * ny) + PA[:, 1] * (ny ** 2 - 1) + PA[:, 2] * (ny * nz))
    CZ = np.sum(PA[:, 0] * (nx * nz) + PA[:, 1] * (ny * nz) + PA[:, 2] * (nz ** 2 - 1))
    C = np.array([CX, CY, CZ])
    P_intersect_results = np.linalg.lstsq(S, C)

    return P_intersect_results[0]


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
    """

    def __init__(
        self,
        cam_id=None,  # non-optional
        Pmat=None,  # non-optional
        res=None,  # non-optional
        helper=None,
        no_error_on_intrinsic_parameter_problem=False,
    ):
        if type(cam_id) != str:
            raise TypeError("cam_id must be string")
        pm = numpy.asarray(Pmat)
        if pm.shape != (3, 4):
            raise ValueError("Pmat must have shape (3,4)")
        if len(res) != 2:
            raise ValueError("len(res) must be 2 (res = %s)" % repr(res))

        self.cam_id = cam_id
        self.Pmat = Pmat
        self.res = res

        if helper is None:
            M = numpy.asarray(Pmat)
            cam_center = pmat2cam_center(M)

            intrinsic_parameters, cam_rotation = my_rq(M[:, :3])
            # intrinsic_parameters = intrinsic_parameters/intrinsic_parameters[2,2] # normalize
            eps = 1e-6
            if abs(intrinsic_parameters[2, 2] - 1.0) > eps:
                if no_error_on_intrinsic_parameter_problem:
                    warnings.warn(
                        "expected last row/col of intrinsic "
                        "parameter matrix to be unity. It is %s"
                        % intrinsic_parameters[2, 2]
                    )
                    intrinsic_parameters = (
                        intrinsic_parameters / intrinsic_parameters[2, 2]
                    )
                    print(intrinsic_parameters)
                else:
                    print(
                        "WARNING: expected last row/col of intrinsic "
                        "parameter matrix to be unity"
                    )
                    print("intrinsic_parameters[2,2]", intrinsic_parameters[2, 2])
                    raise ValueError(
                        "expected last row/col of intrinsic "
                        "parameter matrix to be unity"
                    )

            fc1 = intrinsic_parameters[0, 0]
            cc1 = intrinsic_parameters[0, 2]
            fc2 = intrinsic_parameters[1, 1]
            cc2 = intrinsic_parameters[1, 2]

            helper = reconstruct_utils.ReconstructHelper(
                fc1,
                fc2,  # focal length
                cc1,
                cc2,  # image center
                0,
                0,  # radial distortion
                0,
                0,
            )  # tangential distortion
        if not isinstance(helper, reconstruct_utils.ReconstructHelper):
            raise TypeError(
                "helper must be reconstruct_utils.ReconstructHelper instance"
            )
        self.helper = helper

        self.pmat_inv = numpy.linalg.pinv(self.Pmat)

    @classmethod
    def from_pymvg(cls, pymvg_cam):
        pmat = pymvg_cam.get_M()
        camdict = pymvg_cam.to_dict()
        if np.sum(abs(np.array(camdict["D"]))) != 0:
            rect = np.array(pymvg_cam.to_dict()["R"])
            rdiff = rect - np.eye(3)
            rsum = np.sum(abs(rdiff.ravel()))
            if rsum != 0:
                raise NotImplementedError("no support for rectification")
            d = np.array(camdict["D"])
            assert d.ndim == 1
            assert d.shape == (5,)
            # fc1, fc1, cc1, cc2
            r1, r2, t1, t2, r3 = d[:5]
            K = pymvg_cam.get_K()
            fc1 = K[0, 0]
            fc2 = K[1, 1]
            cc1 = K[0, 2]
            cc2 = K[1, 2]
            helper = reconstruct_utils.ReconstructHelper(
                fc1,
                fc2,  # focal length
                cc1,
                cc2,  # image center
                r1,
                r2,  # radial distortion
                t1,
                t2,  # tangential distortion
                k3=r3,  # more radial distortion
            )
        else:
            helper = None
        cam_id = camdict["name"]
        result = cls(
            cam_id=cam_id,
            Pmat=pmat,
            res=(camdict["width"], camdict["height"]),
            helper=helper,
        )
        return result

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        return (
            (self.cam_id == other.cam_id)
            and numpy.allclose(self.Pmat, other.Pmat)
            and numpy.allclose(self.res, other.res)
            and self.helper == other.helper
        )

    def convert_to_pymvg(self):
        import pymvg.camera_model
        import pymvg.util

        if not self.helper.simple:
            raise ValueError("this camera cannot be converted to PyMVG")

        K, R = self.get_KR()
        if not pymvg.util.is_rotation_matrix(R):
            # RQ may return left-handed rotation matrix. Make right-handed.
            R2 = -R
            K2 = -K
            assert np.allclose(np.dot(K2, R2), np.dot(K, R))
            K, R = K2, R2

        K = K / K[2, 2]  # normalize

        P = np.zeros((3, 4))
        P[:3, :3] = K
        KK = self.helper.get_K()

        # (ab)use PyMVG's rectification to do coordinate transform
        # for MCSC's undistortion.

        # The intrinsic parameters used for 3D -> 2D.
        ex = P[0, 0]
        bx = P[0, 2]
        Sx = P[0, 3]
        ey = P[1, 1]
        by = P[1, 2]
        Sy = P[1, 3]

        # Parameters used to define undistortion coordinates.
        fx = KK[0, 0]
        fy = KK[1, 1]
        cx = KK[0, 2]
        cy = KK[1, 2]

        rect = np.array(
            [
                [ex / fx, 0, (bx + Sx - cx) / fx],
                [0, ey / fy, (by + Sy - cy) / fy],
                [0, 0, 1],
            ]
        ).T

        k1, k2, p1, p2, k3 = self.helper.get_nlparams()
        distortion = [k1, k2, p1, p2, k3]

        C = self.get_cam_center()
        rot = R
        t = -np.dot(rot, C)[:, 0]

        d = {
            "width": self.res[0],
            "height": self.res[1],
            "P": P,
            "K": KK,
            "R": rect,
            "translation": t,
            "Q": rot,
            "D": distortion,
            "name": self.cam_id,
        }
        cnew = pymvg.camera_model.CameraModel.from_dict(d)
        return cnew

    def get_pmat(self):
        return self.Pmat

    def get_copy(self):
        Pmat = np.array(self.Pmat, copy=True)
        copy = SingleCameraCalibration(
            cam_id=self.cam_id, Pmat=Pmat, res=self.res, helper=self.helper,
        )
        return copy

    def get_aligned_copy(self, M):
        aligned_Pmat = flydra_core.align.align_pmat2(M, self.Pmat)
        aligned = SingleCameraCalibration(
            cam_id=self.cam_id, Pmat=aligned_Pmat, res=self.res, helper=self.helper,
        )
        return aligned

    def get_res(self):
        return self.res

    def get_cam_center(self):
        """get the 3D location of the camera center in world coordinates"""
        # should be called get_camera_center?
        return pmat2cam_center(self.Pmat)

    def get_M(self):
        """return parameters except extrinsic translation params"""
        return self.Pmat[:, :3]

    def get_t(self):
        """return extrinsic translation parameters"""
        return self.Pmat[:, 3, numpy.newaxis]

    def get_KR(self):
        """return intrinsic params (K) and extrinsic rotation/scale params (R)"""
        M = self.get_M()
        K, R = my_rq(M)
        ##        if K[2,2] != 0.0:
        ##            # normalize K
        ##            K = K/K[2,2]
        return K, R

    def get_mean_focal_length(self):
        K, R = self.get_KR()
        return (K[0, 0] + K[1, 1]) / 2.0

    def get_image_center(self):
        K, R = self.get_KR()
        return K[0, 2], K[1, 2]

    def get_extrinsic_parameter_matrix(self):
        """contains rotation and translation information"""
        C_ = self.get_cam_center()
        K, R = self.get_KR()
        t = numpy.dot(-R, C_)
        ext = numpy.concatenate((R, t), axis=1)
        return ext

    def get_example_3d_point_creating_image_point(self, image_point, w_val=1.0):
        # project back through principal point to get 3D line
        c1 = self.get_cam_center()[:, 0]

        x2d = (image_point[0], image_point[1], 1.0)
        c2 = numpy.dot(self.pmat_inv, as_column(x2d))[:, 0]
        c2 = c2[:3] / c2[3]

        direction = c2 - c1
        direction = direction / numpy.sqrt(numpy.sum(direction ** 2))
        c3 = c1 + direction * w_val
        return c3

    def get_optical_axis(self):
        # project back through principal point to get 3D line
        c1 = self.get_cam_center()[:, 0]
        pp = self.get_image_center()

        x2d = (pp[0], pp[1], 1.0)
        c2 = numpy.dot(self.pmat_inv, as_column(x2d))[:, 0]
        c2 = c2[:3] / c2[3]
        c1 = fastgeom.ThreeTuple(c1)
        c2 = fastgeom.ThreeTuple(c2)
        return fastgeom.line_from_points(c1, c2)

    def get_up_vector(self):
        # create up vector from image plane
        pp = self.get_image_center()
        x2d_a = (pp[0], pp[1], 1.0)
        c2_a = numpy.dot(self.pmat_inv, as_column(x2d_a))[:, 0]
        c2_a = c2_a[:3] / c2_a[3]

        x2d_b = (pp[0], pp[1] + 1, 1.0)
        c2_b = numpy.dot(self.pmat_inv, as_column(x2d_b))[:, 0]
        c2_b = c2_b[:3] / c2_b[3]

        up_dir = c2_b - c2_a
        return norm_vec(up_dir)

    def to_file(self, filename):
        fd = open(filename, "wb")
        fd.write('cam_id = "%s"\n' % self.cam_id)

        fd.write("pmat = [\n")
        for row in self.Pmat:
            fd.write("        [%s, %s, %s, %s],\n" % tuple([repr(x) for x in row]))
        fd.write("       ]\n")

        fd.write("res = (%d,%d)\n" % (self.res[0], self.res[1]))

        fd.write("K = [\n")
        for row in self.helper.get_K():
            fd.write("     [%s, %s, %s],\n" % tuple([repr(x) for x in row]))
        fd.write("    ]\n")

        k1, k2, p1, p2, k3 = self.helper.get_nlparams()
        fd.write("radial_params = %s, %s, %s\n" % (repr(k1), repr(k2), repr(k3)))
        fd.write("tangential_params = %s, %s\n" % (repr(p1), repr(p2)))

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
        K, R = self.get_KR()
        t = self.get_t()[:, 0]  # drop a dimension

        # extract sba parameters from intrinsic parameter matrix
        focal = K[0, 0]
        ppx = K[0, 2]
        ppy = K[1, 2]
        aspect = K[1, 1] / K[0, 0]
        skew = K[0, 1]

        # extract rotation quaternion from rotation matrix
        q = mat2quat(R)

        qw = q.w
        qx = q.x
        qy = q.y
        qz = q.z
        t0 = t[0]
        t1 = t[1]
        t2 = t[2]
        result = (
            "%(focal)s %(ppx)s %(ppy)s %(aspect)s %(skew)s "
            "%(qw)s %(qx)s %(qy)s %(qz)s "
            "%(t0)s %(t1)s %(t2)s" % locals()
        )
        return result

    def get_3D_plane_and_ray(self, x0d, y0d, slope):
        """return a ray and plane defined by a point and a slope at that point

        The 2D distorted point (x0d, y0d) is transfored to a 3D
        ray. Together with the slope at that point, a 3D plane that
        passes through the 2D point in the direction of the slope is
        returned.

        """
        # undistort
        x0u, y0u = self.helper.undistort(x0d, y0d)
        rise = slope
        run = 1.0
        meter_me = self
        pmat_meters_inv = meter_me.pmat_inv
        # homogeneous coords for camera centers
        camera_center = np.ones((4,))
        camera_center[:3] = self.get_cam_center()[:, 0]
        try:
            tmp = do_3d_operations_on_2d_point(
                self.helper, x0u, y0u, self.pmat_inv, camera_center, x0d, y0d, rise, run
            )
        except:
            print("camera_center", camera_center)
            raise
        (p1, p2, p3, p4, ray0, ray1, ray2, ray3, ray4, ray5) = tmp
        plane = (p1, p2, p3, p4)
        ray = (ray0, ray1, ray2, ray3, ray4, ray5)
        return plane, ray

    def add_element(self, parent):
        """add self as XML element to parent"""
        assert ET.iselement(parent)
        elem = ET.SubElement(parent, "single_camera_calibration")

        cam_id = ET.SubElement(elem, "cam_id")
        cam_id.text = self.cam_id

        pmat = ET.SubElement(elem, "calibration_matrix")
        fd = six.StringIO()
        save_ascii_matrix(self.Pmat, fd)
        mystr = fd.getvalue()
        mystr = mystr.strip()
        mystr = mystr.replace("\n", "; ")
        pmat.text = mystr
        fd.close()

        res = ET.SubElement(elem, "resolution")
        res.text = " ".join(map(str, self.res))

        self.helper.add_element(elem)

    def as_obj_for_json(self):
        result = dict(
            cam_id=self.cam_id,
            calibration_matrix=[row.tolist() for row in self.Pmat],
            resolution=list(self.res),
            non_linear_parameters=self.helper.as_obj_for_json(),
        )
        return result


def SingleCameraCalibration_fromfile(filename):
    params = {}
    execfile(filename, params)
    pmat = numpy.asarray(params["pmat"])  # XXX redundant information in pmat and K
    K = numpy.asarray(params["K"])
    cam_id = params["cam_id"]
    res = params["res"]
    pp = params["pp"]
    k1, k2 = params["radial_params"]
    p1, p2 = params["tangential_params"]

    fc1 = K[0, 0]
    cc1 = K[0, 2]
    fc2 = K[1, 1]
    cc2 = K[1, 2]

    helper = reconstruct_utils.ReconstructHelper(
        fc1,
        fc2,  # focal length
        cc1,
        cc2,  # image center
        k1,
        k2,  # radial distortion
        p1,
        p2,
    )  # tangential distortion
    return SingleCameraCalibration(cam_id=cam_id, Pmat=pmat, res=res, helper=helper)


def SingleCameraCalibration_from_xml(elem, helper=None):
    """ loads a camera calibration from an Elementree XML node """
    assert ET.iselement(elem)
    assert elem.tag == "single_camera_calibration"
    cam_id = elem.find("cam_id").text
    pmat = numpy.array(np.asmatrix(elem.find("calibration_matrix").text))
    res = numpy.array(np.asmatrix(elem.find("resolution").text))[0, :]
    scale_elem = elem.find("scale_factor")
    if NO_BACKWARDS_COMPAT:
        assert scale_elem is None, "XML file has outdated <scale_factor>"
    else:
        if scale_elem is not None:
            # backwards compatibility
            scale = float(scale_elem.text)
            if scale != 1.0:
                warnings.warn("converting old scaled calibration")
                scale_array = numpy.ones((3, 4))
                scale_array[:, 3] = scale  # mulitply last column by scale
                pmat = scale_array * pmat  # element-wise multiplication

    if not helper:
        helper_elem = elem.find("non_linear_parameters")
        if helper_elem is not None:
            helper = reconstruct_utils.ReconstructHelper_from_xml(helper_elem)
        else:
            # make with no non-linear stuff (i.e. make linear)
            helper = reconstruct_utils.ReconstructHelper(
                1,
                1,  # focal length
                0,
                0,  # image center
                0,
                0,  # radial distortion
                0,
                0,
            )  # tangential distortion

    return SingleCameraCalibration(cam_id=cam_id, Pmat=pmat, res=res, helper=helper)


def SingleCameraCalibration_from_xmlfile(fname, *args, **kwargs):
    root = ET.parse(fname).getroot()
    return SingleCameraCalibration_from_xml(root, *args, **kwargs)


def SingleCameraCalibration_from_basic_pmat(pmat, **kw):
    M = numpy.asarray(pmat)
    cam_center = pmat2cam_center(M)

    intrinsic_parameters, cam_rotation = my_rq(M[:, :3])
    # intrinsic_parameters = intrinsic_parameters/intrinsic_parameters[2,2] # normalize
    if abs(intrinsic_parameters[2, 2] - 1.0) > intrinsic_normalized_eps:
        raise ValueError(
            "expected last row/col of intrinsic parameter matrix to be unity"
        )

    # (K = intrinsic parameters)

    # cam_translation = numpy.dot( -cam_rotation, cam_center )
    # extrinsic_parameters = numpy.concatenate( (cam_rotation, cam_translation), axis=1 )

    # mean_focal_length = (intrinsic_parameters[0,0]+intrinsic_parameters[1,1])/2.0
    # center = intrinsic_parameters[0,2], intrinsic_parameters[1,2]

    # focalLength, center = compute_stuff_from_cal_matrix(cal)

    fc1 = intrinsic_parameters[0, 0]
    cc1 = intrinsic_parameters[0, 2]
    fc2 = intrinsic_parameters[1, 1]
    cc2 = intrinsic_parameters[1, 2]

    if "helper" in kw:
        helper = kw.pop("helper")
    else:
        helper = reconstruct_utils.ReconstructHelper(
            fc1,
            fc2,  # focal length
            cc1,
            cc2,  # image center
            0,
            0,  # radial distortion
            0,
            0,
        )  # tangential distortion
    return SingleCameraCalibration(Pmat=M, helper=helper, **kw)


def pretty_dump(e, ind=""):
    # from http://www.devx.com/opensource/Article/33153/0/page/4

    # start with indentation
    s = ind
    # put tag (don't close it just yet)
    s += "<" + e.tag
    # add all attributes
    for (name, value) in e.items():
        s += " " + name + "=" + "'%s'" % value
    # if there is text close start tag, add the text and add an end tag
    if e.text and e.text.strip():
        s += ">" + e.text + "</" + e.tag + ">"
    else:
        # if there are children...
        if len(e) > 0:
            # close start tag
            s += ">"
            # add every child in its own line indented
            for child in e:
                s += "\n" + pretty_dump(child, ind + "  ")
            # add closing tag in a new line
            s += "\n" + ind + "</" + e.tag + ">"
        else:
            # no text and no children, just close the starting tag
            s += " />"
    return s


def Reconstructor_from_xml(elem):
    """ Loads a reconstructor from an Elementree XML node """
    assert ET.iselement(elem)
    assert elem.tag == "multi_camera_reconstructor"
    sccs = []
    minimum_eccentricity = None
    has_water = False

    for child in elem:
        if child.tag == "single_camera_calibration":
            scc = SingleCameraCalibration_from_xml(child)
            sccs.append(scc)
        elif child.tag == "minimum_eccentricity":
            minimum_eccentricity = float(child.text)
        elif child.tag == "water":
            refractive_index = None
            if hasattr(child, "text"):
                if child.text is not None:
                    if len(child.text.strip()) != 0:
                        refractive_index = float(child.text)
            if refractive_index is None:
                refractive_index = DEFAULT_WATER_REFRACTIVE_INDEX
            has_water = True
        else:
            raise ValueError("unknown tag: %s" % child.tag)
    r = Reconstructor(sccs, minimum_eccentricity=minimum_eccentricity)
    if has_water:
        wateri = water.WaterInterface(
            refractive_index=refractive_index, water_roots_eps=WATER_ROOTS_EPS
        )
        r.add_water(wateri)
    return r


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

    def __init__(
        self,
        cal_source=None,
        do_normalize_pmat=True,
        minimum_eccentricity=None,
        wateri=None,
    ):
        self.cal_source = cal_source

        if isinstance(self.cal_source, six.string_types):
            if not self.cal_source.endswith("h5"):
                if os.path.isdir(self.cal_source):
                    self.cal_source_type = "normal files"
                elif self.cal_source.endswith(".json"):
                    self.cal_source_type = "json file"
                else:
                    self.cal_source_type = "xml file"
            else:
                self.cal_source_type = "pytables filename"
        elif hasattr(self.cal_source, "__len__"):  # is sequence
            for i in range(len(self.cal_source)):
                if not isinstance(self.cal_source[i], SingleCameraCalibration):
                    raise TypeError(
                        "If calsource is a sequence, it must "
                        "be a string specifying calibration "
                        "directory or a sequence of "
                        "SingleCameraCalibration instances."
                    )
            self.cal_source_type = "SingleCameraCalibration instances"
        else:
            self.cal_source_type = "pytables"

        close_cal_source = False
        if self.cal_source_type == "pytables filename":
            import tables as PT  # PyTables

            use_cal_source = PT.open_file(self.cal_source, mode="r")
            close_cal_source = True
            self.cal_source_type = "pytables"
        else:
            use_cal_source = self.cal_source

        if self.cal_source_type == "normal files":
            fd = open(os.path.join(use_cal_source, "camera_order.txt"), "r")
            cam_ids = fd.read().split("\n")
            fd.close()
            if cam_ids[-1] == "":
                del cam_ids[-1]  # remove blank line
        elif self.cal_source_type == "xml file":
            root = ET.parse(use_cal_source).getroot()
            if root.tag == "multi_camera_reconstructor":
                next_self = Reconstructor_from_xml(root)
            else:
                r_node = root.find("multi_camera_reconstructor")
                if r_node is None:
                    raise ValueError("XML file does not contain reconstructor node")
            cam_ids = next_self.cam_ids
        elif self.cal_source_type == "json file":
            from pymvg.multi_camera_system import MultiCameraSystem

            mvg = MultiCameraSystem.from_pymvg_file(use_cal_source)
            next_self = Reconstructor.from_pymvg(mvg)
            cam_ids = next_self.cam_ids
        elif self.cal_source_type == "pytables":
            results = use_cal_source
            nodes = results.root.calibration.pmat._f_list_nodes()
            cam_ids = []
            for node in nodes:
                cam_ids.append(node.name)
        elif self.cal_source_type == "SingleCameraCalibration instances":
            cam_ids = [scci.cam_id for scci in use_cal_source]
        else:
            raise ValueError("unknown cal_source_type '%s'" % self.cal_source_type)

        if minimum_eccentricity is None:
            self.minimum_eccentricity = None

            # not specified in call to constructor
            if self.cal_source_type == "pytables":
                # load from file
                mi_col = results.root.calibration.additional_info[:][
                    "minimum_eccentricity"
                ]
                assert len(mi_col) == 1
                self.minimum_eccentricity = mi_col[0]

                if hasattr(results.root.calibration, "refractive_interfaces"):
                    ri = results.root.calibration.refractive_interfaces[:]
                    assert len(ri) == 1
                    row = ri[0]
                    wateri = water.WaterInterface(
                        refractive_index=row["n2"], water_roots_eps=WATER_ROOTS_EPS
                    )
                    assert row["n1"] == wateri.n1
                    assert row["n2"] == wateri.n2
                    assert row["plane_normal_x"] == 0
                    assert row["plane_normal_y"] == 0
                    assert row["plane_normal_z"] != 0
                    assert row["plane_dist_from_origin"] == 0

            elif self.cal_source_type == "normal files":
                min_e_fname = os.path.join(use_cal_source, "minimum_eccentricity.txt")
                if os.path.exists(min_e_fname):
                    fd = open(min_e_fname, "r")
                    self.minimum_eccentricity = float(fd.read().strip())
                    fd.close()
            elif self.cal_source_type in ["xml file", "json file"]:
                self.minimum_eccentricity = next_self.minimum_eccentricity
            if self.minimum_eccentricity is None:
                # use default
                if int(os.environ.get("FORCE_MINIMUM_ECCENTRICITY", "0")):
                    raise ValueError("minimum_eccentricity cannot be default")
                else:
                    warnings.warn("No minimum eccentricity specified, using default")
                    self.minimum_eccentricity = DEFAULT_MINIMUM_ECCENTRICITY
        else:
            # use the value that was passed in to constructor
            self.minimum_eccentricity = minimum_eccentricity

        N = len(cam_ids)
        # load calibration matrices
        self.Pmat = {}
        self.Res = {}
        self._helper = {}

        if self.cal_source_type == "normal files":
            res_fd = open(os.path.join(use_cal_source, "Res.dat"), "r")
            for i, cam_id in enumerate(cam_ids):
                fname = "camera%d.Pmat.cal" % (i + 1)
                pmat = np.loadtxt(opj(use_cal_source, fname))  # 3 rows x 4 columns
                if do_normalize_pmat:
                    pmat_orig = pmat
                    pmat = normalize_pmat(pmat)
                ##                    if not numpy.allclose(pmat_orig,pmat):
                ##                        assert numpy.allclose(pmat2cam_center(pmat_orig),pmat2cam_center(pmat))
                ##                        #print('normalized pmat, but camera center should  changed for %s'%cam_id)
                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = list(map(int, res_fd.readline().split()))
            res_fd.close()

            # load non linear parameters
            rad_files = glob.glob(os.path.join(use_cal_source, "*.rad"))
            for cam_id_enum, cam_id in enumerate(cam_ids):
                filename = os.path.join(
                    use_cal_source, "basename%d.rad" % (cam_id_enum + 1,)
                )
                if not os.path.exists(filename):
                    if len(rad_files):
                        raise RuntimeError(
                            '.rad files present but none named "%s"' % filename
                        )
                    warnings.warn(
                        "no non-linear data (e.g. radial distortion) "
                        "in calibration for %s" % cam_id
                    )
                    self._helper[cam_id] = SingleCameraCalibration_from_basic_pmat(
                        self.Pmat[cam_id], cam_id=cam_id, res=self.Res[cam_id]
                    ).helper
                    continue

                self._helper[
                    cam_id
                ] = reconstruct_utils.make_ReconstructHelper_from_rad_file(filename)

        elif self.cal_source_type == "pytables":
            import tables as PT  # pytables

            for cam_id in cam_ids:
                pmat = nx.array(results.root.calibration.pmat.__getattr__(cam_id))
                res = tuple(results.root.calibration.resolution.__getattr__(cam_id))
                K = nx.array(
                    results.root.calibration.intrinsic_linear.__getattr__(cam_id)
                )
                nlparams = list(
                    results.root.calibration.intrinsic_nonlinear.__getattr__(cam_id)
                )
                if len(nlparams) == 4:
                    nlparams = nlparams + [0.0]
                assert len(nlparams) == 5
                try:
                    scale = nx.array(
                        results.root.calibration.scale_factor2meters.__getattr__(cam_id)
                    )
                except PT.exceptions.NoSuchNodeError:
                    pass
                else:
                    if NO_BACKWARDS_COMPAT:
                        raise ("old 'scale_factor2meters' present in h5 file")

                    if scale != 1.0:
                        warnings.warn("converting old scaled calibration")
                        scale_array = numpy.ones((3, 4))
                        scale_array[:, 3] = scale  # mulitply last column by scale
                        pmat = scale_array * pmat  # element-wise multiplication

                self.Pmat[cam_id] = pmat
                self.Res[cam_id] = res
                self._helper[cam_id] = reconstruct_utils.ReconstructHelper(
                    K[0, 0],
                    K[1, 1],
                    K[0, 2],
                    K[1, 2],
                    nlparams[0],
                    nlparams[1],
                    nlparams[2],
                    nlparams[3],
                    k3=nlparams[4],
                )

        elif self.cal_source_type == "SingleCameraCalibration instances":
            # find instance
            for cam_id in cam_ids:
                for scci in use_cal_source:
                    if scci.cam_id == cam_id:
                        break
                self.Pmat[cam_id] = scci.Pmat
                self.Res[cam_id] = scci.res
                self._helper[cam_id] = scci.helper
        elif self.cal_source_type in ["xml file", "json file"]:
            self.Pmat = next_self.Pmat
            self.Res = next_self.Res
            self._helper = next_self._helper
            wateri = next_self.wateri

        self.pmat_inv = {}
        for cam_id in cam_ids:
            # For speed reasons, make sure self.Pmat has only numpy arrays.
            self.Pmat[cam_id] = numpy.array(self.Pmat[cam_id])
            self.pmat_inv[cam_id] = numpy.linalg.pinv(self.Pmat[cam_id])

        self.cam_ids = cam_ids

        self.add_water(wateri)

        self.cam_combinations = [s for s in setOfSubsets(cam_ids) if len(s) >= 2]

        def cmpfunc(a, b):
            if len(a) > len(b):
                return -1
            else:
                return 0

        # order camera combinations from most cameras to least
        if sys.version_info.major == 3:
            self.cam_combinations.sort(key=functools.cmp_to_key(cmpfunc))
        else:
            # python2
            self.cam_combinations.sort(cmpfunc)
        self.cam_combinations_by_size = {}
        for cc in self.cam_combinations:
            self.cam_combinations_by_size.setdefault(len(cc), []).append(cc)
        # fill self._cam_centers_cache
        self._cam_centers_cache = {}
        for cam_id in self.cam_ids:
            self._cam_centers_cache[cam_id] = self.get_camera_center(cam_id)[
                :, 0
            ]  # make rank-1

        if close_cal_source:
            use_cal_source.close()

    def add_water(self, wateri):
        self.wateri = wateri
        if self.wateri is not None:
            assert isinstance(self.wateri, water.WaterInterface)

        self._model_with_jacobian = {}

        for cam_id in self.cam_ids:
            if self.wateri is None:
                model = _pmat_jacobian.PinholeCameraModelWithJacobian(self.Pmat[cam_id])
            else:
                model = _pmat_jacobian_water.PinholeCameraWaterModelWithJacobian(
                    self.Pmat[cam_id], self.wateri, WATER_ROOTS_EPS
                )
            self._model_with_jacobian[cam_id] = model

    @classmethod
    def from_pymvg(cls, mvg):
        d = mvg.get_camera_dict()
        sccs = [SingleCameraCalibration.from_pymvg(d[k]) for k in d]
        result = cls(sccs)
        return result

    def convert_to_pymvg(self, ignore_water=False):
        from pymvg.multi_camera_system import MultiCameraSystem

        if not ignore_water and (self.wateri is not None):
            raise ValueError(
                "Without dropping refractive boundary, this "
                "Reconstructor cannot be converted to PyMVG."
            )

        orig_sccs = [
            self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids
        ]
        cams = [o.convert_to_pymvg() for o in orig_sccs]
        result = MultiCameraSystem(cams)
        return result

    def get_scaled(self):
        """return a copy of self. (DEPRECATED.)"""
        warnings.warn(
            "reconstruct.Reconstructor.get_scaled() is deprecated. "
            "It is maintained only for backwards-compatibility.",
            DeprecationWarning,
        )
        return self.get_copy()

    def get_copy(self):
        orig_sccs = [
            self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids
        ]
        aligned_sccs = [scc.get_copy() for scc in orig_sccs]
        result = Reconstructor(
            aligned_sccs, minimum_eccentricity=self.minimum_eccentricity
        )
        result.add_water(self.wateri)
        return result

    def get_aligned_copy(self, M, update_water_boundary=True):
        orig_sccs = [
            self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids
        ]
        aligned_sccs = [scc.get_aligned_copy(M) for scc in orig_sccs]
        result = Reconstructor(
            aligned_sccs, minimum_eccentricity=self.minimum_eccentricity
        )
        wateri = self.wateri
        if wateri is not None:
            if update_water_boundary:
                raise ValueError("cannot update water boundary")
        result.add_water(wateri)
        return result

    def set_minimum_eccentricity(self, minimum_eccentricity):
        self.minimum_eccentricity = minimum_eccentricity

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        orig_sccs = [
            self.get_SingleCameraCalibration(cam_id) for cam_id in self.cam_ids
        ]
        other_sccs = [
            other.get_SingleCameraCalibration(cam_id) for cam_id in other.cam_ids
        ]
        eq = True
        for my_scc, other_scc in zip(orig_sccs, other_sccs):
            if my_scc != other_scc:
                if int(os.environ.get("DEBUG_EQ", "0")):
                    for attr in ["cam_id", "Pmat", "res", "helper"]:
                        print()
                        print("attr", attr)
                        print("my_scc", getattr(my_scc, attr))
                        print("other_scc", getattr(other_scc, attr))
                eq = False
                break
        return eq

    def get_3D_plane_and_ray(self, cam_id, *args, **kwargs):
        s = self.get_SingleCameraCalibration(cam_id)
        return s.get_3D_plane_and_ray(*args, **kwargs)

    def get_extrinsic_parameter_matrix(self, cam_id):
        scc = self.get_SingleCameraCalibration(cam_id)
        return scc.get_extrinsic_parameter_matrix()

    def get_cam_ids(self):
        return self.cam_ids

    def save_to_files_in_new_directory(self, new_dirname):
        if os.path.exists(new_dirname):
            raise RuntimeError('directory "%s" already exists' % new_dirname)
        os.mkdir(new_dirname)

        fd = open(os.path.join(new_dirname, "camera_order.txt"), "w")
        for cam_id in self.cam_ids:
            fd.write(cam_id + "\n")
        fd.close()

        fd = open(os.path.join(new_dirname, "minimum_eccentricity.txt"), "w")
        fd.write(repr(self.minimum_eccentricity) + "\n")
        fd.close()

        res_fd = open(os.path.join(new_dirname, "Res.dat"), "w")
        for cam_id in self.cam_ids:
            res_fd.write(" ".join(map(str, self.Res[cam_id])) + "\n")
        res_fd.close()

        for i, cam_id in enumerate(self.cam_ids):
            fname = "camera%d.Pmat.cal" % (i + 1)
            pmat_fd = open(os.path.join(new_dirname, fname), "w")
            save_ascii_matrix(self.Pmat[cam_id], pmat_fd)
            pmat_fd.close()

        # non linear parameters
        for i, cam_id in enumerate(self.cam_ids):
            fname = "basename%d.rad" % (i + 1)
            self._helper[cam_id].save_to_rad_file(os.path.join(new_dirname, fname))

    def save_to_xml_filename(self, xml_filename):
        root = ET.Element("root")
        self.add_element(root)
        child = root[0]
        result = pretty_dump(child, ind="  ")
        fd = open(xml_filename, mode="w")
        fd.write(result)
        fd.close()

    def as_obj_for_json(self):
        result = dict(cameras=[], minimum_eccentricity=self.minimum_eccentricity)
        for cam_id in self.cam_ids:
            scc = self.get_SingleCameraCalibration(cam_id)
            result["cameras"].append(scc.as_obj_for_json())
        return result

    def save_to_h5file(self, h5file, OK_to_delete_old_calibration=False):
        """create groups with calibration information"""

        import tables as PT  # pytables

        class AdditionalInfo(PT.IsDescription):
            cal_source_type = PT.StringCol(20)
            cal_source = PT.StringCol(80)
            minimum_eccentricity = (
                PT.Float32Col()
            )  # record what parameter was used during reconstruction

        class RefractiveInterfaces(PT.IsDescription):
            name = PT.StringCol(20)
            n1 = PT.Float64Col()
            n2 = PT.Float64Col()
            plane_normal_x = PT.Float64Col()
            plane_normal_y = PT.Float64Col()
            plane_normal_z = PT.Float64Col()
            plane_dist_from_origin = PT.Float64Col()

        pytables_filt = numpy.asarray
        ct = h5file.create_table  # shorthand
        root = h5file.root  # shorthand

        cam_ids = list(self.Pmat.keys())
        cam_ids.sort()

        if hasattr(root, "calibration"):
            if OK_to_delete_old_calibration:
                h5file.remove_node(root.calibration, recursive=True)
            else:
                raise RuntimeError("not deleting old calibration.")

        cal_group = h5file.create_group(root, "calibration")

        pmat_group = h5file.create_group(cal_group, "pmat")
        for cam_id in cam_ids:
            h5file.create_array(
                pmat_group, cam_id, pytables_filt(self.get_pmat(cam_id))
            )
        res_group = h5file.create_group(cal_group, "resolution")
        for cam_id in cam_ids:
            res = self.get_resolution(cam_id)
            h5file.create_array(res_group, cam_id, pytables_filt(res))

        intlin_group = h5file.create_group(cal_group, "intrinsic_linear")
        for cam_id in cam_ids:
            intlin = self.get_intrinsic_linear(cam_id)
            h5file.create_array(intlin_group, cam_id, pytables_filt(intlin))

        intnonlin_group = h5file.create_group(cal_group, "intrinsic_nonlinear")
        for cam_id in cam_ids:
            h5file.create_array(
                intnonlin_group,
                cam_id,
                pytables_filt(self.get_intrinsic_nonlinear(cam_id)),
            )

        h5additional_info = ct(cal_group, "additional_info", AdditionalInfo, "")
        row = h5additional_info.row
        row["cal_source_type"] = self.cal_source_type
        if isinstance(self.cal_source, list):
            row["cal_source"] = "(originally was list - not saved here)"
        else:
            if not isinstance(self.cal_source, PT.File):
                row["cal_source"] = self.cal_source
            else:
                row["cal_source"] = self.cal_source.filename
        row["minimum_eccentricity"] = self.minimum_eccentricity
        row.append()
        h5additional_info.flush()

        if self.wateri:
            h5refractive_interfaces = ct(
                cal_group, "refractive_interfaces", RefractiveInterfaces
            )
            row = h5refractive_interfaces.row
            row["name"] = "water1"
            row["n1"] = self.wateri.n1
            row["n2"] = self.wateri.n2
            row["plane_normal_x"] = 0
            row["plane_normal_y"] = 0
            row["plane_normal_z"] = 1
            row["plane_dist_from_origin"] = 0
            row.append()
            h5refractive_interfaces.flush()

    def get_resolution(self, cam_id):
        return self.Res[cam_id]

    def get_pmat(self, cam_id):
        return self.Pmat[cam_id]

    def get_pmat_inv(self, cam_id):
        return self.pmat_inv[cam_id]

    def get_model_with_jacobian(self, cam_id):
        return self._model_with_jacobian[cam_id]

    def get_camera_center(self, cam_id):
        # should be called get_cam_center?
        return pmat2cam_center(self.Pmat[cam_id])

    def get_intrinsic_linear(self, cam_id):
        return self._helper[cam_id].get_K()

    def get_intrinsic_nonlinear(self, cam_id):
        return self._helper[cam_id].get_nlparams()

    def undistort(self, cam_id, x_kk):
        return self._helper[cam_id].undistort(x_kk[0], x_kk[1])

    def distort(self, cam_id, xl):
        return self._helper[cam_id].distort(xl[0], xl[1])

    def get_reconstruct_helper_dict(self):
        return self._helper

    def find3d(
        self,
        cam_ids_and_points2d,
        return_X_coords=True,
        return_line_coords=True,
        orientation_consensus=0,
        undistort=False,
        simulate_via_tracking_dynamic_model=None,
    ):
        """Find 3D coordinate using all data given

        Implements a linear triangulation method to find a 3D
        point. For example, see Hartley & Zisserman section 12.2
        (p.312). Also, 12.8 for intersecting lines.

        Note: For a function which does hyptothesis testing to
        selectively choose 2D to incorporate, see
        hypothesis_testing_algorithm__find_best_3d() in
        reconstruct_utils.

        This function can optionally undistort points.

        """
        if simulate_via_tracking_dynamic_model is not None:
            # simulate via tracking uses flydra tracking framework to
            # iteratively attempt to find location
            assert return_X_coords == True
            assert return_line_coords == False

            start_guess = self.find3d(
                cam_ids_and_points2d,
                return_X_coords=return_X_coords,
                return_line_coords=return_line_coords,
            )

            import flydra_core.kalman.flydra_tracker

            frame = 0
            tro = flydra_core.kalman.flydra_tracker.TrackedObject(
                self,
                0,
                frame,
                start_guess,
                None,
                [],
                [],
                kalman_model=simulate_via_tracking_dynamic_model,
                disable_image_stat_gating=True,
            )

            # Create data structures as used by TrackedObject. Ohh,
            # this is so ugly. Sniff. :(

            camn2cam_id = {}
            cam_id2camn = {}
            data_dict = {}
            next_camn = 0
            for cam_id, pt2d in cam_ids_and_points2d:
                if cam_id not in cam_id2camn:
                    cam_id2camn[cam_id] = next_camn
                    camn2cam_id[next_camn] = cam_id
                    next_camn += 1
                camn = cam_id2camn[cam_id]
                if undistort:
                    x_undistorted, y_undistorted = self.undistort(cam_id, pt2d)
                else:
                    x_undistorted, y_undistorted = pt2d[:2]
                rise = np.nan
                run = np.nan
                area = 100.0
                slope = np.nan
                eccentricity = 10.0
                (
                    p1,
                    p2,
                    p3,
                    p4,
                    ray0,
                    ray1,
                    ray2,
                    ray3,
                    ray4,
                    ray5,
                ) = do_3d_operations_on_2d_point(
                    self._helper,
                    x_undistorted,
                    y_undistorted,
                    self.pmat_inv[cam_id],
                    self.get_camera_center(cam_id),
                    pt2d[0],
                    pt2d[1],
                    rise,
                    run,
                )
                frame_pt_idx = 0
                line_found = False
                cur_val = None
                mean_val = None
                sumsqf_val = None
                pt_undistorted = (
                    x_undistorted,
                    y_undistorted,
                    area,
                    slope,
                    eccentricity,
                    p1,
                    p2,
                    p3,
                    p4,
                    line_found,
                    frame_pt_idx,
                    cur_val,
                    mean_val,
                    sumsqf_val,
                )

                pluecker_hz = (ray0, ray1, ray2, ray3, ray4, ray5)
                projected_line = fastgeom.line_from_HZline(pluecker_hz)
                data_dict[camn] = [(pt_undistorted, projected_line)]

            delta_dist = 1.0
            eps = 1e-3
            max_count = 100
            while delta_dist > eps:
                if frame > max_count:
                    warnings.warn("exceeded max count")
                    break
                frame += 1
                tro.calculate_a_posteriori_estimate(frame, data_dict, camn2cam_id)
                vals = tro.xhats
                last_two = vals[-2:]
                if len(last_two) == 2:
                    prev, cur = last_two
                    delta_dist = np.sqrt(np.sum((prev[:3] - cur[:3]) ** 2))
            return cur[:3]

        if self.wateri is not None and return_line_coords:
            raise NotImplementedError()

        svd = scipy.linalg.svd
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)

        # Construct matrices
        A = []
        P = []
        PA = []
        PB = []
        for m, (cam_id, value_tuple) in enumerate(cam_ids_and_points2d):
            if len(value_tuple) == 2:
                # only point information ( no line )
                x, y = value_tuple
                if undistort:
                    x, y = self.undistort(cam_id, (x, y))
                have_line_coords = False
                if return_line_coords:
                    raise ValueError(
                        "requesting 3D line coordinates, but no 2D line coordinates given"
                    )
            else:
                if undistort:
                    raise ValueError("Undistoring line coords not implemented")
                # get shape information from each view of a blob:
                x, y, area = value_tuple[:3]
                have_line_coords = True

            if return_X_coords:
                if self.wateri is not None:
                    # Per camera:
                    #   Get point on water surface in this direction.
                    #   Get underwater line from water surface (using Snell's Law).
                    # All lines -> closest point to all lines.

                    # Get point on water surface in this direction.
                    pluecker_hz = self.get_projected_line_from_2d(cam_id, (x, y))
                    projected_line = slowgeom.line_from_HZline(pluecker_hz)
                    surface_pt_g = projected_line.intersect(Z0_plane)
                    surface_pt = surface_pt_g.vals
                    pt1 = self.get_camera_center(cam_id)[:, 0]
                    pt2 = np.array(pt1, copy=True)
                    pt2[
                        2
                    ] = 0  # closest point to camera on water surface, assumes water at z==0
                    surface_pt_cam = surface_pt - pt2

                    # Get underwater line from water surface (using Snell's Law).
                    pt_angle = np.arctan2(
                        surface_pt_cam[1], surface_pt_cam[0]
                    )  # OK to here
                    pt_horiz_dist = np.sqrt(
                        np.sum(surface_pt_cam ** 2)
                    )  # horizontal distance from camera to water surface
                    theta_air = np.arctan2(pt_horiz_dist, pt1[2])
                    # sin(theta_water)/sin(theta_air) = n_air/n_water
                    sin_theta_water = (
                        np.sin(theta_air) * self.wateri.n1 / self.wateri.n2
                    )
                    theta_water = np.arcsin(sin_theta_water)
                    horiz_dist_at_depth_1 = np.tan(theta_water)
                    horiz_dist_cam_depth_1 = (
                        horiz_dist_at_depth_1 + pt_horiz_dist
                    )  # total horizontal distance
                    deep_pt_cam = (
                        horiz_dist_cam_depth_1 * np.cos(pt_angle),
                        horiz_dist_cam_depth_1 * np.sin(pt_angle),
                        -1,
                    )
                    deep_pt = deep_pt_cam + pt2

                    PA.append(surface_pt)
                    PB.append(deep_pt)

                    # See http://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines
                else:
                    # Similar to code in
                    # _reconstruct_utils.hypothesis_testing_algorithm__find_best_3d()
                    Pmat = self.Pmat[cam_id]  # Pmat is 3 rows x 4 columns
                    row3 = Pmat[2, :]
                    A.append(x * row3 - Pmat[0, :])
                    A.append(y * row3 - Pmat[1, :])

            if return_line_coords and have_line_coords:
                slope, eccentricity, p1, p2, p3, p4 = value_tuple[3:]
                if (
                    eccentricity > self.minimum_eccentricity
                ):  # require a bit of elongation
                    P.append((p1, p2, p3, p4))

        # Calculate best point
        if return_X_coords:
            if self.wateri is not None:
                PA = np.array(PA)
                PB = np.array(PB)
                X = lineIntersect3D(PA, PB)
                return X
            else:
                A = nx.array(A)
                u, d, vt = svd(A)
                X = vt[-1, 0:3] / vt[-1, 3]  # normalize
                if not return_line_coords:
                    return X

        if not return_line_coords or len(P) < 2:
            Lcoords = None
        else:
            if orientation_consensus < 0:
                orientation_consensus = (
                    len(cam_ids_and_points2d) + orientation_consensus
                )

            if orientation_consensus != 0 and len(P) > orientation_consensus:
                # test for better fit with fewer orientations
                allps = [
                    np.array(ps)
                    for ps in setOfSubsets(P)
                    if len(ps) == orientation_consensus
                ]
                all_Lcoords = []
                for ps in allps:
                    all_Lcoords.append(intersect_planes_to_find_line(ps))
                N = len(all_Lcoords)
                distance_matrix = np.zeros((N, N))
                for i in range(N):
                    for j in range(i + 1, N):
                        d = line_direction_distance(all_Lcoords[i], all_Lcoords[j])
                        # distances are symmetric
                        distance_matrix[i, j] = d
                        distance_matrix[j, i] = d
                sum_distances = np.sum(distance_matrix, axis=0)
                closest_to_neighbors_idx = np.argmin(sum_distances)
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

    def find2d(self, cam_id, X, Lcoords=None, distorted=False, bypass_refraction=False):
        """
        find projection of 3D points in X onto 2D image plan for cam_id

        for rank1 case:
          X : shape==(3,) single point, or shape==(4,) homogeneous single point
          returns shape==(2,) projection
        else:
          X : (N,4) array of homogeneous points
          returns shape==(2,N) projection

        """

        # see Hartley & Zisserman (2003) p. 449
        X = np.array(X)
        rank1 = X.ndim == 1
        if rank1:
            # make homogenous coords, rank2
            if len(X) == 3:
                X = nx.array([[X[0]], [X[1]], [X[2]], [1.0]])
            else:
                X = X[:, nx.newaxis]  # 4 rows, 1 column
        else:
            assert X.shape[1] == 4
            X = X.T  # 4 rows, N columns

        N_points = X.shape[1]

        underwater_cond = None
        if self.wateri is not None and not bypass_refraction:
            if Lcoords is not None:
                raise NotImplementedError()

            pts3d = X.T
            if 1:
                w = pts3d[:, 3]
                assert np.allclose(w, np.ones_like(w))

            pts3d = pts3d[:, :3]
            underwater_cond = pts3d[:, 2] < 0
            underwater_pts = pts3d[underwater_cond]

            if len(underwater_pts):
                x_underwater = water.view_points_in_water(
                    self, cam_id, underwater_pts, self.wateri, distorted=distorted
                )
            else:
                x_underwater = None

        if underwater_cond is not None:
            X_nowater = X[:, ~underwater_cond]
        else:
            X_nowater = X

        N_pts_nowater = X_nowater.shape[1]
        if N_pts_nowater:
            Pmat = self.Pmat[cam_id]
            x_nowater = nx.dot(Pmat, X_nowater)

            x_nowater = x_nowater[0:2, :] / x_nowater[2, :]  # normalize

            if distorted:
                if rank1:
                    xd, yd = self.distort(cam_id, x_nowater)
                    x_nowater[0] = xd
                    x_nowater[1] = yd
                else:
                    N_pts = x_nowater.shape[1]
                    for i in range(N_pts):
                        xpt = x_nowater[:, i]
                        xd, yd = self.distort(cam_id, xpt)
                        x_nowater[0, i] = xd
                        x_nowater[1, i] = yd

        if underwater_cond is not None:
            x = np.empty((2, N_points), dtype=np.float64)
            if x_underwater is not None:
                x[:, underwater_cond] = x_underwater
            if N_pts_nowater:
                x[:, ~underwater_cond] = x_nowater
        else:
            x = x_nowater

        assert x.shape[1] == N_points

        # XXX The rest of this function hasn't been (recently) checked
        # for >1 points. (i.e. not rank1)

        if Lcoords is not None:
            if distorted:

                # Impossible to distort Lcoords. The image of the line
                # could be distorted downstream.

                raise RuntimeError("cannot (easily) distort line")

            if not rank1:
                raise NotImplementedError(
                    "Line reconstruction not yet implemented for rank-2 data"
                )

            # see Hartley & Zisserman (2003) p. 198, eqn 8.2
            L = Lcoords2Lmatrix(Lcoords)
            # XXX could be made faster by pre-computing line projection matrix
            lx = nx.dot(Pmat, nx.dot(L, nx.transpose(Pmat)))
            l3 = lx[2, 1], lx[0, 2], lx[1, 0]  # (p. 581)
            return x, l3
        else:
            if rank1:
                # convert back to rank1
                return x[:, 0]
            else:
                return x

    def find3d_single_cam(self, cam_id, x):
        """see also SingleCameraCalibration.get_example_3d_point_creating_image_point()"""
        return nx.dot(self.pmat_inv[cam_id], as_column(x))

    def get_projected_line_from_2d(self, cam_id, xy):
        """project undistorted points into pluecker line

        see also SingleCameraCalibration.get_example_3d_point_creating_image_point()"""
        # XXX Could use nullspace method of back-projection of lines (that pass through camera center, HZ section 8.1)
        # image of 2d point in 3d space (on optical ray)
        XY = self.find3d_single_cam(cam_id, (xy[0], xy[1], 1.0))
        XY = XY[:3, 0] / XY[3]  # convert to rank1
        C = self._cam_centers_cache[cam_id]
        return pluecker_from_verts(XY, C)

    def get_SingleCameraCalibration(self, cam_id):
        return SingleCameraCalibration(
            cam_id=cam_id,
            Pmat=self.Pmat[cam_id],
            res=self.Res[cam_id],
            helper=self._helper[cam_id],
        )

    def get_distorted_line_segments(self, cam_id, line3d):
        dummy = [0, 0, 0]  # dummy 3D coordinate

        # project 3d line into projected 2d line
        dummy2d, proj = self.find2d(cam_id, dummy, line3d)

        # now distort 2d line into 2d line segments

        # calculate undistorted 2d line segments

        # line at x = -100
        l = numpy.array([1, 0, 100])

        # line at x = 1000
        r = numpy.array([1, 0, -1000])

        lh = numpy.cross(proj, l)
        rh = numpy.cross(proj, r)

        if lh[2] == 0 or rh[2] == 0:
            if 1:
                raise NotImplementedError("cannot deal with exactly vertical lines")
            b = numpy.array([0, 1, 100])
            t = numpy.array([0, 1, -1000])
            bh = numpy.cross(proj, b)
            th = numpy.cross(proj, t)

        x0 = lh[0] / lh[2]
        y0 = lh[1] / lh[2]

        x1 = rh[0] / rh[2]
        y1 = rh[1] / rh[2]

        dy = y1 - y0
        dx = x1 - x0
        n_pts = 10000
        frac = numpy.linspace(0, 1, n_pts)
        xs = x0 + frac * dx
        ys = y0 + frac * dy

        # distort 2d segments
        xs_d = []
        ys_d = []
        for xy in zip(xs, ys):
            x_distorted, y_distorted = self.distort(cam_id, xy)
            if -100 <= x_distorted <= 800 and -100 <= y_distorted <= 800:
                xs_d.append(x_distorted)
                ys_d.append(y_distorted)
        return xs_d, ys_d

    def add_element(self, parent):
        """add self as XML element to parent"""
        assert ET.iselement(parent)
        elem = ET.SubElement(parent, "multi_camera_reconstructor")
        for cam_id in self.cam_ids:
            scc = self.get_SingleCameraCalibration(cam_id)
            scc.add_element(elem)
        with np.printoptions(legacy='1.25'):
            if 1:
                me_elem = ET.SubElement(elem, "minimum_eccentricity")
                me_elem.text = repr(self.minimum_eccentricity)
            if self.wateri is not None:
                water_elem = ET.SubElement(elem, "water")
                water_elem.text = repr(self.wateri.n2)


def align_calibration():
    usage = """%prog [options]
     Creates a new reconstructor from an existing reconstructor and a new
     description of camera centres (such as another reconstructor, a raw
     alignment file, or an explicit list of camera centres).

     Raw alignment files supported include json (--align-json)
     and python syntax (--align-raw, the file is exec'd).
     The file should have contents like the following,
     where s is scale, R is a 3x3 rotation matrix, and t is a 3 vector
     specifying translation::

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

    parser.add_option(
        "--orig-reconstructor",
        type="string",
        help="calibration/reconstructor path (required)",
    )

    parser.add_option(
        "--dest-dir", type="string", help="output calibration/reconstructor path"
    )

    parser.add_option("--align-raw", type="string", help="raw alignment file path")

    parser.add_option("--align-json", type="string", help=".json alignment file path")

    parser.add_option(
        "--align-cams", type="string", help="new camera locations alignment file path"
    )

    parser.add_option(
        "--align-reconstructor",
        type="string",
        help="reconstructor with desired camera locations",
    )

    parser.add_option(
        "--output-xml",
        action="store_true",
        default=False,
        help="save the new reconstructor in xml format",
    )

    (options, args) = parser.parse_args()

    if options.orig_reconstructor is None:
        raise ValueError("--orig-reconstructor must be specified")

    if (
        options.align_raw is None
        and options.align_cams is None
        and options.align_json is None
        and options.align_reconstructor is None
    ):
        raise ValueError("an --align-XXX method must be specified")

    src = options.orig_reconstructor
    origR = Reconstructor(cal_source=src)

    if options.dest_dir is None:
        dst = src + ".aligned"
        if options.output_xml:
            dst += ".xml"
    else:
        if options.output_xml:
            raise ValueError("cannot specify both --dest-dir and --output-xml")
        dst = options.dest_dir
    if os.path.exists(dst):
        raise RuntimeError("destination %s exists" % dst)

    srcR = origR

    except_cams = []

    import flydra_core.align as align

    if options.align_raw is not None:
        mylocals = {}
        myglobals = {}
        execfile(options.align_raw, myglobals, mylocals)
        s = mylocals["s"]
        R = np.array(mylocals["R"])
        t = np.array(mylocals["t"])
    elif options.align_cams is not None:
        cam_ids = srcR.get_cam_ids()
        ccs = [srcR.get_camera_center(cam_id)[:, 0] for cam_id in cam_ids]
        print("ccs", ccs)
        orig_cam_centers = np.array(ccs).T
        print("orig_cam_centers", orig_cam_centers.T)
        new_cam_centers = np.loadtxt(options.align_cams).T
        print("new_cam_centers", new_cam_centers.T)
        s, R, t = align.estsimt(orig_cam_centers, new_cam_centers)
    elif options.align_reconstructor is not None:
        cam_ids = srcR.get_cam_ids()
        print(cam_ids)
        ccs = [
            srcR.get_camera_center(cam_id)[:, 0]
            for cam_id in cam_ids
            if cam_id not in except_cams
        ]
        print("ccs", ccs)
        orig_cam_centers = np.array(ccs).T
        print("orig_cam_centers", orig_cam_centers.T)
        tmpR = Reconstructor(cal_source=options.align_reconstructor)
        nccs = [
            tmpR.get_camera_center(cam_id)[:, 0]
            for cam_id in cam_ids
            if cam_id not in except_cams
        ]
        new_cam_centers = np.array(nccs).T
        print("new_cam_centers", new_cam_centers.T)
        s, R, t = align.estsimt(orig_cam_centers, new_cam_centers)
    elif options.align_json is not None:
        import json

        with open(options.align_json, mode="r") as fd:
            buf = fd.read()
        mylocals = json.loads(buf)
        s = mylocals["s"]
        R = np.array(mylocals["R"])
        t = np.array(mylocals["t"])
    else:
        raise Exception("Alignment not supported")

    print("s", s)
    print("R", R)
    print("t", t)

    M = align.build_xform(s, R, t)

    alignedR = srcR.get_aligned_copy(M)
    if options.output_xml:
        alignedR.save_to_xml_filename(dst)
    else:
        alignedR.save_to_files_in_new_directory(dst)

    print("wrote", dst)


def print_cam_centers():
    filename = sys.argv[1]
    R = flydra_core.reconstruct.Reconstructor(filename)
    for cam_id in R.cam_ids:
        print(cam_id, R.get_camera_center(cam_id)[:, 0])


def flip_calibration():
    import pymvg.multi_camera_system

    filename = sys.argv[1]
    R = flydra_core.reconstruct.Reconstructor(filename)
    if R.wateri:
        warnings.warn("ignoring water in reconstructor")
    system = R.convert_to_pymvg(ignore_water=True)
    new_cams = []
    for cam_name in system.get_camera_dict():
        new_cams.append(system.get_camera(cam_name).get_flipped_camera())
    flipped = pymvg.multi_camera_system.MultiCameraSystem(new_cams)
    Rflipped = Reconstructor.from_pymvg(flipped)

    flipped_fname = os.path.splitext(filename)[0] + "-flipped.xml"
    Rflipped.save_to_xml_filename(flipped_fname)


if __name__ == "__main__":
    test()
