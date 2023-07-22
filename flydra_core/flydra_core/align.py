from __future__ import print_function
import numpy as np
import scipy.linalg


def estsimt(X1, X2):
    # from estsimt.m in MultiCameSelfCal

    # ESTimate SIMilarity Transformation
    #
    # [s,R,T] = estsimt(X1,X2)
    #
    # X1,X2 ... 3xN matrices with corresponding 3D points
    #
    # X2 = s*R*X1 + T
    # s ... scalar scale
    # R ... 3x3 rotation matrix
    # T ... 3x1 translation vector
    #
    # This is done according to the paper:
    # "Least-Squares Fitting of Two 3-D Point Sets"
    # by K.S. Arun, T. S. Huang and S. D. Blostein

    N = X1.shape[1]
    if N != X2.shape[1]:
        raise ValueError("both X1 and X2 must have same number of points")

    X1cent = np.mean(X1, axis=1)
    X2cent = np.mean(X2, axis=1)
    # normalize coordinate systems for both set of points
    x1 = X1 - X1cent[:, np.newaxis]
    x2 = X2 - X2cent[:, np.newaxis]

    # mutual distances
    d1 = x1[:, 1:] - x1[:, :-1]
    d2 = x2[:, 1:] - x2[:, :-1]
    ds1 = np.sqrt(np.sum(d1 ** 2, axis=0))
    ds2 = np.sqrt(np.sum(d2 ** 2, axis=0))

    # print 'ds1'
    # print ds1

    scales = ds2 / ds1
    s = np.median(scales)

    # print 's', s

    # undo scale
    x1s = s * x1

    # finding rotation
    H = np.zeros((3, 3))
    for i in range(N):
        tmp1 = x1s[:, i, np.newaxis]
        # print 'tmp1',tmp1
        tmp2 = x2[np.newaxis, :, i]
        # print 'tmp2',tmp2
        tmp = np.dot(tmp1, tmp2)
        # print 'tmp'
        # print tmp
        H += tmp
        # print 'H'
        # print H

    U, S, Vt = scipy.linalg.svd(H)
    # print 'U'
    # print U
    # print 'S'
    # print S
    # print 'Vt'
    # print Vt
    V = Vt.T
    X = np.dot(V, U.T)
    R = X

    T = X2cent - s * np.dot(R, X1cent)
    return s, R, T


def build_xform(s, R, t):
    T = np.zeros((4, 4), dtype=np.float64)
    T[:3, :3] = R
    T = s * T
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T


def align_points(s, R, T, X):
    assert X.ndim == 2
    assert X.shape[0] in [3, 4]  # either 3D or 3D homogeneous
    T = build_xform(s, R, T)
    if X.shape[0] == 3:
        # make homogeneous
        Xnew = np.ndarray((4, X.shape[1]), dtype=X.dtype)
        Xnew[3, :].fill(1)
        Xnew[:3, :] = X
        X = Xnew
    X = np.dot(T, X)
    return X


def align_pmat(s, R, T, P):
    T = build_xform(s, R, T)
    P = np.dot(P, scipy.linalg.inv(T))
    return P


def align_pmat2(M, P):
    P = np.dot(P, scipy.linalg.inv(M))
    return P


def test_align():
    orig_points = np.array(
        [
            [3.36748406, 1.61036404, 3.55147255],
            [3.58702265, 0.06676394, 3.64695356],
            [0.28452026, -0.11188296, 3.78947735],
            [0.25482713, 1.57828256, 3.6900808],
            [3.54938525, 1.74057692, 5.13329681],
            [3.6855626, 0.10335229, 5.26344841],
            [0.25025385, -0.06146044, 5.57085135],
            [0.20742481, 1.71073272, 5.41823085],
        ]
    ).T

    ft2inch = 12.0
    inch2cm = 2.54
    cm2m = 0.01
    ft2m = ft2inch * inch2cm * cm2m

    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = np.array([10, 5, 5]) * ft2m

    new_points = np.array(
        [
            [x2, y2, z2],
            [x2, y1, z2],
            [x1, y1, z2],
            [x1, y2, z2],
            [x2, y2, z1],
            [x2, y1, z1],
            [x1, y1, z1],
            [x1, y2, z1],
        ]
    ).T

    print(orig_points.T)
    print(new_points.T)

    s, R, t = estsimt(orig_points, new_points)
    print("s=%s" % repr(s))
    print("R=%s" % repr(R.tolist()))
    print("t=%s" % repr(t.tolist()))
    Xnew = align_points(s, R, t, orig_points)

    # measure distance between elements
    mean_absdiff = np.mean(abs(Xnew[:3] - new_points).flatten())
    assert mean_absdiff < 0.05

    pmat_orig = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    print("Xnew.T")
    print(Xnew.T)

    pmat_new = align_pmat(s, R, t, pmat_orig)
    print("pmat_new")
    print(pmat_new)

    ## print 's',s
    ## print 'R'
    ## print R
    ## print 'T'
    ## print T
