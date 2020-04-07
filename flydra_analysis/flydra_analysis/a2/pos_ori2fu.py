from __future__ import print_function
import cgtypes  # cgtypes 1.x
import numpy as np

# See math in kdviewer.py (annotatePick function)

D2R = np.pi / 180.0


def rotate(q, p):
    if isinstance(p, cgtypes.vec3):
        p = cgtypes.quat((0, p[0], p[1], p[2]))
    r = q * p * q.inverse()
    result = cgtypes.vec3((r.x, r.y, r.z))
    return result


def pos_ori2fu(pos, ori):
    """convert position (xyz) vector and orientation (wxyz) quat into focal_point and view_up"""
    pos = cgtypes.vec3(pos)
    ori = cgtypes.quat(ori)
    if np.isnan(ori.w):
        raise ValueError("ori is nan")
    forward = cgtypes.vec3((0, 0, -1))
    up = cgtypes.vec3((0, 1, 0))
    mori = ori.toMat4().inverse()
    # view_forward_dir = rotate(ori,forward)
    view_forward_dir = mori * forward
    focal_point = pos + view_forward_dir
    if np.isnan(focal_point[0]):
        raise ValueError("focal point is nan")
    # view_up_dir = rotate(ori,up)
    view_up_dir = mori * up
    return focal_point, view_up_dir


def check_close(a, b):
    print(a, "?=", b)
    return np.allclose(np.asarray(a), np.asarray(b))


def test():

    if 1:
        pos = cgtypes.vec3(0, 0, 0)
        forward = cgtypes.vec3(0, 0, -1)
        up = cgtypes.vec3(0, 1, 0)

        q = cgtypes.quat().fromAngleAxis(0, cgtypes.vec3(1, 0, 0))
        fp = rotate(q, forward)
        vu = rotate(q, up)
        assert check_close(fp, forward)
        assert check_close(vu, up)

        q = cgtypes.quat().fromAngleAxis(np.pi, cgtypes.vec3(1, 0, 0))
        fp = rotate(q, forward)
        vu = rotate(q, up)
        assert check_close(fp, (0, 0, 1))
        assert check_close(vu, (0, -1, 0))

        q = cgtypes.quat().fromAngleAxis(np.pi / 2, cgtypes.vec3(0, 1, 0))
        q_mat = q.toMat4()
        fp = rotate(q, forward)
        vu = rotate(q, up)
        assert check_close(fp, (-1, 0, 0))
        assert check_close(vu, (0, 1, 0))

        fp2 = q_mat * forward
        assert check_close(fp, fp2)
        vu2 = q_mat * up
        assert check_close(vu, vu2)
        print("*" * 40)

    if 0:

        def test_rotate(q, x):
            print("x", x)
            r1 = rotate(q, x)
            qmat = q.toMat4()
            print("qmat")
            print(qmat)
            r2 = qmat * x
            assert check_close(r1, r2)
            print()

        q = cgtypes.quat().fromAngleAxis(
            46.891594344015928 * D2R,
            (-0.99933050325884276, -0.028458760896155278, 0.022992263583291334),
        )

        a = cgtypes.vec3(1, 0, 0)
        b = cgtypes.vec3(0, 1, 0)
        c = cgtypes.vec3(0, 0, 1)
        test_rotate(q, a)
        test_rotate(q, b)
        test_rotate(q, c)

    if 1:
        pos = (0.40262157300195972, 0.12141447782035097, 1.0)
        ori = cgtypes.quat().fromAngleAxis(0.0, (0.0, 0.0, 1.0))
        print("ori.toAngleAxis()", ori.toAngleAxis())
        print(ori)

        fp_good = np.array((0.40262157300195972, 0.12141447782035097, 0.0))
        vu_good = np.array((0, 1, 0))

        # fp,vu = pos_ori2fu(pos,ori)
        fp, vu = pos_ori2fu(pos, ori)
        print("fp_good", fp_good)
        print("fp", fp)
        print()
        print("vu_good", vu_good)
        print("vu", vu)
        assert check_close(fp, fp_good)
        assert check_close(vu, vu_good)

    if 1:

        def test_em(position, orientation, fp_good, vu_good):
            ori = cgtypes.quat().fromAngleAxis(
                orientation[0] * D2R, (orientation[1], orientation[2], orientation[3])
            )
            if 0:
                print("ori.toAngleAxis()", ori.toAngleAxis())
                print(ori)
                o2 = cgtypes.quat().fromAngleAxis(*(ori.toAngleAxis()))
                print(o2)

            fp, vu = pos_ori2fu(position, ori)

            # focal point direction is important, not FP itself...
            fpdir_good = np.array(fp_good) - position
            fpdir = np.array(fp) - position

            print("focal_point direction", end=" ")
            assert check_close(fpdir, fpdir_good)
            print("view_up", end=" ")
            assert check_close(vu, vu_good)
            print()

        # values from VTK's camera.position, camera.orientation_wxyz, camera.focal_point, camera.view_up
        position = (0.4272878670512405, -0.55393877027170979, 0.68354826464002871)
        orientation = (
            46.891594344015928,
            -0.99933050325884276,
            -0.028458760896155278,
            0.022992263583291334,
        )
        focal_point = (0.41378612268658882, 0.17584165753292449, 0.0)
        view_up = (0.025790333690774738, 0.68363731594647714, 0.72936608019129556)

        test_em(position, orientation, focal_point, view_up)


if __name__ == "__main__":
    test()
