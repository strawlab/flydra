from __future__ import division
import traits.api as traits
from traitsui.api import View, Item, Group
import cgtypes  # import cgkit 1.x
import numpy as np

D2R = np.pi / 180.0


def cgmat2np(cgkit_mat):
    """convert cgkit matrix to numpy matrix"""
    arr = np.array(cgkit_mat.toList())
    if len(arr) == 9:
        arr.shape = 3, 3
    elif len(arr) == 16:
        arr.shape = 4, 4
    else:
        raise ValueError("unknown shape")
    return arr.T


def test_cgmat2mp():
    point1 = (1, 0, 0)
    point1_out = (0, 1, 0)

    cg_quat = cgtypes.quat().fromAngleAxis(90.0 * D2R, (0, 0, 1))
    cg_in = cgtypes.vec3(point1)

    m_cg = cg_quat.toMat3()
    cg_out = m_cg * cg_in
    cg_out_tup = (cg_out[0], cg_out[1], cg_out[2])
    assert np.allclose(cg_out_tup, point1_out)

    m_np = cgmat2np(m_cg)
    np_out = np.dot(m_np, point1)
    assert np.allclose(np_out, point1_out)


class Alignment(traits.HasTraits):
    s = traits.Float(1.0)
    tx = traits.Float(0)
    ty = traits.Float(0)
    tz = traits.Float(0)

    r_x = traits.Range(-180.0, 180.0, 0.0, mode="slider", set_enter=True)
    r_y = traits.Range(-180.0, 180.0, 0.0, mode="slider", set_enter=True)
    r_z = traits.Range(-180.0, 180.0, 0.0, mode="slider", set_enter=True)

    flip_x = traits.Bool(False)
    flip_y = traits.Bool(False)
    flip_z = traits.Bool(False)

    traits_view = View(
        Group(
            (
                Item("s"),
                Item("tx"),
                Item("ty"),
                Item("tz"),
                Item("r_x", style="custom"),
                Item("r_y", style="custom"),
                Item("r_z", style="custom"),
                Item("flip_x"),
                Item("flip_y"),
                Item("flip_z"),
            )
        ),
        title="Alignment Parameters",
    )

    def get_matrix(self):
        qx = cgtypes.quat().fromAngleAxis(self.r_x * D2R, cgtypes.vec3(1, 0, 0))
        qy = cgtypes.quat().fromAngleAxis(self.r_y * D2R, cgtypes.vec3(0, 1, 0))
        qz = cgtypes.quat().fromAngleAxis(self.r_z * D2R, cgtypes.vec3(0, 0, 1))
        Rx = cgmat2np(qx.toMat3())
        Ry = cgmat2np(qy.toMat3())
        Rz = cgmat2np(qz.toMat3())
        R = np.dot(Rx, np.dot(Ry, Rz))

        t = np.array([self.tx, self.ty, self.tz], np.float64)
        s = self.s

        T = np.zeros((4, 4), dtype=np.float64)
        T[:3, :3] = s * R
        T[:3, 3] = t
        T[3, 3] = 1.0

        # convert bool to -1 or 1
        fx = fy = fz = 1
        if self.flip_x:
            fx = -1
        if self.flip_y:
            fy = -1
        if self.flip_z:
            fz = -1

        flip = np.array(
            [[fx, 0, 0, 0], [0, fy, 0, 0], [0, 0, fz, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        T = np.dot(flip, T)
        # T = np.dot(T,flip)
        return T

    def as_dict(self):
        qx = cgtypes.quat().fromAngleAxis(self.r_x * D2R, cgtypes.vec3(1, 0, 0))
        qy = cgtypes.quat().fromAngleAxis(self.r_y * D2R, cgtypes.vec3(0, 1, 0))
        qz = cgtypes.quat().fromAngleAxis(self.r_z * D2R, cgtypes.vec3(0, 0, 1))
        Rx = cgmat2np(qx.toMat3())
        Ry = cgmat2np(qy.toMat3())
        Rz = cgmat2np(qz.toMat3())
        _R = np.dot(Rx, np.dot(Ry, Rz))

        # convert bool to -1 or 1
        fx = fy = fz = 1
        if self.flip_x:
            fx = -1
        if self.flip_y:
            fy = -1
        if self.flip_z:
            fz = -1

        flip = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, fz]], dtype=np.float64)
        _R = np.dot(flip, _R)

        s = float(self.s)
        t = map(float, [self.tx, self.ty, self.tz])
        R = []
        for row in _R:
            R.append(map(float, [i for i in row]))

        result = {"s": s, "t": t, "R": R}
        return result
