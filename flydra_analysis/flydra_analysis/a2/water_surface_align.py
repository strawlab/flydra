#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from optparse import OptionParser
import numpy as np
import flydra_analysis.a2.core_analysis as core_analysis
import flydra_core.align as align
from flydra_core.reconstruct import Reconstructor, DEFAULT_WATER_REFRACTIVE_INDEX
import flydra_core.water as water
from . import ransac
import cgtypes  # cgkit 1.x
import os

from flydra_core.common_variables import WATER_ROOTS_EPS

svd = np.linalg.svd
D2R = np.pi / 180.0


def norm(vec):
    vlen = np.sqrt(np.sum(vec ** 2))
    return vec / vlen


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


def test_cgmat2np():
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


class PlaneModelHelper:
    def fit(self, data):
        # http://stackoverflow.com/a/10904220/1633026
        data = np.array(data, dtype=np.float64)
        assert data.ndim == 2
        assert data.shape[1] == 3
        nrows = len(data)
        G = np.empty((nrows, 4))
        G[:, :3] = data
        G[:, 3] = 1.0
        u, d, vt = svd(G, full_matrices=True)
        plane_model = vt[3, :]
        assert plane_model.ndim == 1
        assert plane_model.shape[0] == 4
        return plane_model

    def get_error(self, data, plane_model):
        # http://mathworld.wolfram.com/Point-PlaneDistance.html
        assert data.ndim == 2
        n_pts = data.shape[0]
        assert data.shape[1] == 3

        assert plane_model.ndim == 1
        assert plane_model.shape[0] == 4

        a, b, c, d = plane_model
        denom = np.sqrt(a ** 2 + b ** 2 + c ** 2)

        x0 = data[:, 0]
        y0 = data[:, 1]
        z0 = data[:, 2]
        numer = np.abs(a * x0 + b * y0 + c * z0 + d)
        all_distances = numer / denom
        assert all_distances.ndim == 1
        assert all_distances.shape[0] == n_pts
        return all_distances


def doit(
    filename=None, obj_only=None, do_ransac=False, show=False,
):
    # get original 3D points -------------------------------
    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(filename)
    if obj_only is not None:
        use_obj_ids = np.array(obj_only)

    x = []
    y = []
    z = []
    for obj_id in use_obj_ids:
        obs_rows = ca.load_dynamics_free_MLE_position(obj_id, data_file)
        goodcond = ~np.isnan(obs_rows["x"])
        good_rows = obs_rows[goodcond]
        x.append(good_rows["x"])
        y.append(good_rows["y"])
        z.append(good_rows["z"])
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)

    recon = Reconstructor(cal_source=data_file)
    extra["kresults"].close()  # close file

    data = np.empty((len(x), 3), dtype=np.float64)
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z

    # calculate plane-of-best fit ------------

    helper = PlaneModelHelper()
    if not do_ransac:
        plane_params = helper.fit(data)
    else:
        # do RANSAC

        """
        n: the minimum number of data values required to fit the model
        k: the maximum number of iterations allowed in the algorithm
        t: a threshold value for determining when a data point fits a model
        d: the number of close data values required to assert that a model fits well to data
        """
        n = 20
        k = 100
        t = np.mean([np.std(x), np.std(y), np.std(z)])
        d = 100
        plane_params = ransac.ransac(data, helper, n, k, t, d, debug=False)

    # Calculate rotation matrix from plane-of-best-fit to z==0 --------
    orig_normal = norm(plane_params[:3])
    new_normal = np.array([0, 0, 1], dtype=np.float64)
    rot_axis = norm(np.cross(orig_normal, new_normal))
    cos_angle = np.dot(orig_normal, new_normal)
    angle = np.arccos(cos_angle)
    q = cgtypes.quat().fromAngleAxis(angle, rot_axis)
    m = q.toMat3()
    R = cgmat2np(m)

    # Calculate aligned data without translation -----------------
    s = 1.0
    t = np.array([0, 0, 0], dtype=np.float64)

    aligned_data = align.align_points(s, R, t, data.T).T

    # Calculate aligned data so that mean point is origin -----------------
    t = -np.mean(aligned_data[:, :3], axis=0)
    aligned_data = align.align_points(s, R, t, data.T).T

    M = align.build_xform(s, R, t)
    r2 = recon.get_aligned_copy(M)
    wateri = water.WaterInterface(
        refractive_index=DEFAULT_WATER_REFRACTIVE_INDEX, water_roots_eps=WATER_ROOTS_EPS
    )
    r2.add_water(wateri)

    dst = os.path.splitext(filename)[0] + "-water-aligned.xml"
    r2.save_to_xml_filename(dst)
    print("saved to", dst)

    if show:
        import matplotlib.pyplot as plt
        from pymvg.plot_utils import plot_system
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax1.plot(data[:, 0], data[:, 1], "b.")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2 = fig.add_subplot(222)
        ax2.plot(data[:, 0], data[:, 2], "b.")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")

        ax3 = fig.add_subplot(223)
        ax3.plot(aligned_data[:, 0], aligned_data[:, 1], "b.")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

        ax4 = fig.add_subplot(224)
        ax4.plot(aligned_data[:, 0], aligned_data[:, 2], "b.")
        ax4.set_xlabel("x")
        ax4.set_ylabel("z")

        fig2 = plt.figure("cameras")
        ax = fig2.add_subplot(111, projection="3d")
        system = r2.convert_to_pymvg(ignore_water=True)
        plot_system(ax, system)
        x = np.linspace(-0.1, 0.1, 10)
        y = np.linspace(-0.1, 0.1, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot(X.ravel(), Y.ravel(), Z.ravel(), "b.")
        ax.set_title("aligned camera positions")

        plt.show()


def main():
    usage = "%prog FILE [options]"
    parser = OptionParser(usage)
    parser.add_option("--obj-only", type="string")
    parser.add_option("--ransac", action="store_true", default=False)
    parser.add_option("--show", action="store_true", default=False)
    (options, args) = parser.parse_args()

    h5_filename = args[0]

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    doit(
        filename=h5_filename,
        obj_only=options.obj_only,
        do_ransac=options.ransac,
        show=options.show,
    )


if __name__ == "__main__":
    main()
