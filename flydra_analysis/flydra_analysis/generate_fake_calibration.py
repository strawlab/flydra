from __future__ import division
from __future__ import print_function
import numpy
import numpy as np
import scipy.linalg
import flydra_core.reconstruct as reconstruct
import cgtypes  # cgkit 1.x
import os
import flydra_core._reconstruct_utils as reconstruct_utils

from flydra_analysis.analysis.flydra_analysis_generate_recalibration import (
    save_calibration_directory,
)


def generate_calibration(
    n_cameras=5, return_full_info=False, radial_distortion=False,
):
    pi = numpy.pi

    sccs = []

    # 1. extrinsic parameters:
    if 1:
        # method 1:
        #  arrange cameras in circle around common point
        common_point = numpy.array((0, 0, 0), dtype=numpy.float64)
        r = 10.0

        theta = numpy.linspace(0, 2 * pi, n_cameras, endpoint=False)
        x = numpy.cos(theta)
        y = numpy.sin(theta)
        z = numpy.zeros(y.shape)

        cc = numpy.c_[x, y, z]
        # cam_up = numpy.array((0,0,1))

        # cam_ups = numpy.resize(cam_up,cc.shape)
        # cam_forwads = -cc
        cam_centers = r * cc + common_point

        # Convert up/forward into rotation matrix.
        if 1:
            Rs = []
            for i, th in enumerate(theta):
                pos = cam_centers[i]
                target = common_point
                up = (0, 0, 1)
                if 0:
                    print("pos", pos)
                    print("target", target)
                    print("up", up)
                R = cgtypes.mat4().lookAt(pos, target, up)
                # print 'R4',R
                R = R.getMat3()
                # print 'R3',R
                R = numpy.asarray(R).T
                # print 'R',R
                # print
                Rs.append(R)

        else:
            # (Camera coords: looking forward -z, up +y, right +x)
            R = cgtypes.mat3().identity()

            if 1:
                # (looking forward -z, up +x, right -y)
                R = R.rotation(-pi / 2, (0, 0, 1))

                # (looking forward +x, up +z, right -y)
                R = R.rotation(-pi / 2, (0, 1, 0))

                # rotate to point -theta (with up +z)
                Rs = [R.rotation(float(th) + pi, (0, 0, 1)) for th in theta]
                # Rs = [ R for th in theta ]
            else:
                Rs = [R.rotation(pi / 2.0, (1, 0, 0)) for th in theta]
                # Rs = [ R for th in theta ]
            Rs = [numpy.asarray(R).T for R in Rs]
            print("Rs", Rs)

    # 2. intrinsic parameters

    resolutions = {}
    for cam_no in range(n_cameras):
        cam_id = "fake_%d" % (cam_no + 1)

        # resolution of image
        res = (1600, 1200)
        resolutions[cam_id] = res

        # principal point
        cc1 = res[0] / 2.0
        cc2 = res[1] / 2.0

        # focal length
        fc1 = 1.0
        fc2 = 1.0
        alpha_c = 0.0
        #        R = numpy.asarray(Rs[cam_no]).T # conversion between cgkit and numpy
        R = Rs[cam_no]
        C = cam_centers[cam_no][:, numpy.newaxis]

        K = numpy.array(((fc1, alpha_c * fc1, cc1), (0, fc2, cc2), (0, 0, 1)))
        t = numpy.dot(-R, C)
        Rt = numpy.concatenate((R, t), axis=1)
        P = numpy.dot(K, Rt)
        if 0:
            print("cam_id", cam_id)
            print("P")
            print(P)
            print("K")
            print(K)
            print("Rt")
            print(Rt)
            print()
            KR = numpy.dot(K, R)
            print("KR", KR)
            K3, R3 = reconstruct.my_rq(KR)
            print("K3")
            print(K3)
            print("R3")
            print(R3)
            K3R3 = numpy.dot(K3, R3)
            print("K3R3", K3R3)

            print("*" * 60)

        if radial_distortion:
            f = 1000.0
            r1 = 0.8
            r2 = -0.2

            helper = reconstruct_utils.ReconstructHelper(
                f,
                f,  # focal length
                cc1,
                cc2,  # image center
                r1,
                r2,  # radial distortion
                0,
                0,
            )  # tangential distortion

        scc = reconstruct.SingleCameraCalibration_from_basic_pmat(
            P, cam_id=cam_id, res=res,
        )
        sccs.append(scc)
        if 1:
            # XXX test
            K2, R2 = scc.get_KR()
            if 0:
                print("C", C)
                print("t", t)
                print("K", K)
                print("K2", K2)
                print("R", R)
                print("R2", R2)
                print("P", P)
                print("KR|t", numpy.dot(K, Rt))
                t2 = scc.get_t()
                print("t2", t2)
                Rt2 = numpy.concatenate((R2, t2), axis=1)
                print("KR2|t", numpy.dot(K2, Rt2))
                print()
            KR2 = numpy.dot(K2, R2)
            KR = numpy.dot(K, R)
            if not numpy.allclose(KR2, KR):
                if not numpy.allclose(KR2, -KR):
                    raise ValueError("expected KR2 and KR to be identical")
                else:
                    print("WARNING: weird sign error in calibration math FIXME!")
    recon = reconstruct.Reconstructor(sccs)

    full_info = {
        "reconstructor": recon,
        "center": common_point,  # where all the cameras are looking
        "camera_dist_from_center": r,
        "resolutions": resolutions,
    }
    if return_full_info:
        return full_info
    return recon


def generate_point_cloud(full_info, n_pts=200):
    recon = full_info["reconstructor"]
    std = full_info["camera_dist_from_center"] / 3.0
    mean = full_info["center"][:, np.newaxis]
    np.random.seed(3)
    X = np.random.normal(size=(3, n_pts)) * std + mean
    del n_pts  # meaning above is different from meaning below

    IdMat = []
    points = []

    for idx in range(X.shape[1]):
        n_pts = 0
        IdMat_row = []
        points_row = []
        for cam_id in recon.get_cam_ids():
            # get the distorted projection
            x2di = recon.find2d(cam_id, X[:, idx], distorted=True)

            found = True
            if not found:
                IdMat_row.append(0)
                points_row.extend([numpy.nan, numpy.nan, numpy.nan])
            else:
                n_pts += 1
                IdMat_row.append(1)
                points_row.extend([x2di[0], x2di[1], 1.0])

        IdMat.append(IdMat_row)
        points.append(points_row)
    IdMat = numpy.array(IdMat, dtype=numpy.uint8).T
    points = numpy.array(points, dtype=numpy.float32).T
    results = {
        "IdMat": IdMat,
        "points": points,
    }
    return results


def test(calib_dir=None, radial_distortion=True, square_pixels=True):
    """generate a fake calibration and save it.

    Arguments
    ---------
    calib_dir : string (optional)
      the directory name to save the resulting calibration data
    radial_distortion : boolean
      whether or not the calibration should have radial distortion
    square_pixels : boolen
      whether or not the pixels are square
    """
    full_info = generate_calibration(
        return_full_info=True, radial_distortion=radial_distortion
    )
    results = generate_point_cloud(full_info)
    Res = full_info["resolutions"]
    dirname = "test_cal_dir"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cam_ids = full_info["reconstructor"].get_cam_ids()

    Res = []
    for cam_id in cam_ids:
        imsize = full_info["reconstructor"].get_resolution(cam_id)
        Res.append(imsize)
    Res = numpy.array(Res)

    basename = "basename"
    if calib_dir is not None:
        save_cal_dir = save_calibration_directory(
            IdMat=results["IdMat"],
            points=results["points"],
            Res=Res,
            calib_dir=calib_dir,
            cam_ids=cam_ids,
            radial_distortion=radial_distortion,
            square_pixels=square_pixels,
            reconstructor=full_info["reconstructor"],
        )


if __name__ == "__main__":
    test(calib_dir="test_cal_dir", radial_distortion=True)
