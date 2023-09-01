from __future__ import print_function
import numpy as np
import adskalman.adskalman as adskalman


def running_average(src, min_dist):
    if len(src) < min_dist:
        raise ValueError("the input sequence is too short")
    min_dist = 5
    accum = []
    for i in range(min_dist):
        accum.append(src[i : (len(src) - min_dist + i + 1)])
    accum = np.array(accum)
    near = np.mean(accum, axis=0)
    return near


def test_running_average():
    a = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5]
    b = [0, 1, 2, 3, 4, 5]
    bex = running_average(a, 5)
    assert np.allclose(bex, b)

    a = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.bool_)
    b = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bex = running_average(a, 5)
    assert np.allclose(bex, b)

    try:
        running_average([1, 2], 5)
    except ValueError:
        pass
    else:
        raise ValueError("failed to abort on short input")


def ori_smooth(directions, frames_per_second=None, return_missing=False):
    """smooth orientations using an RTS smoother

    This treats orientations as XYZ positions
    """
    dt = 1.0 / frames_per_second

    A = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    Q = 1.0 * np.eye(6)

    C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    R = 30.0 * np.eye(3)

    init_x = np.hstack((directions[0, :], np.zeros((3,))))
    assert not np.any(np.isnan(init_x)), "cannot start with missing orientation"
    init_V = 2 * Q
    y = directions
    dirsmooth, V = adskalman.kalman_smoother(y, A, C, Q, R, init_x, init_V)
    dirsmooth = dirsmooth[:, :3]  # take only position information
    dirsmooth_missing = np.array(dirsmooth, copy=True)

    # remove results too distant from observations
    avlen = 5
    if len(y) >= avlen:
        good = ~np.isnan(y[:, 0])
        near_good = running_average(good, avlen)
        pad = avlen // 2
        near_good = np.hstack((np.zeros((pad,)), near_good, np.zeros((pad,))))

        good_cond = near_good > 0.2
        bad_cond = ~good_cond

        if bad_cond.shape != dirsmooth[:, 0].shape:
            print("xxxx")
            print(bad_cond.shape)
            print(dirsmooth[:, 0].shape)
            print(directions.shape)
        dirsmooth[bad_cond, :] = np.nan

    # normalize lengths to unit vectors
    np.sum(dirsmooth ** 2, axis=1)
    dirsmooth = (dirsmooth.T / np.sqrt(np.sum(dirsmooth ** 2, axis=1))).T

    if return_missing:
        return dirsmooth, dirsmooth_missing
    else:
        return dirsmooth


def test_ori_smooth():
    directions = np.zeros([300, 3])
    directions[:, 0] = 1.0
    smoothed = ori_smooth(directions, frames_per_second=1.0)
