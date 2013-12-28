import _refraction
import numpy as np

def test_refraction():
    n1 = 1.0
    n2 = 2.0

    v1 = 1.0/n1
    v2 = 1.0/n2

    height = 1.0
    r = 1.2345

    # Test case where depth is zero: r == r0
    depth = 0.0

    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth)
    assert abs(r-r0) < 1e-6

    # Test case with real depth
    depth = 3.0
    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth)

    # test correctness with Snell's Law
    theta1 = np.arctan( r0 / height )
    theta2 = np.arctan( (r-r0) / depth )

    ratio1 = v1/v2
    ratio2 = np.sin(theta1) / np.sin(theta2)

    assert abs(ratio1 - ratio2) < 1e-6
