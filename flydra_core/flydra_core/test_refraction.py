import flydra_core._refraction as _refraction
import flydra_core._Roots3And4 as _Roots3And4

import numpy as np
import scipy.optimize
import math


def test_refraction():
    n1 = 1.0
    n2 = 2.0

    v1 = 1.0 / n1
    v2 = 1.0 / n2

    height = 1.0
    r = 1.2345

    # Test case where depth is zero: r == r0
    depth = 0.0

    eps = 1e-7

    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth, eps)
    assert abs(r - r0) < 1e-6

    # Test case with real depth
    depth = 3.0
    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth, eps)

    # test correctness with Snell's Law
    theta1 = np.arctan(r0 / height)
    theta2 = np.arctan((r - r0) / depth)

    ratio1 = v1 / v2
    ratio2 = np.sin(theta1) / np.sin(theta2)

    assert abs(ratio1 - ratio2) < 1e-6

    # test case that was actually failing in real-world use
    n1 = 1.0003
    n2 = 1.333
    height = 0.40279482775695
    r = 0.793324332905365
    depth = 1.7437460383163346
    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth, eps)


def find_fastest_path_fermat_numeric(n1, n2, z1, h, z2):
    def duration(h1):
        h1 = h1[0]
        # Fermat's principle of least time
        h2 = h - h1
        return n1 * math.sqrt(h1 * h1 + z1 * z1) + n2 * math.sqrt(z2 * z2 + h2 * h2)

    initial_r0 = h
    (final_r0,) = scipy.optimize.fmin(duration, [initial_r0], disp=0)
    return final_r0


def test_numeric_vs_algebraic():
    n1 = 1.0003
    n2 = 1.333
    height = 0.303
    r = 0.153
    depth = 0.010
    r0 = _refraction.find_fastest_path_fermat(n1, n2, height, r, depth, eps=1e-7)
    r0_numeric = find_fastest_path_fermat_numeric(n1, n2, height, r, depth)
    assert abs(r0 - r0_numeric) < 1e-5


def test_roots():
    poly = [
        -0.77628891,
        0.23754440646,
        -0.181206489286,
        0.0499191270735,
        -0.00381881322112,
    ]
    actual = _Roots3And4.py_real_nonnegative_root_less_than(
        poly[0], poly[1], poly[2], poly[3], poly[4], 0.153, 1e-7
    )
    expected_list = np.roots(poly)
    found = 0
    for maybe_expected in expected_list:
        if abs(actual - maybe_expected) < 1e-7:
            found += 1
    assert found == 1


if __name__ == "__main__":
    test_refraction()
    test_numeric_vs_algebraic()
    test_roots()
