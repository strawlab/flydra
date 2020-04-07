from __future__ import division
from __future__ import print_function
import math

import numpy
import scipy.special as special

from numpy import sum, cos, sin, sqrt


def A1(kappa):
    # XXX R has these exponentially scaled, but this seems to work
    result = special.i1(kappa) / special.i0(kappa)
    return result


def A1inv(R):
    if 0 <= R < 0.53:
        return 2 * R + R ** 3 + (5 * R ** 5) / 6
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        return 1 / (R ** 3 - 4 * R ** 2 + 3 * R)


def mle_vonmises(theta):
    results = {}
    n = len(theta)
    C = sum(cos(theta))
    S = sum(sin(theta))
    R = sqrt(C ** 2 + S ** 2)
    mean_direction = math.atan2(S, C)
    results["mu"] = mean_direction
    mean_R = R / n
    kappa = A1inv(mean_R)
    results["kappa"] = kappa
    if 0:
        z = F(theta - mu_hat)
        z.sort()

        z_bar = sum(z) / n
        if 0:
            tmp = 0
            for i in range(n):
                tmp += (z[i] - 2 * i / (2 * n)) ** 2
            U2 = tmp - n * (z_bar - 0.5) ** 2 + 1 / (12 * n)
        else:
            U2 = (
                sum((z - 2 * numpy.arange(n) / (2 * n)) ** 2)
                - n * (z_bar - 0.5) ** 2
                + 1 / (12 * n)
            )
        results["U2"] = U2
    return results


def lm_circular_cl(y, x, init, verbose=False, tol=1e-10):
    """circular-linear regression

    y in radians
    x is linear
    """
    y = numpy.mod(y, 2 * numpy.pi)
    betaPrev = init
    n = len(y)
    S = numpy.sum(numpy.sin(y - 2 * numpy.arctan(x * betaPrev))) / n
    C = numpy.sum(numpy.cos(y - 2 * numpy.arctan(x * betaPrev))) / n
    R = numpy.sqrt(S ** 2 + C ** 2)
    mu = numpy.arctan2(S, C)
    k = A1inv(R)
    diff = tol + 1
    iter = 0

    while diff > tol:
        iter += 1
        u = k * numpy.sin(y - mu - 2 * numpy.arctan(x * betaPrev))
        A = k * A1(k) * numpy.eye(n)
        g_p = 2 / (1 + betaPrev * x) ** 2 * numpy.eye(n)
        D = g_p * x
        raise NotImplementedError()


def test():
    print("A1inv(5.0)", A1inv(5.0))
    print("A1inv(0.0)", A1inv(0.0))
    print("A1inv(3421324)", repr(A1inv(3421324)))
    print("A1inv(-0.5)", repr(A1inv(-0.5)))
    print("A1inv(0.5)", repr(A1inv(0.5)))
    print("A1inv(0.7)", repr(A1inv(0.7)))
    # print 'A1inv(numpy.inf)',A1inv(numpy.inf)


def raw_data_plot(ax, theta, r=1.0, **kwargs):
    # plot ring
    ct = list(numpy.arange(0, 2 * numpy.pi, 0.01))
    ct = ct + [ct[0]]  # wrap
    cr = 0.9
    cx = cr * numpy.cos(ct)
    cy = cr * numpy.sin(ct)
    ax.plot(cx, cy, "k-")

    x = r * numpy.cos(theta)
    y = r * numpy.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    test()
