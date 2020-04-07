from __future__ import absolute_import
import math
import numpy as nx
from pylab import linspace
from . import PQmath
from numpy.random import normal

if __name__ == "__main__":
    if 0:
        P = nx.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2.1, 3.01, 4, 5, 6]])
        P.transpose()
    else:
        t = linspace(0, 10 * math.pi, 400)
        P = nx.sin(t)[:, nx.newaxis]
        P = P + normal(0, 0.1, P.shape)

    Pstar = PQmath.smooth_position(P, 0.01, 0.5, 1e-9, 1e12)

    from pylab import *

    plot(t, P, "b-")
    plot(t, Pstar, "r-")
    show()
