from __future__ import division
import math
import numpy as np

def calc_online(x):
    cumsum = 0
    cumsumsq = 0
    N = len(x)
    for i in range(N):
        cumsum += x[i]
        cumsumsq += x[i]*x[i]

    mean = cumsum/N
    std = math.sqrt(cumsumsq/N - (mean)**2)
    return mean,std

# compare against the slow method
def calc_classic(x):
    cumstd = 0
    N = len(x)
    mean = np.mean(x)
    for i in range(N):
        cumstd += (x[i]-mean)**2
    std = math.sqrt( cumstd/(N) )
    return mean,std

def test_std():
    # Test the fast method of generating a standard deviation
    # compatible with online computation.

    x=[1,2,3,4,5,6,7,1.2,3.4]
    m1, s1 = calc_online(x)
    m2, s2 = calc_classic(x)
    assert np.allclose(m1,m2)
    assert np.allclose(s1,s2)
