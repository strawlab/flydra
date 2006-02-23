import time

flavor = 'numarray'
for flavor in ['numpy','numarray','Numeric']:
    if flavor=='numpy':
        import numpy as nx
        svd = nx.linalg.svd
    elif flavor=='numarray':
        import numarray as nx
        import numarray.linear_algebra
        svd = numarray.linear_algebra.singular_value_decomposition
    elif flavor=='Numeric':
        import Numeric
        import LinearAlgebra
        svd = LinearAlgebra.singular_value_decomposition

    A=nx.array([[ -8.66169699e+00,   1.14101524e+00,   5.86120905e+00,
                  4.33167687e+03],
                [  5.29672728e-01,   1.03148534e+01,  -1.22279830e+00,
                   -2.12074919e+03],
                [  1.02362368e+01,  -1.63800749e+00,   1.13378266e+00,
                   -5.01902657e+03],
                [ -1.49785754e+00,  -9.76800223e+00,  -2.89216682e+00,
                  2.50619381e+03],
                [  1.57361401e+00,   1.01013828e+01,   1.63416950e-01,
                   -2.60341213e+03],
                [  6.50981736e+00,  -8.82618159e-01,  -7.84807894e+00,
                   -3.27889477e+03],
                [ -5.72762757e+00,  -2.03115295e-01,   8.20604874e+00,
                  3.04527951e+03],
                [ -2.08506003e-01,   1.00567483e+01,   1.73205928e-01,
                  -1.68579242e+03],
                [ -7.54548555e+00,   4.07042510e-01,  -7.93926917e+00,
                  3.81059741e+03],
                [  5.96820986e-01,   1.08763697e+01,  -8.42269506e-02,
                   -2.24941005e+03]])

    n=10000
    t1=time.time()
    for i in range(n):
        svd(A)
    t2=time.time()
    dur = t2-t1
    print flavor,n/dur,'svd/sec'

