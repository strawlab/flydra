from __future__ import division
import math

# test the fast method of generating a standard deviation
x=[1,2,3,4,5,6,7,1.2,3.4]

cumsum = 0
cumsumsq = 0
N = len(x)
for i in range(N):
    cumsum += x[i]
    cumsumsq += x[i]*x[i]

mean = cumsum/N
std = math.sqrt(cumsumsq/N - (mean)**2)

print 'mean, std',mean, std

# compare against the slow method

cumstd = 0
for i in range(N):
    cumstd += (x[i]-mean)**2
std = math.sqrt( cumstd/(N) )

print 'mean, std',mean, std
