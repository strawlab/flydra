from __future__ import print_function
import numpy
import scipy  # use numpy if scipy unavailable
import scipy.linalg  # use numpy if scipy unavailable

## Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:

##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.

##     * Redistributions in binary form must reproduce the above
##       copyright notice, this list of conditions and the following
##       disclaimer in the documentation and/or other materials provided
##       with the distribution.

##     * Neither the name of the Andrew D. Straw nor the names of its
##       contributors may be used to endorse or promote products derived
##       from this software without specific prior written permission.

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm
    
This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        assert test_err.ndim == 1
        assert test_err.shape[0] == len(test_points)
        also_idxs = test_idxs[
            test_err < t
        ]  # select indices of rows with accepted points
        alsoinliers = data[also_idxs]
        if debug:
            # print 'test_err.min()',test_err.min()
            # print 'test_err.max()',test_err.max()
            # print 'numpy.mean(test_err)',numpy.mean(test_err)
            # print 'iteration %d:len(alsoinliers) = %d'%(
            #    iterations,len(alsoinliers))
            print(
                "  %  20.3f         % 6d     % 6d"
                % (numpy.mean(test_err), iterations, len(alsoinliers))
            )
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = numpy.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {"inliers": best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange(n_data)
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2
