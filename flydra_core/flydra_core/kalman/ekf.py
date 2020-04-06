import numpy
import numpy as np


class EKF(object):
    """Extendend Kalman Filter

    This is only a "half" EKF - only the observation model is
    non-linear.  This non-linear function h() maps the a priori
    estimate of position into camera coordinates for N cameras.

    """

    def __init__(self, initial_x=None, initial_P=None):
        # These 2 attributes are the only state that changes during
        # filtering:
        self.xhat_k1 = initial_x  # a posteriori state estimate from step (k-1)
        self.P_k1 = initial_P  # a posteriori error estimate from step (k-1)

        self.ss = len(self.xhat_k1)

    def step1__calculate_a_priori(self, A, Q, isinitial=False):
        dot = numpy.dot  # shorthand
        ############################################
        #          update state-space

        # compute a priori estimate of statespace
        if not isinitial:
            AT = A.T
            xhatminus = dot(A, self.xhat_k1)
            # compute a priori estimate of errors
            Pminus = dot(dot(A, self.P_k1), AT) + Q
        else:
            xhatminus = self.xhat_k1
            Pminus = self.P_k1

        return xhatminus, Pminus

    def step2__calculate_a_posteriori(
        self, xhatminus, Pminus, y=None, hx=None, C=None, R=None, missing_data=False
    ):

        ss = len(xhatminus)  # state space size
        assert Pminus.shape == (ss, ss)

        if not missing_data:

            CT = C.T

            y = np.asanyarray(y)
            hx = np.asanyarray(hx)

            os = len(y)
            assert y.ndim == 1, "observations must be vector"
            assert hx.ndim == 1, "expected values must be vector"
            assert CT.shape == (ss, os)
            assert len(hx) == os
            assert R.shape == (os, os)

            dot = numpy.dot
            inv = numpy.linalg.inv

            ############################################
            #          incorporate observation

            # calculate a posteriori state estimate

            # calculate Kalman gain
            Knumerator = dot(Pminus, CT)
            Kdenominator = dot(dot(C, Pminus), CT) + R
            K = dot(Knumerator, inv(Kdenominator))  # Kalman gain

            residuals = y - hx
            xhat = xhatminus + dot(K, residuals)
            one_minus_KC = numpy.eye(ss) - dot(K, C)

            # compute a posteriori estimate of errors
            P = dot(one_minus_KC, Pminus)
        else:
            # no observation
            xhat = xhatminus
            P = Pminus

        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P
        ##         if 1:
        ##             print 'xhat'
        ##             print xhat
        ##             print
        return xhat, P

    def step(self, A, Q):
        xhatminus, Pminus = self.step1__calculate_a_priori(A, Q, isinitial=False)
        return self.step2__calculate_a_posteriori(xhatminus, Pminus, missing_data=True)
