import numpy

class EKF(object):
    """Extendend Kalman Filter

    This is only a "half" EKF - only the observation model is
    non-linear.  This non-linear function h() maps the a priori
    estimate of position into camera coordinates for N cameras.

    """
    def __init__(self,A=None,Q=None,initial_x=None,initial_P=None):
        self.A = A # process update model
        self.Q = Q # process covariance matrix
        # observation model and measurement covariance are generated with each step

        # These 2 attributes are the only state that changes during
        # filtering:
        self.xhat_k1 = initial_x # a posteriori state estimate from step (k-1)
        self.P_k1 = initial_P    # a posteriori error estimate from step (k-1)

        self.ss = self.A.shape[0] # ndim in state space
        self.AT = self.A.T

        if len(initial_x)!=self.ss:
            raise ValueError( 'initial_x must be a vector with ss components' )

    def step1__calculate_a_priori(self,isinitial=False):
        dot = numpy.dot # shorthand
        ############################################
        #          update state-space

        # compute a priori estimate of statespace
        if not isinitial:
            xhatminus = dot(self.A,self.xhat_k1)
            # compute a priori estimate of errors
            Pminus = dot(dot(self.A,self.P_k1),self.AT)+self.Q
        else:
            xhatminus = self.xhat_k1
            Pminus = self.P_k1

        return xhatminus, Pminus

    def step2__calculate_a_posteriori(self,
                                      xhatminus, Pminus,
                                      C,R,missing_data=False):

        ss = len(xhatminus) # state space size

        if not missing_data:

            CT = C.T

            dot= numpy.dot
            inv = numpy.linalg.inv

            ############################################
            #          incorporate observation

            # calculate a posteriori state estimate

            # calculate Kalman gain
            Knumerator = dot(Pminus,CT)
            Kdenominator = dot(dot(C,Pminus),CT)+R
            K = dot(Knumerator,inv(Kdenominator)) # Kalman gain

            residuals = y-hx
            xhat = xhatminus+dot(K, residuals)
            one_minus_KC = numpy.eye(ss)-dot(K,C)

            # compute a posteriori estimate of errors
            P = dot(one_minus_KC,Pminus)
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

    def step(self):
        xhatminus, Pminus = self.step1__calculate_a_priori(isinitial=False)
        pmats_and_points_cov = []
        return self.step2__calculate_a_posteriori(xhatminus, Pminus, pmats_and_points_cov)
