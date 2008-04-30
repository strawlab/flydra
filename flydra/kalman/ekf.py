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
        self.xhat_k1 = initial_x # a posteri state estimate from step (k-1)
        self.P_k1 = initial_P    # a posteri error estimate from step (k-1)

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

    def step2__calculate_a_posteri(self,
                                   xhatminus, Pminus,
                                   pmats_and_points_cov):

        ss = len(xhatminus) # state space size

        N = len(pmats_and_points_cov) # number of observations
        if N > 0:
            missing_data = False
        else:
            missing_data = True

        if not missing_data:
            # Create 2N vector of N observations.
            y = numpy.empty((2*N,), dtype=numpy.float64)

            # Create 2N x 4 observation model matrix (jacobian of h() at xhatminus).
            C = numpy.zeros((2*N,ss), dtype=numpy.float64)

            # Create 2N vector h(xhatminus) where xhatminus is the a
            # priori estimate and h() is the observation model.
            hx = numpy.empty((2*N,), dtype=numpy.float64)

            # Create 2N x 2N observation covariance matrix
            R = numpy.zeros((2*N,2*N), dtype=numpy.float64)

            # evaluate jacobian for each participating camera
            for i,(pmat,nonlin_model,xy2d_obs,cov) in enumerate(pmats_and_points_cov):

                # fill prediction vector [ h(xhatminus) ]
                hx_i = nonlin_model(xhatminus)
                hx[2*i:2*i+2] = hx_i

                # fill observation  vector
                y[2*i:2*i+2] = xy2d_obs

                # fill observation model
                C_i = nonlin_model.evaluate_jacobian_at(xhatminus)
                C[2*i:2*i+2,:3] = C_i

                # fill observation covariance
                R[2*i:2*i+2,2*i:2*i+2]=cov

    ##         if 1:
    ##             print 'C'
    ##             print C
    ##             print 'y'
    ##             print y
    ##             print 'hx'
    ##             print hx
    ##             print 'R'
    ##             print R

            CT = C.T

            dot= numpy.dot
            inv = numpy.linalg.inv

            ############################################
            #          incorporate observation

            # calculate a posteri state estimate

            # calculate Kalman gain
            Knumerator = dot(Pminus,CT)
            Kdenominator = dot(dot(C,Pminus),CT)+R
            K = dot(Knumerator,inv(Kdenominator)) # Kalman gain

            residuals = y-hx
            xhat = xhatminus+dot(K, residuals)
            one_minus_KC = numpy.eye(ss)-dot(K,C)

            # compute a posteri estimate of errors
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
        return self.step2__calculate_a_posteri(xhatminus, Pminus, pmats_and_points_cov)
