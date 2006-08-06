import numpy

class KalmanFilter:
    def __init__(self,A,C,Q,R,initial_x,initial_P):
        self.A = A # process update model
        self.C = C # observation model
        self.Q = Q # process covariance matrix
        self.R = R # measurement covariance matrix
        self.xhat_k1 = initial_x # a posteri state estimate from step (k-1)
        self.P_k1 = initial_P    # a posteri error estimate from step (k-1)

        self.ss = self.A.shape[0] # ndim in state space
        self.os = self.C.shape[0] # ndim in observation space
        self.AT = self.A.T
        self.CT = self.C.T
        
    def step(self,y=None,return_error_estimate=False):
        """perform a single time-step in a Kalman filter process

        y represents the observation for this time-step
        """
        dot = numpy.dot # shorthand
        inv = numpy.linalg.inv
        
        ############################################
        #          update state-space
        
        # compute a priori estimate of statespace
        xhatminus = dot(self.A,self.xhat_k1)

        # compute a priori estimate of errors
        Pminus = dot(dot(self.A,self.P_k1),self.AT)+self.Q

        ############################################
        #          incorporate observation

        # calculate Kalman gain
        Knumerator = dot(Pminus,self.CT)
        Kdenominator = dot(dot(self.C,Pminus),self.CT)+self.R
        # XXX need to implement direct method
        K = dot(Knumerator,inv(Kdenominator))

        # calculate a posteri state estimate
        if y is not None:
            residuals = y-dot(self.C,xhatminus) # error/innovation
            xhat = xhatminus+dot(K, residuals)
        else:
            xhat = xhatminus
            
        one_minus_KC = numpy.eye(self.ss)-dot(K,self.C)
        
        # compute a posteri estimate of errors
        P = dot(one_minus_KC,Pminus)

        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P

        if return_error_estimate:
            return xhat, P
        else:
            return xhat
