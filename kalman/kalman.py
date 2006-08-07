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

        self.n_skipped = 0

        if len(initial_x)!=self.ss:
            raise ValueError( 'initial_x must be a vector with ss components' )
        
    def step(self,y=None):
        xhatminus, Pminus = self.step1__calculate_a_priori()
        return self.step2__calculate_a_posteri(xhatminus, Pminus)
    
    def step1__calculate_a_priori(self):
        dot = numpy.dot # shorthand
        ############################################
        #          update state-space
        
        # compute a priori estimate of statespace
        xhatminus = dot(self.A,self.xhat_k1)

        # compute a priori estimate of errors
        Pminus = dot(dot(self.A,self.P_k1),self.AT)+self.Q

        return xhatminus, Pminus

    def step2__calculate_a_posteri(self,xhatminus,Pminus,y=None):
        """
        y represents the observation for this time-step
        """
        dot = numpy.dot # shorthand
        inv = numpy.linalg.inv
        
        ############################################
        #          incorporate observation

        if y is None:
            self.n_skipped += 1
        else:
            self.n_skipped = 0

        # With each skipped data point, measurement uncertainty doubles,
        # which means variance goes up 4x.

        factor = 2.0**self.n_skipped**2.0
        this_R = factor*self.R

        # calculate Kalman gain
        Knumerator = dot(Pminus,self.CT)
        Kdenominator = dot(dot(self.C,Pminus),self.CT)+this_R
        K = dot(Knumerator,inv(Kdenominator))

        # calculate a posteri state estimate
        if y is not None:
            print 'KALMAN y',y
            residuals = y-dot(self.C,xhatminus) # error/innovation
            print 'KALMAN residuals',residuals
            xhat = xhatminus+dot(K, residuals)
            print 'KALMAN xhat',xhat
        else:
            xhat = xhatminus
            
        one_minus_KC = numpy.eye(self.ss)-dot(K,self.C)
        
        # compute a posteri estimate of errors
        P = dot(one_minus_KC,Pminus)

        # this step (k) becomes next step's prior (k-1)
        self.xhat_k1 = xhat
        self.P_k1 = P

        return xhat, P
