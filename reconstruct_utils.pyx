import numarray as nx

cdef class ReconstructHelper:
    cdef float fc1, fc2, cc1, cc2
    cdef float k1, k2, p1, p2

    def __init__(self, fc1, fc2, cc1, cc2, k1, k2, p1, p2 ):
        self.fc1 = fc1
        self.fc2 = fc2
        self.cc1 = cc1
        self.cc2 = cc2
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def get_K(self):
        K = nx.array((( self.fc1, 0, self.cc1),
                      ( 0, self.fc2, self.cc2),
                      ( 0, 0, 1)))
        return K

    def get_nlparams(self):
        return (self.k1, self.k2, self.p1, self.p2)

    def undistort(self, float x_kk, float y_kk):
        """undistort 2D coordinate pair

        Iteratively performs an undistortion using camera intrinsic
        parameters.

        Implementation translated from CalTechCal.

        See also the OpenCV reference manual, which has the equation
        used.
        """
        
        cdef float xl, yl

        cdef float xd, yd, x, y
        cdef float r_2, k_radial, delta_x, delta_y
        cdef int i

        # undoradial.m
        
        xd = ( x_kk - self.cc1 ) / self.fc1
        yd = ( y_kk - self.cc2 ) / self.fc2

        # comp_distortion_oulu.m
        
        # initial guess
        x = xd 
        y = yd

        for i from 0<=i<20:
            r_2 = x*x + y*y
            k_radial = 1.0 + (self.k1) * r_2 + (self.k2) * r_2*r_2
            delta_x = 2.0 * (self.p1)*x*y + (self.p2)*(r_2 + 2.0*x*x)
            delta_y = (self.p1) * (r_2 + 2.0*y*y)+2.0*(self.p2)*x*y
            x = (xd-delta_x)/k_radial
            y = (yd-delta_y)/k_radial

        # undoradial.m
        
        xl = (self.fc1)*x + (self.cc1)
        yl = (self.fc2)*y + (self.cc2)
        return (xl, yl)

    def distort(self, float xl, float yl):
        """distort 2D coordinate pair"""
        
        cdef float x, y, r_2, term1, xd, yd
        
        x = ( xl - self.cc1 ) / self.fc1
        y = ( yl - self.cc2 ) / self.fc2
        
        r_2 = x*x + y*y
        term1 = self.k1*r_2 + self.k2*r_2**2
        xd = x + x*term1 + (2*self.p1*x*y + self.p2*(r_2+2*x**2))

        # XXX OpenCV manual may be wrong -- double check this eqn
        # (esp. first self.p2 term):
        yd = y + y*term1 + (2*self.p2*x*y + self.p2*(r_2+2*y**2))

        xd = (self.fc1)*xd + (self.cc1)
        yd = (self.fc2)*yd + (self.cc2)
        
        return (xd, yd)
        
