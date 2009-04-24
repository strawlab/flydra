import numpy as np
import sympy

class SymbolicStateVector:
    """Give names to the components of the state vector"""
    def __init__(self,x):
        # position state
        self.px  = x[0]
        self.py  = x[1]
        self.pz  = x[2]
        self.pvx = x[3]
        self.pvy = x[4]
        self.pvz = x[5]

        # orientation state
        self.x1 = x[6]
        self.x2 = x[7]
        self.x3 = x[8]
        self.x4 = x[9]
        self.x5 = x[10]
        self.x6 = x[11]
        self.x7 = x[12]

class TargetModel(object):
    ## def f(self,x,t,new_x=None):
    ##     """process update model

    ##     x_{t+1} = f(x_t)

    ##     """
    def get_process_model_ODEs(self,x):
        """
        x should be a sympy.DeferredVector instance

        This formulation partly from Marins, Yun, Bachmann, McGhee, and
        Zyda (2001). An Extended Kalman Filter for Quaternion-Based
        Orientation Estimation Using MARG Sensors. Proceedings of the
        2001 IEEE/RSJ International Conference on Intelligent Robots and
        Systems.

        """
        assert isinstance(x,SymbolicStateVector)

        # ODEs for position state
        fx = x.pvx
        fy = x.pvy
        fz = x.pvz
        fvx = 1
        fvy = 1
        fvz = 1

        # angular rate decay
        if 0:
            tau_rx = Symbol('tau_rx')
            tau_ry = Symbol('tau_ry')
            tau_rz = Symbol('tau_rz')
        else:
            tau_rx = 0.1
            tau_ry = 0.1
            tau_rz = 0.1

        # ODEs for orientation state update
        # (from Marins et al. eqns 9-15 )
        f1 = -1/tau_rx*x.x1
        f2 = -1/tau_ry*x.x2
        f3 = -1/tau_rz*x.x3
        scale = 2*sympy.sqrt(x.x4**2 + x.x5**2 + x.x6**2 + x.x7**2)
        f4 = 1/scale * ( x.x3*x.x5 - x.x2*x.x6 + x.x1*x.x7 )
        f5 = 1/scale * (-x.x3*x.x4 + x.x1*x.x6 + x.x2*x.x7 )
        f6 = 1/scale * ( x.x2*x.x4 - x.x1*x.x5 + x.x3*x.x7 )
        f7 = 1/scale * (-x.x1*x.x4 - x.x2*x.x5 + x.x3*x.x6 )

        # form into column vector defining ODEs
        xdot = sympy.Matrix((fx,fy,fz,fvx,fvy,fvz,
                             f1,f2,f3,f4,f5,f6,f7)).T
        return xdot

    def get_process_model_ODEs_linearized(self,x):
        assert isinstance(x,SymbolicStateVector)
        xdot = self.get_process_model_ODEs(x)
        xdot_linearized = xdot.jacobian((x.px,x.py,x.pz,x.pvx,x.pvy,x.pvz,
                                         x.x1,x.x2,x.x3,x.x4,x.x5,x.x6,x.x7))
        return xdot_linearized
