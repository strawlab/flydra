import pkg_resources
import unittest

import numpy as np
from PQmath import ObjectiveFunctionQuats, QuatSeq, orientation_to_quat, quat_to_orient, CachingObjectiveFunctionQuats, QuatSmoother
import cgkit.cgtypes
import time

class TestPQmath(unittest.TestCase):
    def setUp(self):
        D2R = np.pi/180.0

        fps = 200.0
        t = np.arange(0,0.1*fps)/fps

        yaw_angle = np.sin( (2*np.pi*t) / 4 )
        pitch_angle =  45*D2R*np.ones_like(yaw_angle)

        z_pitch = np.sin(pitch_angle)
        r_pitch = np.cos(pitch_angle)
        direction_vec = np.array( [ r_pitch*np.cos( yaw_angle ),
                                    r_pitch*np.sin( yaw_angle ),
                                    z_pitch ] ).T

        if 1:
            noise_mag = 0.3
            unscaled_noise = np.random.randn( len(t),3 )
            direction_vec += noise_mag*unscaled_noise

        # (re-)normalize

        r=np.sqrt( np.sum( direction_vec**2, axis=1) )
        direction_vec = direction_vec/ r[:,np.newaxis]
        self.Q = QuatSeq([ orientation_to_quat(U) for U in direction_vec ])

        self.fps = fps

    def test_Qsmooth_slow(self):
        Qsmooth = QuatSmoother(frames_per_second=self.fps).smooth_quats(self.Q,objective_func_name='ObjectiveFunctionQuats')

    def test_Qsmooth_caching(self):
        Qsmooth = QuatSmoother(frames_per_second=self.fps).smooth_quats(self.Q,objective_func_name='CachingObjectiveFunctionQuats')

    def test_Qsmooth_both(self):
        Qsmooth_slow = QuatSmoother(frames_per_second=self.fps).smooth_quats(self.Q,objective_func_name='ObjectiveFunctionQuats')
        Qsmooth_cache = QuatSmoother(frames_per_second=self.fps).smooth_quats(self.Q,objective_func_name='CachingObjectiveFunctionQuats')
        for qs, qc in zip(Qsmooth_slow,Qsmooth_cache):
            assert qs==qc

if 0:
    import pylab

    ax=pylab.subplot(3,1,1)
    line,=ax.plot(t,direction_vec[:,0],'.')
    line,=ax.plot(t,direction_vec_smooth[:,0],'-')

    ax=pylab.subplot(3,1,2,sharex=ax)
    line,=ax.plot(t,direction_vec[:,1],'.')
    line,=ax.plot(t,direction_vec_smooth[:,1],'-')

    ax=pylab.subplot(3,1,3,sharex=ax)
    line,=ax.plot(t,direction_vec[:,2],'.')
    line,=ax.plot(t,direction_vec_smooth[:,2],'-')

    pylab.show()

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestPQmath),
                           ]
                          )
    return ts

if __name__=='__main__':
    if 1:
        suite = get_test_suite()
        suite.debug()
    else:
        unittest.main()
