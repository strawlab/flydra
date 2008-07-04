import pkg_resources
import unittest

import numpy as np
from PQmath import ObjectiveFunctionQuats, QuatSeq, orientation_to_quat, quat_to_orient, CachingObjectiveFunctionQuats
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

        self.delta_t = 1.0/fps
        self.beta = 1.0
        self.gamma = 0.0
        self.lambda2 = 1e-11
        self.percent_error_eps_quats = 9
        self.epsilon2 = 0
        self.max_iter2 = 2000

    def test_Qsmooth_slow(self):
        Qsmooth = self._tst_of(ObjectiveFunctionQuats)

    def test_Qsmooth_caching(self):
        Qsmooth = self._tst_of(CachingObjectiveFunctionQuats)

    def test_Qsmooth_both(self):
        Qsmooth_slow = self._tst_of(ObjectiveFunctionQuats)
        Qsmooth_cache = self._tst_of(CachingObjectiveFunctionQuats)
        print 'Qsmooth_slow'
        print Qsmooth_slow
        print 'Qsmooth_cache'
        print Qsmooth_cache
        for qs, qc in zip(Qsmooth_slow,Qsmooth_cache):
            print qs,qc
            try:
                assert qs==qc
            except:
                print 'repr(qs.w - qc.w)',repr(qs.w - qc.w)
                print 'repr(qs.x - qc.x)',repr(qs.x - qc.x)
                print 'repr(qs.y - qc.y)',repr(qs.y - qc.y)
                print 'repr(qs.z - qc.z)',repr(qs.z - qc.z)
                raise

    def _tst_of(self,of):
        of = of(self.Q, self.delta_t, self.beta, self.gamma)
        #no_distance_penalty_idxs=slerped_q_idxs)

        #lambda2 = 2e-9
        #lambda2 = 1e-9
        #lambda2 = 1e-11
        Q_k = self.Q[:] # make copy
        last_err = None
        count = 0
        while count<self.max_iter2:
            count += 1
            start = time.time()
            of.set_cache_qs(Q_k) # set the cache (no-op on non-caching version)
            del_G = of.get_del_G(Q_k)
            #ADS print 'G'
            D = of._getDistance(Q_k)
            #ADS print 'D'
            E = of._getEnergy(Q_k)
            #ADS print 'E'
            R = of._getRoll(Q_k)
            #ADS print '  G = %s + %s*%s + %s*%s'%(str(D),str(self.beta),str(E),str(self.gamma),str(R))
            stop = time.time()
            err = np.sqrt(np.sum(np.array(abs(del_G))**2))
            if err < self.epsilon2:
                #ADS print 'reached epsilon2'
                break
            elif last_err is not None:
                pct_err = (last_err-err)/last_err*100.0
                #ADS print 'Q elapsed: % 6.2f secs,'%(stop-start,),
                #ADS print 'current gradient:',err,
                #ADS print '   (%4.2f%%)'%(pct_err,)

                if err > last_err:
                    #ADS print 'ERROR: error is increasing, aborting'
                    break
                if pct_err < self.percent_error_eps_quats:
                    #ADS print 'reached percent_error_eps_quats'
                    break
            else:
                #ADS print 'Q elapsed: % 6.2f secs,'%(stop-start,),
                #ADS print 'current gradient:',err
                pass
            last_err = err
            Q_k = Q_k*(del_G*-self.lambda2).exp()
        if count>=self.max_iter2:
            #ADS print 'reached max_iter2'
            pass
        Qsmooth = Q_k

        direction_vec_smooth = quat_to_orient(Qsmooth)
        return Qsmooth

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
