from __future__ import division
import numpy
import scipy.linalg
import flydra.reconstruct as reconstruct
import cgtypes # cgkit 1.x

def generate_calibration(n_cameras = 1):
    pi = numpy.pi

    sccs = []

    # 1. extrinsic parameters:
    if 1:
        # method 1:
        #  arrange cameras in circle around common point
        common_point = numpy.array((0,0,0),dtype=numpy.float64)
        r = 10.0

        theta = numpy.linspace(0,2*pi,n_cameras,endpoint=False)
        x = numpy.cos(theta)
        y = numpy.sin(theta)
        z = numpy.zeros( y.shape )

        cc = numpy.c_[x,y,z]
        #cam_up = numpy.array((0,0,1))

        #cam_ups = numpy.resize(cam_up,cc.shape)
        #cam_forwads = -cc
        cam_centers = r*cc + common_point

        # Convert up/forward into rotation matrix.
        if 1:
            Rs = []
            for i,th in enumerate(theta):
                pos = cam_centers[i]
                target = common_point
                up = (0,0,1)
                print 'pos',pos
                print 'target',target
                print 'up',up
                R = cgtypes.mat4().lookAt( pos,
                                           target,
                                           up )
                #print 'R4',R
                R = R.getMat3()
                #print 'R3',R
                R = numpy.asarray(R).T
                #print 'R',R
                #print
                Rs.append(R)

        else:
            # (Camera coords: looking forward -z, up +y, right +x)
            R = cgtypes.mat3().identity()

            if 1:
                # (looking forward -z, up +x, right -y)
                R = R.rotation(-pi/2,(0,0,1))

                # (looking forward +x, up +z, right -y)
                R = R.rotation(-pi/2,(0,1,0))

                # rotate to point -theta (with up +z)
                Rs = [ R.rotation( float(th)+pi, (0,0,1) ) for th in theta ]
                #Rs = [ R for th in theta ]
            else:
                Rs = [ R.rotation( pi/2.0, (1,0,0) ) for th in theta ]
                #Rs = [ R for th in theta ]
            Rs = [ numpy.asarray(R).T for R in Rs ]
            print 'Rs',Rs

    # 2. intrinsic parameters

    for cam_no in range(n_cameras):
        cam_id = 'fake_%d'%(cam_no+1)

        # resolution of image
        res = (1600,1200)

        # principal point
        cc1 = res[0]/2.0
        cc2 = res[1]/2.0

        # focal length
        fc1 = 1.0
        fc2 = 1.0
        alpha_c = 0.0
#        R = numpy.asarray(Rs[cam_no]).T # conversion between cgkit and numpy
        R = Rs[cam_no]
        C = cam_centers[cam_no][:,numpy.newaxis]

        K = numpy.array((( fc1, alpha_c*fc1, cc1),
                         ( 0, fc2, cc2),
                         ( 0, 0, 1)))
        t = numpy.dot( -R, C )
        Rt = numpy.concatenate( (R,t), axis=1)
        P= numpy.dot( K, Rt )
        if 1:
            print 'cam_id',cam_id
            print 'P'
            print P
            print 'K'
            print K
            print 'Rt'
            print Rt
            print
            KR = numpy.dot(K,R)
            print 'KR',KR
            K3,R3 = reconstruct.my_rq(KR)
            print 'K3'
            print K3
            print 'R3'
            print R3
            K3R3 = numpy.dot(K3,R3)
            print 'K3R3',K3R3

            print '*'*60

        scc = reconstruct.SingleCameraCalibration_from_basic_pmat(P,
                                                                  cam_id=cam_id,
                                                                  res=res,
                                                                  scale_factor=1.0)
        sccs.append(scc)
        if 1:
            # XXX test
            K2,R2 = scc.get_KR()
            if 1:
                print 'C',C
                print 't',t
                print 'K',K
                print 'K2',K2
                print 'R',R
                print 'R2',R2
                print 'P',P
                print 'KR|t',numpy.dot(K,Rt)
                t2 = scc.get_t()
                print 't2',t2
                Rt2 = numpy.concatenate((R2,t2),axis=1)
                print 'KR2|t',numpy.dot(K2,Rt2)
                print
            KR2 = numpy.dot(K2,R2)
            KR  = numpy.dot(K,R)
            if not numpy.allclose( KR2, KR):
                if not numpy.allclose( KR2, -KR):
                    raise ValueError('expected KR2 and KR to be identical')
                else:
                    print 'WARNING: weird sign error in calibration math FIXME!'
    recon = reconstruct.Reconstructor(sccs)
    return recon

def test():
    generate_calibration()

if __name__=='__main__':
    test()

