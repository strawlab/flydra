import result_browser
import numpy
import flydra.reconstruct

# implementation of DLT method
# http://www.miba.auc.dk/~lasse/publications/HTML/pilot/cam_cal/camcal.html
# see also the pages at http://kwon3d.com/theory/calib.html
# see also http://users.rsise.anu.edu.au/~hartley/Papers/algebraic/ICCV/final/algebraic.pdf

def build_Bc(X3d,x2d):
    B = []
    c = []

    assert len(X3d)==len(x2d)
    if len(X3d) < 6:
        print 'WARNING: 2 equations and 11 unknowns means we need 6 points!'
    for i in range(len(X3d)):
        X = X3d[i,0]
        Y = X3d[i,1]
        Z = X3d[i,2]
        x = x2d[i,0]
        y = x2d[i,1]

        B.append( [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z] )
        B.append( [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z] )

        c.append( x )
        c.append( y )
    return numpy.array(B), numpy.array(c)
    

filename = 'DATA20060315_170142.h5'
results = result_browser.get_results(filename,mode='r')

frames = [993970,
          993960,
          993950,
          993910,
          421460,
          485201,
          148562,
          ]
w = 1
x2d = numpy.array([[428, 634, w], # 632
                   [432, 559, w], # 532
                   [424, 496, w], # 432
                   [60,  496, w], # 32
                   [968, 857, w], # frame 180
                   [939, 882, w], # frame 800
                   [454, 572, w], # frame 230
                   ],
                  dtype=numpy.Float64)
                  

X3d = []
for frame in frames:
    data3d = results.root.data3d_best
    found = False
    for row in data3d.where(data3d.cols.frame==frame):
        X=row['x'],row['y'],row['z'],1.0
        X3d.append(X)
        if found:
            raise RuntimeError("already found!")
        found = True
    if not found:
        raise RuntimeError("3d point not found!")
X3d=numpy.array(X3d)

if 1:
    # DLT method
    B,c = build_Bc(X3d,x2d)
    print '-='*20
    print B
    print c
    print '-='*20

    DLT_avec_results = numpy.linalg.lstsq(B,c)
    a_vec = DLT_avec_results[0]
    print a_vec
    print '-='*20
    Mhat = numpy.array(list(a_vec)+[1])
    Mhat.shape=(3,4)
    
    if 0:
        # calculate principal point from pages at
        # http://kwon3d.com/theory/calib.html
        # specifically http://kwon3d.com/theory/dlt/dlt.html eq. 27

        # This depends on orthogonality of extrinsic parameter matrix,
        # which wasn't a constraint imposed using our 11-parameter
        # method, so isn't expected to work. A modified DLT can
        # supposedly work.
        
        L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11 = a_vec
        D2 = 1.0/(L9**2 + L10**2 + L11**2)

        u0 = D2*(L1*L9 + L2*L10 + L3*L11)
        v0 = D2*(L5*L9 + L6*L10 + L7*L11)
        pp = u0,v0
    else:
        pp = None

else:
    print X3d
    print '-='*20
    print x2d
    print


    lstsq_results = numpy.linalg.lstsq(X3d,x2d)
    Mhat = lstsq_results[0].transpose()
    print Mhat
    pp = None

cam_id = 'photron'
scci = flydra.reconstruct.SingleCameraCalibration(
    cam_id=cam_id,
    Pmat = Mhat,
    res=(1024,1024),
    pp=pp)
recon = flydra.reconstruct.Reconstructor([scci])
fname = 'photron.scc'
K,R=scci.get_KR()
print '-='*20
print 'K',K

print 'R',R
scci.to_file(fname)

# test everything
scc2 = flydra.reconstruct.SingleCameraCalibration_fromfile(fname)
recon = flydra.reconstruct.Reconstructor([scci])
print recon.find2d(scc2.cam_id,X3d).transpose()

