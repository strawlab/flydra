import numpy
# cam1



# implementation of DLT method
# http://www.miba.auc.dk/~lasse/publications/HTML/pilot/cam_cal/camcal.html
# see also the pages at http://kwon3d.com/theory/calib.html
# see also http://users.rsise.anu.edu.au/~hartley/Papers/algebraic/ICCV/final/algebraic.pdf
def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )
        
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

def center(P):
    orig_determinant = numpy.linalg.det
    def determinant( A ):
        return orig_determinant( numpy.asarray( A ) )
    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )
    
    C_ = numpy.array( [[ X/T, Y/T, Z/T ]] ).T
    return C_

if 1:
    z = 31

    cam1 = [
        [83,288,30,10,0],
        [192,289,40,10,0],
        [561, 293, 70, 10, 0],
        [562, 162, 70, 20, 0],
        [432,419,60,0,0],
        [80,169,30,20,0],
        [149,340,50,10,z],
        [514,346,70,10,z],
        [515,151,70,20,z],
        [146,158,50,20,z],
        ]

    cam2 = [
        [282,350,10,0,0],
        [337,265,20,10,0],
        [652,264,60,10,0],
        [646,148,60,20,0],
        [215,179,0,20,0],
        [30,295,10,10,z],
        [82,297,20,10,z],
        [517,471,70,0,z],
        [638,136,80,20,z],
        [21,191,10,20,z],
        ]

    cam3 = [
        [92,296,20,10,0],
        [190,396,30,0,0],
        [494,392,60,0,0],
        [593,288,70,10,0],
        [594,188,70,20,0],
        [93,195,20,20,0],
        [76,318,30,10,z],
        [607,485,60,0,z],
        [609,130,60,20,z],
        [252,316,40,10,z],
        ]

    cam4 = [
        [104,414,40,0,0],
        [424,268,80,10,0],
        [281,388,60,0,0],
        [427,171,80,20,0],
        [108,164,40,20,0],
        [199,283,50,10,0],
        [160,360,20,10,z],
        [587,303,60,10,z],
        [644,416,70,0,z],
        [593,168,60,20,z],
        [165,154,20,20,z],
        ]

    cam5 = [
        [141,378,20,10,z],
        [501,364,40,10,z],
        [509,183,40,20,z],
        [145,177,20,20,z],
        [83,321,20,10,0],
        [559,312,60,10,0],
        [563,196,60,20,0],
        [447,435,50,0,0],
        [87,188,20,20,0],
        [212,319,30,10,0],
        ]

    cams = [(cam1,'cam1_0'),
            (cam2,'cam2_0'),
            (cam3,'cam3_0'),
            (cam4,'cam4_0'),
            (cam5,'cam5_0'),
            ]

    sccs = []

    for (cam,cam_id) in cams:
        cam = numpy.array(cam, dtype=numpy.float)
        if 1:
            # on 20061121, the upper images were shifted by Y-2
            print 'performing Y correction'
            zcoord = cam[:,4]
            cond = zcoord==z
            cam[:,3] = cam[:,3]+2
        
        X3d = cam[:,2:5]*10.0 # put in mm
        x2d = cam[:,0:2]
        B,c = build_Bc(X3d,x2d)
        DLT_avec_results = numpy.linalg.lstsq(B,c)
        a_vec,residuals = DLT_avec_results[:2]
        Mhat = numpy.array(list(a_vec)+[1])
        Mhat.shape=(3,4)
        print cam_id,center(Mhat).T,'residuals:',float(residuals)
        fname = cam_id + 'DLT.txt'
        save_ascii_matrix(fname,Mhat)

        if 0:
            if cam_id == 'cam2_0':
                all_rows = numpy.arange( cam.shape[0] )
                all_X3d = cam[:,2:5]*10.0 # put in mm
                all_x2d = cam[:,0:2]
                
                for test_bad_row in range(cam.shape[0]):
                    test_good_rows = all_rows[ all_rows != test_bad_row ]

                    
                    X3d = all_X3d[test_good_rows]
                    x2d = all_x2d[test_good_rows]
                    B,c = build_Bc(X3d,x2d)
                    DLT_avec_results = numpy.linalg.lstsq(B,c)
                    a_vec,residuals = DLT_avec_results[:2]
                    Mhat = numpy.array(list(a_vec)+[1])
                    Mhat.shape=(3,4)

                    print 'test_bad_row',test_bad_row
                    print 'residuals',residuals
                    print cam[test_bad_row,:]
                    print
                    
        if 1:
            import flydra.reconstruct

            res = 656,491
            pp = res[0]/2., res[1]/2.
            print 'cam_id',cam_id
            print 'assuming res',res
            print 'assuming pp',pp
            print
            scc = flydra.reconstruct.SingleCameraCalibration(
                cam_id=cam_id,
                Pmat = Mhat,
                res = res,
                pp = pp,
                helper = None, ## XXX should extract focal length?
                scale_factor = 1e-3)
            sccs.append(scc)
            
    if len(sccs):
        reconstructor = flydra.reconstruct.Reconstructor(sccs)
        reconstructor.save_to_files_in_new_directory('cal_20061121_hand')
                
