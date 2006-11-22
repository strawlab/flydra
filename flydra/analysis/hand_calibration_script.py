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
        [86, 458, 50,0,0],
        [550,309,80,10,0],
        [552,153,80,20,0],
        [86, 155, 50, 20,0],
        [182, 384, 60, 10, z],
        [444, 386, 70, 10, z],
        [449, 120, 70, 20, z],
        [182, 123, 60, 20, z],
        [84.9, 308, 50, 10, 0],
        [235, 309, 60, 10, 0],
        [393, 310, 70, 10, 0],
        [550, 309, 80, 10, 0],
        [85.5, 155, 50, 20, 0],
        [237, 153, 60, 20, 0],
        [395, 152, 70, 20, 0],
        [552, 152, 80, 20, 0],
        [180, 382, 60, 10, z],
        [444, 385, 70, 10, z],
        [181, 121, 60, 20, z],
        [447, 118, 70, 20, z],
        ]

    cam2 = [
        [275, 267, 10, 10, 0],
        [559, 266, 50, 10, 0],
        [644, 150, 60, 20, 0],
        [269, 177, 10, 20, 0],
        [545, 319, 70, 10, z],
        [535, 151, 70, 20, z],
        [41.4, 300, 10, 10, z],
        [32.8, 194, 10, 20, z],
        [326, 450, 50, 0, z],
        [280, 350, 10, 0, 0],
        ]

    cam3 = [
        [38,244, 20, 10, 0],
        [599, 254, 80, 0, 0],
        [605, 79.3, 80, 20, 0],
        [17.3, 138, 20, 20, 0],
        [338, 96.1, 50, 20, 0],
        [143, 406, 30, 10, z],
        [532, 484, 50, 0, z],
        [557, 131, 50, 20, z],
        [96.7, 185, 30, 20, z],
        ]

    cam4 = [
        [231,267, 80, 10, 0],
        [552,412, 40, 0, 0],
        [547,162, 40, 20, 0],
        [228, 170, 80, 20, 0],
        [54, 434, 60, 0, z],
        [456,356,20,10,z],
        [450, 160, 20, 20, z],
        [43, 172, 60, 20, z],
        ]

    cam5 = [
        [25,373,40,10,0],
        [517,365, 10,10,0],
        [512,196,10,20,0],
        [18,209,40,20,0],
        [7,462,30,10,z],
        [579,454,10,10,z],
        [571,159,10,20,z],
        [278,166,20,20,z],
        [179,205,30,20,0],
        [352,370,20,10,0],
        ]

    cams = [(cam1,'cam1'),
            (cam2,'cam2'),
            (cam3,'cam3'),
            (cam4,'cam4'),
            (cam5,'cam5'),
            ]

    for (cam,cam_name) in cams:
        cam = numpy.array(cam, dtype=numpy.float)
        X3d = cam[:,2:5]*10.0 # put in mm
        x2d = cam[:,0:2]
        B,c = build_Bc(X3d,x2d)
        DLT_avec_results = numpy.linalg.lstsq(B,c)
        a_vec,residuals = DLT_avec_results[:2]
        Mhat = numpy.array(list(a_vec)+[1])
        Mhat.shape=(3,4)
        print cam_name,center(Mhat).T,'residuals:',float(residuals)
        fname = cam_name + 'DLT.txt'
        save_ascii_matrix(fname,Mhat)
