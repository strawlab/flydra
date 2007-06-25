import numpy
import os

# XXX Note that a no-skew constraint might be used: http://users.rsise.anu.edu.au/~hartley/Papers/algebraic/ICCV/final/algebraic.pdf Minimizing Algebraic Error in Geometric Estimation Problems by Hartley

# Implementation of DLT method
# http://www.miba.auc.dk/~lasse/publications/HTML/pilot/cam_cal/camcal.html
# see also the pages at http://kwon3d.com/theory/calib.html
# see also http://users.rsise.anu.edu.au/~hartley/Papers/algebraic/ICCV/final/algebraic.pdf
print '*'*80
print '*'*80
print 'Need to implement normalization, a la Hartley & Zisserman, Algorithm 4.2'
print '*'*80
print '*'*80
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
    # there is also a copy of this in flydra.reconstruct, but included
    # here so this file doesn't depend on that.
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
    a = 0 # arbitrary number
    b = a+1
    c = a+2
    wall = a+3 # wall
    another_sample = a+4 # wall
    times = {a:'163644',
             b:'164010',
             c:'164129',
             another_sample:'122435',
             }
    datebytime = {times[a]:'20061218',
                  times[b]:'20061218',
                  times[c]:'20061218',
                  times[another_sample]:'20061219',
                  }
                  
    ft2inch = 12.0
    inch2cm = 2.54
    ft2cm = ft2inch*inch2cm
    
    # cam_x cam_y world_x world_y world_z
    
    cam1 = [
        # point A
        [422, 488, 151, 76, 43, a, 1],
        [ 323, 484, 177.5, 76, 43, a, 2],
        [ 427, 342, 151, 76, 81, a, 5],
        [435, 351, 151, 114, 81, a, 8],
        #point B
        [347, 397, 178, 95, 67, b, 1 ],
        [257, 394, 204.5, 95, 67, b, 2],
        [376, 284, 178, 133, 105, b, 8],
        [304, 281, 204.5, 133, 105, b, 7],
        #point C
        [241, 388, 198, 64, 66, c, 2],
        [264, 396, 198, 90.5, 66, c, 1],
        [275, 264, 198, 90.5, 104, c, 5],
        # back wall
        [44.4, 163, 10*ft2cm, 5*ft2cm, 5*ft2cm, wall, 0],
        ]
    cam1 = numpy.array(cam1, dtype=numpy.float)

    cam2 = [
        # point A
        [216,423,151, 114, 43,a,4],
        [337,428,151,76,43,a,1],
        [341,337, 151, 76, 81,a,5],
        [343,282,177.5,76, 81,a,6],
        # point B
        [295,323,178, 95, 67, b,1],
        [166,225, 178,133, 105, b,8],
        [293,224, 178, 95, 105, b,5],
        [294, 179, 204.5, 95, 105,b,6],
        # point C
        [303,283, 198, 90.5, 66,c,1],
        [380,283, 198, 64, 66,c,2],
        [296,182, 198, 90.5, 104,c,5],
        [301,131, 236, 90.5, 104,c,8],
        # wall
        [498, 250, 10*ft2cm, 0, 0, wall, 0],
        ]
    cam2 = numpy.array(cam2, dtype=numpy.float)

    cam3 = [
        # point A
        [304,387, 151, 76,43 ,a,1],
        [226,389, 177.5, 76,43 ,a,2],
        #[303,391, 151, 76, 81,a,5], # bad point
        [221,293, 177.5, 76, 81,a,6],
        [302,236, 151, 114, 81,a,8],
        [226,237, 177.5, 114, 81,a,7],
        # point B
        [235, 308, 178,95 ,67 ,b,1],
        [159, 305, 204.5, 95, 67,b,2],
        [230, 207, 178,95 , 105,b,5],
        [150, 206, 204.5, 95, 105,b, 6],
        [179,252, 204.5, 133, 67,b,3],
        # point C
        [166,317, 198, 90.5, 66,c,1],
        [155,356, 198, 64, 66,c,2],
        [161,211, 198, 90.5, 104,c,5],
        [44,207, 236, 90.5, 104 ,c,8],
        # wall
        [643,345, 0, 5*ft2cm, 0,wall,0],
        ]
    cam3 = numpy.array(cam3, dtype=numpy.float)

    cam4 = [
        # calibrated the next day
        [120,475,10*ft2cm,0,0,another_sample,0],
        [522,460,10*ft2cm,5*ft2cm,0,another_sample,0],
        [497,69.6,10*ft2cm,5*ft2cm,5*ft2cm,another_sample,0],
        [120,76.2,10*ft2cm,0,5*ft2cm,another_sample,0],
        [344, 200, 178, 85, 86, another_sample,0], # feeder suspension
        [349,298, 178, 85, 63, another_sample,0], # feeder bottom
        ]
    cam4 = numpy.array(cam4, dtype=numpy.float)

    #### simple cal ###

    pbox = [
        [435,341,0,0,0],
        [464,179,84,0,0],
        [464,179,84,0,0],
        [279,352,0,106,0],
        [242,221,0,0,140],
        [287,35,84,0,140],
        [123, 82, 84, 106, 140],
        [77,245,0,106,140],
        ]
    pbox = numpy.array(pbox, dtype=numpy.float)

    cams = [(cam1,'cam1_0'),
            (cam2,'cam2_0'),
            (cam3,'cam3_0'),
            (cam4,'cam4_0'),
            ]
    cams = [(pbox,'pbox'),
            ]

    sccs = []

    for (cam,cam_id) in cams:
        ## put in mm
        #cam[:,2:5] = cam[:,2:5]*10.0
        # print 'multiplied by 10 to put in mm'
        
        X3d = cam[:,2:5]
        x2d = cam[:,0:2]
        B,c = build_Bc(X3d,x2d)
        DLT_avec_results = numpy.linalg.lstsq(B,c)
        a_vec,residuals = DLT_avec_results[:2]
        Mhat = numpy.array(list(a_vec)+[1])
        Mhat.shape=(3,4)

        print cam_id,center(Mhat).T,'residuals:',float(residuals)
        fname = cam_id + 'DLT.txt'
        save_ascii_matrix(fname,Mhat)

        if 1:
            import flydra.reconstruct

            res = 640,480
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
                scale_factor = 1e-3, # convert these units to meters
                )
            sccs.append(scc)
            
    if len(sccs):
        reconstructor = flydra.reconstruct.Reconstructor(sccs)
        reconstructor.save_to_files_in_new_directory('cal_hand')


# replot
if 0:
    import FlyMovieFormat
    import pylab
    for scc in sccs:
        for this_time in times:
            time_str = times[this_time]
            date = datebytime[time_str]
            fname = 'full_%s_%s_%s.fmf'%(date,time_str,scc.cam_id)
            #print fname

            if not os.path.exists(fname):
                print fname,'does not exist, skipping'
                continue
            fly_movie = FlyMovieFormat.FlyMovie(fname)
            frame,timestamp = fly_movie.get_frame(0)

            pylab.figure()
            pylab.imshow(frame,interpolation='nearest',origin='lower')
            xlim = pylab.gca().get_xlim()
            pylab.gca().set_xlim(xlim[1],xlim[0])
            pylab.title(fname)
            
            for (pts, cam_id) in cams:
                if cam_id != scc.cam_id:
                    continue
                for pts_row in pts:
                    if pts_row[5] != this_time:
                        if pts_row[5] != wall:
                            continue
                    #print pts_row[:5]

                    world_coord = pts_row[2:5]
                    cam_coord = pts_row[:5]
                    #print 'world_coord',world_coord
                    recon_cam = reconstructor.find2d(scc.cam_id, world_coord)
                    #print cam_coord[:2],'->',recon_cam
                    pylab.plot( [ cam_coord[0], recon_cam[0] ],
                                [ cam_coord[1], recon_cam[1] ],
                                'w-' )
                    pylab.plot( [ cam_coord[0] ],
                                [ cam_coord[1]],
                                'go' )
                    pylab.plot( [ recon_cam[0] ],
                                [ recon_cam[1] ],
                                'ro' )
                                  
            #print
    pylab.show()
