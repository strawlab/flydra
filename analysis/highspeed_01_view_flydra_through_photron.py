import result_browser
import numpy
import pylab
import flydra.reconstruct

def main(max_err=10.0):
    post_diameter = 10 # mm

    cam_cal_filename = 'photron.scc'

    if 0:
        # 20060502
        filename = 'DATA20060502_211811.h5'
        post = [( 466.8, 191.6, 15.8),# bottom
                ( 467.6, 212.7, 223.4)] # top
    elif 1:
        filename = 'DATA20060315_170142.h5'
        # from ukine with recalibration
        post = [( 864.1, 230.0, 17.6) ,
                ( 857.2, 225.2, 221.8)]
        flight = 'C'
        if flight=='A':
            # flight A.  bounce off post!!
            fstart = 148500
            fend = 149000
        elif flight=='B':
            # flight B
            fstart = 349300
            fend = fstart+400
        elif flight=='C':
            # flight C
            fstart = 421400
            fend = fstart+400
        elif flight=='D':
            # flight D.  very good
            fstart = 485000
            fend = fstart+400
        elif flight=='E':
            # flight E.  very good
            fstart = 725500
            fend = fstart+400
        elif flight=='F':
            # flight F.  very good
            fstart = 993890
            fend = 994070
    elif 0:
        # 20060515
        filename = 'DATA20060515_190905.h5'
        post = [( 471.5, 191.2, 22.7),
                ( 479.7, 205.1, 225.2),
                
                ]
        if 0:
            fstart = 369430
            fend = 377515
        elif 1:
            fstart = 374420
            fend = 374720
    elif 0:
        # 20060516 head fixed
        filename = 'newDATA20060516_194920.h5'
        post = [( 443.9, 247.1,  7.8),
                ( 456.9, 243.2, 226.7)
                ]
        if 1:
            fstart = 345796
            fend = 345998
    
        
    results = result_browser.get_results(filename,mode='r')
    post_top = numpy.array(post[1])
    
    scci = flydra.reconstruct.SingleCameraCalibration_fromfile(cam_cal_filename)
    cam_id = scci.cam_id
    recon = flydra.reconstruct.Reconstructor([scci])

    X = []
    data3d = results.root.data3d_best
    for row in data3d.where( fstart <= data3d.cols.frame <= fend ):
        X.append( (row['x'], row['y'], row['z'], 1.0) )
    X = numpy.asarray(X)
    x2d = recon.find2d(cam_id,X,distorted=True) # distort back to image
    xs = x2d[0,:]
    ys = x2d[1,:]
    ax = pylab.subplot(1,1,1)
    pylab.plot(xs,ys,'.')
    ax.set_xlim( (0, scci.res[0]) )
    ax.set_ylim( (1, scci.res[1]) )
    pylab.show()
    
if __name__=='__main__':
    main()
