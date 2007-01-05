
def gaussian(x,sigma):
    return nx.exp(-x**2/sigma**2)

def save_ecc(results):
    # old eccentricity debugging code -- should delete!
    import flydra.reconstruct, flydra.undistort
    ioff()
    try:
        rcn = flydra.reconstruct.Reconstructor(results)
        cam_id = 'cam4:0'
        camn = 19
        use_roi2 = True
        #for roi2_radius in [5,10,15]:
        for roi2_radius in [5]:
            for fno in range(7440+549,7440+550):
    #        for fno in range(10000+600,10000+640):
                print fno
                clf()
                frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                                          fno, camn,
                                                                          9.0, 0.2 )
                rt.roi2_radius = roi2_radius
                points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
                bright_point = rt.get_last_bright_point()
                wi = rt.get_working_image()
                pt = points[0]
                x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4 = pt

                x = x0_abs
                y = y0_abs

                xmin = bright_point[0]-roi2_radius
                xmax = bright_point[0]+roi2_radius
                ymin = bright_point[1]-roi2_radius
                ymax = bright_point[1]+roi2_radius

                a=slope
                b=-1
                c=y-a*x

                x1=xmin
                y1=-(c+a*x1)/b
                if y1 < ymin:
                    y1 = ymin
                    x1 = -(c+b*y1)/a
                elif y1 > ymax:
                    y1 = ymax
                    x1 = -(c+b*y1)/a

                x2=xmax
                y2=-(c+a*x2)/b
                if y2 < ymin:
                    y2 = ymin
                    x2 = -(c+b*y2)/a
                elif y2 > ymax:
                    y2 = ymax
                    x2 = -(c+b*y2)/a 

                title('%d %f'%(fno,eccentricity))
                imshow(flydra.undistort.undistort(rcn,wi,cam_id),
                       interpolation='nearest',origin='lower')
                plot([x0_abs],[y0_abs],'o',mfc=None,mec='white',mew=2)
                plot([x1,x2],[y1,y2],'w-',lw=2)
                setp(gca(),'xlim',[xmin,xmax])
                setp(gca(),'ylim',[ymin,ymax])
                savefig('roi%02d_ecc%d_%s.png'%(roi2_radius,fno,cam_id))
    finally:
        ion()

        
def wi_test2(results):
    import flydra.reconstruct, flydra.undistort
    rcn = flydra.reconstruct.Reconstructor(results)
    cam_id = 'cam4:0'
    camn = 19
    use_roi2 = True
    roi2_radius = 15
    fno = 7440+549
    res = {}
    vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for clear_threshold in vals:
        frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                                  fno, camn,
                                                                  9.0, clear_threshold )
        rt.roi2_radius = roi2_radius
        points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
        
        area = points[0][2]
        print clear_threshold, area

        bright_point = rt.get_last_bright_point()
        wi = rt.get_working_image()

        xmin = bright_point[0]-roi2_radius
        xmax = bright_point[0]+roi2_radius
        ymin = bright_point[1]-roi2_radius
        ymax = bright_point[1]+roi2_radius

        wi2 = wi[ymin:ymax,xmin:xmax].astype(nx.Float)
        
        res[clear_threshold] = wi2
    return res
    
def get_wi(results):
    import numarray.convolve as conv_mod
    import flydra.reconstruct, flydra.undistort
    rcn = flydra.reconstruct.Reconstructor(results)
    cam_id = 'cam4:0'
    camn = 19
    use_roi2 = True
    roi2_radius = 25
    fno = 7440+549
    frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                              fno, camn,
                                                              9.0, 0.2 )
    rt.roi2_radius = roi2_radius
    points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
    bright_point = rt.get_last_bright_point()
    wi = rt.get_working_image()
    
    xmin = bright_point[0]-roi2_radius
    xmax = bright_point[0]+roi2_radius
    ymin = bright_point[1]-roi2_radius
    ymax = bright_point[1]+roi2_radius

    wi2 = wi[ymin:ymax,xmin:xmax].astype(nx.Float)
    print 'points',points
    print 'bright_point',bright_point

    kernel = gaussian( nx.arange(5)-2, 2.0 )
    kernel = kernel/ nx.sum( kernel ) # normalize

    wi3 = []
    for row in range(wi2.shape[0]):
        res = conv_mod.convolve(wi2[row,:], kernel, mode=conv_mod.VALID )
        wi3.append( res )
    wi3 = nx.array(wi3)

    wi4 = []
    for col in range(wi3.shape[1]):
        res = conv_mod.convolve(wi3[:,col], kernel, mode=conv_mod.VALID )
        wi4.append( res )
    wi4 = nx.array(wi4)
    wi4.transpose()
    
    return wi, wi2, wi4
