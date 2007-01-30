import pickle
import numpy
import vtk_results
import flydra.reconstruct
import vtk.util.colors as colors
import PQmath
import cgtypes

if 1:
    stereo=False

    renWin, renderers = vtk_results.init_vtk(stereo=stereo)
    camera = renderers[0].GetActiveCamera()
    if 1:
        camera.SetParallelProjection(0)
        camera.SetFocalPoint (454.5249156600417, 225.47636954408182, 109.52533015392012)
        camera.SetPosition (413.70426149461372, 74.567002631882403, 133.73690403992671)
        camera.SetViewAngle(15.0)
        camera.SetViewUp (0.076509671824590592, 0.13773815471317677, 0.98750922570551325)
        camera.SetClippingRange (4.1956543377518649, 419.56543377518648)
        camera.SetParallelScale(319.400653668)

    #ORIG= True
    #ORIG= False
    ORIG= True
    if ORIG:
        fd = open('fXl.pkl','rb')
        fXl = pickle.load(fd)
        fd.close()

        frame = fXl[:,0].astype(numpy.int64)
        X = fXl[:,1:4]
        line3d = fXl[:,4:]

        # create Pluecker line between points
        XQ = []
        for i in range(len(frame)-1):
            
            Xi = X[i]
            Udir = cgtypes.vec3(*(X[i+1]-Xi))
            U = Udir.normalize()
            
            Q = PQmath.orientation_to_quat( (U[0],U[1],U[2]) )

            XQ.append( list(Xi) + [Q.w, Q.x, Q.y, Q.z] )
        fd = open('XQ.txt',mode='w')
        for xq in XQ:
            fd.write( ' '.join( map(repr,xq) ) + '\n' )
        fd.close()
        
        #sys.exit()
            
        
        L = line3d
        #L = line3d[numpy.newaxis,:] # Plucker coordinates
        #print frame,'L',L
        U = flydra.reconstruct.line_direction(line3d)
        
        if 1:
            # flip orientations so that one end is always down
            negU_cond = (U[:,2] < 0.0)
            #U[negU_cond] = -U[negU_cond]
            line3d[negU_cond] = -line3d[negU_cond]
            U = flydra.reconstruct.line_direction(line3d)
            orientation_corrected = True

            if 1:
                fXl[:,4:] = line3d
                import pickle
                fd = open('fXl-fixed.pkl',mode='wb')
                pickle.dump( fXl, fd )
                fd.close()
        else:
            orientation_corrected = False
    else:
        import scipy.io
        results = scipy.io.loadmat('smoothed')
        print results
        frame = results['frames']
        psmooth = results['psmooth']
        qsmooth = results['qsmooth']
        
        print 'frame.shape',frame.shape
        print 'qsmooth.shape',qsmooth.shape

        X = psmooth
        qsmooth = qsmooth.T
        qsmooth = PQmath.QuatSeq([cgtypes.quat(q) for q in qsmooth])
            
        U = PQmath.quat_to_orient(qsmooth)
        orientation_corrected = True
        print 'X.shape',X.shape
        print 'U.shape',U.shape

    if 1:
        tube_length = 0.004 # meters ( 4 mm)
        
    if orientation_corrected:
        pt1 = X-tube_length*U
        pt2 = X
    else:
        pt1 = X-tube_length*.5*U
        pt2 = X+tube_length*.5*U

    print pt1[0], pt2[0]

    plot_scale = 1000.0
    actors = vtk_results.show_longline(renderers,
                                       X*plot_scale,
                                       radius = 0.001*plot_scale,
                                       nsides=3,opacity=0.5,
                                       color=colors.lime_green)
    
    actors = vtk_results.show_tubes(renderers,
                                    pt1*plot_scale,
                                    pt2*plot_scale)
    
    
    vtk_results.interact_with_renWin(renWin)
    
    camera = renderers[0].GetActiveCamera()
    vtk_results.print_cam_props(camera)
