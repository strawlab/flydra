import result_browser
import numpy
import vtk_results
import vtk.util.colors as colors

def main(max_err=10.0):
    post_diameter = 10 # mm

    if 0:
        # 20060502
        filename = 'DATA20060502_211811.h5'
        post = [( 466.8, 191.6, 15.8),# bottom
                ( 467.6, 212.7, 223.4)] # top
    elif 0:
        filename = 'DATA20060315_170142.h5'
        # from ukine with recalibration
        post = [
            ( 863.6, 248.0,-22.5),
            ( 854.6, 239.2, 210.7)]
        if 1:
            fstart = 168235-100
            fend = fstart+1000
        else:
            flight = 'A'
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
    elif 1:
        # 20060515
        filename = 'DATA20060515_190905.h5'
        post = [( 418.3, 192.3, 18.4),
                (434.3, 205.6, 222.2),
                ]
##        if 0:
##            fstart = 811962
##            fend = fstart+1612
            
##        elif 0:
##            fstart = 123757-100
##            fend = fstart+400
##        elif 0:
##            fstart = 137555-16000
##            fend = fstart+18400
##        elif 0:
##            fstart = 212541-500
##            fend = fstart+400
##        elif 0:
##            fstart = 5454018-100
##            fend = fstart+400

        if 0:
            pass
        elif 0: # "near_landing"
            fstart = 873955-200
            fend = fstart+600

        elif 0: # "descent"
            fstart = 1019607-200
            fend = fstart+600
            
        elif 0:
            fstart = 800887-200
            fend = fstart+600
        elif 0:
            fstart = 565708-200
            fend = fstart+600
            
        elif 0: # nice slow approach towards post
            fstart,fend = 86354,86994
        elif 0: # take off
            fstart = 295161-200
            fend = fstart+600
        elif 0: # descent
            fstart = 429129-200
            fend = fstart+600
        elif 0: # nice slow approach towards post
            fstart = 440493-200
            fend = fstart+600
        elif 0:
            fstart,fend = 1021266,1021378
        elif 1:
            fstart,fend = 316418,317533
    elif 0:
        # 20060516 head fixed
        filename = 'DATA20060516_194920.h5'
        # from ukine with recalibration
        1/0
        post = [( 444.9, 246.9,  8.9),
                ( 457.2, 243.3, 227.2),
                ]
        if 0:
            fstart = 4989909
            fend = 4990076
        elif 0:
            fstart,fend = 4885653,4885782 # bouncing off ceiling
        elif 1:
            fstart,fend = 4676487,4676678
            
    results = result_browser.get_results(filename,mode='r')
    post_top = numpy.array(post[1])
    
    renWin, renderers = vtk_results.init_vtk()#stereo=True)
    if 1:
        camera = renderers[0].GetActiveCamera()
        if 0:
            camera.SetFocalPoint (*tuple(post_top))
            camera.SetPosition (*tuple(post_top+numpy.array((100.0,0,0))))
            camera.SetViewUp (0,0,1)
            camera.SetClippingRange (127.81089961095051, 1824.5666015625093)
            if 1:
                camera.SetParallelProjection(0)
                camera.SetViewAngle(40.0)
            else:
                camera.SetParallelProjection(1)
                camera.SetParallelScale(160)
        elif 0:
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (808.43166362536795, 184.67205062508307, 151.75948665695657 )
            camera.SetPosition (850.58344054989595, 297.63051569911852, 137.89990793211075)
            camera.SetViewAngle(40.0)
            camera.SetViewUp (0.048377407666256686, 0.10383595509816884, 0.99341719375916981 )
            camera.SetClippingRange (7.9558204083045361, 795.58204083045359)
            camera.SetParallelScale(319.400653668)
        elif 0:
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (802.76620083254261, 187.74006842929163, 158.43756755570874)
            camera.SetPosition (852.94940952562899, 298.237113583819, 157.71394500772169)
            camera.SetViewAngle(40.0)
            camera.SetViewUp (0.0097668683103851132, 0.0021127590084137594, 0.99995007101993849)
            camera.SetClippingRange (8.090446917224277, 809.04469172242773)
            camera.SetParallelScale(319.400653668)
        elif 0:
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (803.1303085674632, 187.51926171773758, 153.83304155478751)
            camera.SetPosition (848.74023523537619, 299.96934187210161, 152.04723226142741)
            camera.SetViewAngle(40.0)
            camera.SetViewUp (0.013526602921373277, 0.010392185583130462, 0.99985450616187799)
            camera.SetClippingRange (7.8983022785496155, 789.83022785496155)
            camera.SetParallelScale(319.400653668)
        elif 1:
            camera = renderers[0].GetActiveCamera()
            camera.SetParallelProjection(0)
            camera.SetFocalPoint (426.50702447336062, 165.88903952531726, 168.54898337291434)
            camera.SetPosition (897.11470783001005, -423.58860888119352, 368.86388791475287)
            camera.SetViewAngle(15.0)
            camera.SetViewUp (-0.06424622629673464, 0.27472176826859618, 0.95937499052560493)
            camera.SetClippingRange (476.57608671532716, 1161.5197421319072)
            camera.SetParallelScale(319.400653668)
        elif 0:
            import flydra.reconstruct
            import flydra.geom
            if 0:
                cam_cal_filename = 'photron.scc'
                scci = flydra.reconstruct.SingleCameraCalibration_fromfile(cam_cal_filename)
            else:
                recon = flydra.reconstruct.Reconstructor(results)
                #scci = recon.get_SingleCameraCalibration('cam2:0')
                scci = recon.get_SingleCameraCalibration('cam5_0')
            center = scci.get_cam_center()[:,0]
            up = scci.get_up_vector()
            optical_axis = scci.get_optical_axis()

            if 1:
                # find point on optical axis closest to post top
                post_top_tt = flydra.geom.ThreeTuple( post_top )
                shifted_optical_axis = optical_axis.translate(-post_top_tt)
                nearest_post_top_shifted = shifted_optical_axis.closest()
                nearest_post_top = nearest_post_top_shifted + post_top_tt
                focal = nearest_post_top
            else:
                tunnel_midplane = flydra.geom.Plane(
                    flydra.geom.ThreeTuple((0,-1,0)), 150)
                focal = optical_axis.intersect(tunnel_midplane)

            print 'center',center
            print 'focal',focal
            print 'up',up

            camera.SetPosition(*center)
            camera.SetFocalPoint(*focal)
            camera.SetViewUp(*up)
            camera.SetParallelProjection(0)
            camera.SetViewAngle(15)

    vtk_results.show_frames_vtk(results,renderers,fstart,fend,1,
                                render_mode='ball_and_stick',
                                
                                labels=True,
                                #labels=False,
                                
                                orientation_corrected=False,
                                #orientation_corrected=True,
                                
                                bounding_box=False,
                                #frame_no_offset=fstart+pre_frames,
                                show_warnings=False,
                                max_err=max_err,
                                show_debug_info=True,
                                
                                #color_change_frame=trig_fno,
                                #ball_color1=ball_color1,
                                #line_color1=ball_color1,
                                #ball_color2=ball_color2,
                                #line_color2=ball_color2,
                                #X_zero_frame=X_zero_frame,
                                )
    results.close()
    vtk_results.show_line(renderers,post[0],post[1],colors.black,post_diameter/2)
    vtk_results.interact_with_renWin(renWin)
    
    camera = renderers[0].GetActiveCamera()
    vtk_results.print_cam_props(camera)
    
if __name__=='__main__':
    main()
