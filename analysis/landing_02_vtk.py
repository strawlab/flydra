import result_browser
import numpy
import pylab
import vtk_results
import vtk.util.colors as colors
post_diameter = 10 # mm

post = [( 466.8, 191.6, 15.8),# bottom
        ( 467.6, 212.7, 223.4)] # top

def main(max_err=10.0):
    results = result_browser.get_results('DATA20060502_211811.h5',mode='r')
    fstart = 67000
    fend = 70000
    post_top = numpy.array(post[1])
    
    renWin, renderers = vtk_results.init_vtk(stereo=True)
    if 1:
        camera = renderers[0].GetActiveCamera()
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
    vtk_results.show_frames_vtk(results,renderers,fstart,fend,1,
                                render_mode='ball_and_stick',
                                labels=True,
                                orientation_corrected=False,
                                #use_timestamps=True,
                                #bounding_box=bounding_box,
                                #frame_no_offset=fstart+pre_frames,
                                show_warnings=False,
                                max_err=max_err,
                                #color_change_frame=trig_fno,
                                #ball_color1=ball_color1,
                                #line_color1=ball_color1,
                                #ball_color2=ball_color2,
                                #line_color2=ball_color2,
                                #X_zero_frame=X_zero_frame,
                                )
    vtk_results.show_line(renderers,post[0],post[1],colors.black,1.0)
    vtk_results.interact_with_renWin(renWin)
    
if __name__=='__main__':
    main()
