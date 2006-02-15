import result_browser
import numpy as nx
import PQmath
import math, glob, time

import vtk_results
    
# find segments to use
#analysis_file = open('downwind_strict_data.txt','r')
analysis_file = open('upwind_strict_data.txt','r')
f_segments = [line.strip().split() for line in analysis_file.readlines()]

renWin, renderers = vtk_results.init_vtk(stereo=True)

camera = renderers[0].GetActiveCamera()
camera.SetParallelProjection(1)
camera.SetFocalPoint (672.12908756192337, 138.75279457807446, 108.6435815205222)
camera.SetPosition (295.51103702665949, -419.1937746759927, 703.55834107871351)
camera.SetViewAngle(30.0)
camera.SetViewUp (0.086021586810189621, 0.69895221378646744, 0.70997611892630108)
camera.SetClippingRange (127.81089961095051, 1824.5666015625093)
camera.SetParallelScale(349.904794877)

h5files = {}

did_bbox = False
for line in f_segments:
    fstart, trig_fno, fend, h5filename = line
    fstart = int(fstart)
    trig_fno = int(trig_fno)
    fend = int(fend)
    if h5filename not in h5files:
        h5files[h5filename] = result_browser.get_results(h5filename)
    results = h5files[h5filename]

    
    vtk_results.show_frames_vtk(results,renderers,fstart,fend,1,
                                render_mode='ball_and_stick',
                                labels=False,#True,
                                orientation_corrected=False,
                                use_timestamps=True,
                                bounding_box=not did_bbox,#bbox,
                            #frame_no_offset=fstart+pre_frames,
                                show_warnings=False,
                                max_err=10)
    if not did_bbox:
        did_bbox=True

if 0:
    vtk_results.show_cameras(results,renderers)
    
vtk_results.interact_with_renWin(renWin)#,renderers)
for renderer in renderers:
    vtk_results.print_cam_props(renderer.GetActiveCamera())



