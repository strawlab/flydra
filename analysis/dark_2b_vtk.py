import result_browser
import vtk
import numpy as nx
import PQmath
import math, glob, time, sys, os

import vtk_results

import vtk.util.colors as colors

import datetime
import pytz # from http://pytz.sourceforge.net/
pacific = pytz.timezone('US/Pacific')

# find segments to use
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'strict_data.txt'
print 'opening',fname
analysis_file = open(fname,'r')
f_segments = [line.strip().split() for line in analysis_file.readlines() if not line.strip().startswith('#')]

renWin, renderers = vtk_results.init_vtk()#stereo=True)

horsetail = False
#horsetail = True

condition = 'dark'
#condition = 'strobe_once'
#condition = 'focal_once'

#dir = 'upwind'
#dir = 'downwind'

if 1:
    # draw 3-color axes
    axes = vtk.vtkAxes()
    axes.ComputeNormalsOff()
    if horsetail:
        axes.SetOrigin(0, 50, -50)
    else:
        axes.SetOrigin(720, 230, 0)
    axes.SetScaleFactor(100)
    axesMapper = vtk.vtkPolyDataMapper()
    axesMapper.SetInput(axes.GetOutput())
    axesActor = vtk.vtkActor()
    axesActor.SetMapper(axesMapper)
    for renderer in renderers:
        renderer.AddActor( axesActor )

h5files = {}
rough_timestamps = {}

did_bbox = False
count = 0
max_n_frames = 0
for line_no,line in enumerate(f_segments):
    upwind_orig, fstart, trig_fno, fend, h5filename, stimulus_condition = line

    if upwind_orig == 'False':
        upwind = False
    elif upwind_orig == 'True':
        upwind = True
    else:
        raise ValueError('hmm')

##    if dir=='upwind' and not upwind:
##        continue
    
##    if dir=='downwind' and upwind:
##        continue

    fstart = int(fstart)
    trig_fno = int(trig_fno)
    fend = int(fend)
    n_frames = fend-fstart
    max_n_frames = max(n_frames,max_n_frames)

    if 0:
        if condition == 'dark':
            if stimulus_condition!='0':
                continue
        elif condition == 'strobe_once':
            if stimulus_condition!='S1':
                continue
        elif condition == 'focal_once':
            if stimulus_condition!='F1':
                continue

    ball_color1 = colors.purple
    ball_color2 = colors.purple
    
    if h5filename not in h5files:
        h5files[h5filename] = result_browser.get_results(h5filename)
        results = h5files[h5filename]
        if 1:
            data3d = results.root.data3d_best
            for row in data3d:
                ts_float = row['timestamp']
                dt_ts = datetime.datetime.fromtimestamp(ts_float,pacific)
                rough_timestamps[h5filename] = dt_ts
                break
    results = h5files[h5filename]
    rough_timestamp = rough_timestamps[h5filename]

    if stimulus_condition=='0':
        ball_color1 = colors.white
        ball_color2 = colors.black
    elif stimulus_condition=='S1':
        ball_color1 = colors.white
        ball_color2 = colors.red
    elif stimulus_condition=='F1':
        ball_color1 = colors.white
        ball_color2 = colors.purple
        

    if horsetail:
        X_zero_frame=trig_fno
        bounding_box=False
    else:
        X_zero_frame=None
        if 0:
            bounding_box=not did_bbox
            if not did_bbox:
                did_bbox=True
        else:
            bounding_box=False

    vtk_results.show_frames_vtk(results,renderers,fstart,fend,1,
                                render_mode='ball_and_stick',
                                labels=False,#True,
                                orientation_corrected=False,
                                #use_timestamps=True,
                                bounding_box=bounding_box,
                                #frame_no_offset=fstart+pre_frames,
                                show_warnings=False,
                                max_err=10,
                                color_change_frame=trig_fno,
                                ball_color1=ball_color1,
                                line_color1=ball_color1,
                                ball_color2=ball_color2,
                                line_color2=ball_color2,
                                X_zero_frame=X_zero_frame,
                                )
    count += 1
##    if count==10:
##        break
print 'examining %d traces'%count


if 1:
    if not horsetail:
        # show tunnel walls
        y0 = 40 # calibration offset
        y1 = y0+12*25.4
        z0 = 0
        z1 = 12*25.4
        x0 = 0
        x1 = 6*12*25.4

        v0 = (x0,y0,z0)
        v1 = (x1,y0,z0)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)

        v0 = (x0,y1,z0)
        v1 = (x1,y1,z0)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
        
        v0 = (x0,y0,z1)
        v1 = (x1,y0,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)

        v0 = (x0,y1,z1)
        v1 = (x1,y1,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)

        v0 = (x0,y0,z0)
        v1 = (x0,y0,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
                              
        v0 = (x0,y1,z0)
        v1 = (x0,y1,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
        
        v0 = (x1,y1,z0)
        v1 = (x1,y1,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
        
        v0 = (x1,y0,z0)
        v1 = (x1,y0,z1)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
                              
        v0 = (x0,y0,z0)
        v1 = (x0,y1,z0)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
                              
        v0 = (x1,y0,z0)
        v1 = (x1,y1,z0)
        vtk_results.show_line(renderers,v0,v1,colors.black,0.4)
        
if 0:
    # show weather balloons
    vel = -0.4 # meters per sec
    fps = 100.0 # frames per second
    mm_per_frame = vel*1000.0/fps
    X = nx.arange(max_n_frames)*mm_per_frame 
    X = X-nx.mean(X)
    if not horsetail:
        offset = nx.array([ 780.0, y1, z0] )
    else:
        offset = nx.array([ 0.0, -150.0, 0] )
    X = [ (offset[0]+x, offset[1], offset[2]) for x in X ]
    vtk_results.show_spheres(renderers,X)
                              

if horsetail:
    fp = nx.zeros((3,),dtype=nx.Float)
else:
    fp = nx.array([700,150,160],dtype=nx.Float)
    
camera = renderers[0].GetActiveCamera()
camera.SetFocalPoint (*tuple(fp))
camera.SetViewUp (0,0,1)
camera.SetClippingRange (127.81089961095051, 1824.5666015625093)
if 1:
    camera.SetParallelProjection(0)
    camera.SetViewAngle(40.0)
else:
    camera.SetParallelProjection(1)
    camera.SetParallelScale(160)

if 1:
    D2R = math.pi/180.0
    nfs = 30 # number of frames scaled
    azs = nx.r_[
        nx.linspace( 225.0, 270.0, nfs ),
        nx.linspace( 270.0, 180.0, nfs*2 ),
        nx.linspace( 180.0, 225.0, nfs*2 ),
        ]
    azs = azs[:-1] # last frame is repeat
    azs = azs*D2R
    
    # set initial value for interactive viewing
    az = azs[0]
    offset = 500*math.cos(az), 500*math.sin(az), 0
    camera.SetPosition (*tuple(fp+offset))

if 0:
    vtk_results.show_cameras(results,renderers)
    
if 1:
    vtk_results.interact_with_renWin(renWin)#,renderers)
    for renderer in renderers:
        vtk_results.print_cam_props(renderer.GetActiveCamera())
        
else:
    
    save_to_file = True
    
    if save_to_file:
        imf = vtk.vtkWindowToImageFilter()
        imf.SetInput(renWin)
        imf.Update()
    #while 1:
    fname_base = 'frame%04d.png'
    #fname_base = 'frame%04d.jpg'
    if 1:
        # delete old frames
        files = os.listdir('.')
        for i in range(10000):
            fname = fname_base%i
            if fname in files:
                print 'deleting',fname
                os.unlink(fname)
    if 1:
        # generate new frames
        for frame_no,az in enumerate(azs):
            for renderer in renderers:
                camera = renderer.GetActiveCamera()
                offset = 500*math.cos(az), 500*math.sin(az), 0
                camera.SetPosition (*tuple(fp+offset))
                renderer.ResetCameraClippingRange()
            renWin.Render()
            if save_to_file:

                if fname_base.endswith('.png'):
                    writer = vtk.vtkPNGWriter()
                elif fname_base.endswith('.jpg'):
                    writer = vtk.vtkJPEGWriter()
                    writer.SetQuality(100) # max
                else:
                    raise ValueError("don't know format")
                
                #writer.SetInput(imf.GetOutput())

                imf.Modified()

                writer.SetInput(imf.GetOutput())
                fname = fname_base%(frame_no+1,)
                print 'saving',fname
                writer.SetFileName(fname)
                writer.Write()
    if 1:
        w,h = renWin.GetSize()
        #mpeg_name = 'movie_%s_%s.mpeg'%(condition,dir)
        mpeg_name = 'movie_%s.mpeg'%(condition,)
        cmd = 'ffmpeg -hq -b 20000 -f mpeg2video -r 30 -s %dx%d -i %s %s'%(
            w,h,fname_base,mpeg_name)
        print 'executing:',cmd
        os.system(cmd)

# Ubuntu:
# mencoder "mf://frame*.png" -mf fps=15 -nosound -ovc lavc -lavcopts vcodec=mpeg2video:vbitrate=15000 -of mpeg -noskip -o last_15k.mpg

# debian sarge64:
# ffmpeg -hq -b 15000 -f mpeg2video -r 30 -i 'frame%04d.png' last_ffmpeg.mpg
