import result_browser
import matplotlib.numerix.ma as M
from matplotlib.numerix.ma import array
import matplotlib.numerix.mlab as mlab
import matplotlib.numerix as nx
import PQmath
import math, glob

#pylab_or_vtk = 'vtk'
pylab_or_vtk = 'pylab'

if pylab_or_vtk == 'pylab':
    import pylab
else:
    import vtk_results
    
if 0:
    # still air
    h5files = [
        
        ]

    logfiles = [
                ]
else:
    # wind
    h5files = glob.glob('*.h5')
    logfiles = glob.glob('escape_wall2005*.log')

trig_fnos = {}
tf_hzs = {}
all_tf_hzs = []
for logfile in logfiles:
    fd = open(logfile,'rb')
    for line in fd.readlines():
        if line.startswith('#'):
            continue
        fno, ftime, tf_hz = line.split()
        fno = int(fno)
        ftime = float(ftime)
        tf_hz = float(tf_hz)
        trig_fnos[ftime] = fno
        tf_hzs[ftime] = tf_hz
        if tf_hz not in all_tf_hzs:
            all_tf_hzs.append( tf_hz )

print 'TFs of escape wall:', all_tf_hzs

all_results = [result_browser.get_results(h5file,mode='r+') for h5file in h5files]
all_results_times = [result_browser.get_start_stop_times( results ) for results in all_results ]

trig_times = trig_fnos.keys()
trig_times.sort()

RAD2DEG = 180.0/math.pi

if pylab_or_vtk == 'pylab':
    ax_x = None
else:
    renWin, renderers = vtk_results.init_vtk()

    camera = renderers[0].GetActiveCamera()
    camera.SetParallelProjection(1)
    camera.SetFocalPoint (672.12908756192337, 138.75279457807446, 108.6435815205222)
    camera.SetPosition (295.51103702665949, -419.1937746759927, 703.55834107871351)
    camera.SetViewAngle(30.0)
    camera.SetViewUp (0.086021586810189621, 0.69895221378646744, 0.70997611892630108)
    camera.SetClippingRange (127.81089961095051, 1824.5666015625093)
    camera.SetParallelScale(349.904794877)
    renderers[0].SetActiveCamera(camera)
    did_bbox = False


upwind_xs = []
upwind_ys = []
upwind_zs = []
upwind_heading = []
upwind_yvels = []
upwind_IFI_dist_mm = []

downwind_xs = []
downwind_ys = []
downwind_zs = []
downwind_heading = []
downwind_yvels = []
downwind_IFI_dist_mm = []

count = 0
good_count = 0
quitnow = False
for trig_time in trig_times:
    if quitnow:
        break
    tf_hz = tf_hzs[trig_time]
    if tf_hz != 5.0:
        continue
    for i in range(len(all_results)):
        results = all_results[i]
        results_start, results_stop = all_results_times[i]
        if results_start < trig_time < results_stop:
            trig_fno = trig_fnos[trig_time]
        else:
            continue

        print
        if pylab_or_vtk == 'pylab':
            if ax_x is None:
                fig_pos = pylab.figure()
                ax_x = fig_pos.add_subplot(4,1,1)

                ax_y = fig_pos.add_subplot(4,1,2, sharex=ax_x)
                ax_z = fig_pos.add_subplot(4,1,3, sharex=ax_x)
                ax_err = fig_pos.add_subplot(4,1,4, sharex=ax_x)
                
                fig_yvel = pylab.figure()
                ax_yvel = fig_yvel.add_subplot(2,1,1)
                ax_heading = fig_yvel.add_subplot(2,1,2)

        pre_frames = 50
        post_frames = 150
        fstart = trig_fno-pre_frames
        fend = trig_fno+post_frames

        data3d = results.root.data3d_best
        frame_rels = []

        frame_rels = nx.arange(pre_frames+post_frames+1)-pre_frames
        xs = nx.ones(frame_rels.shape,nx.Float)*-10000
        ys = nx.ones(frame_rels.shape,nx.Float)*-10000
        zs = nx.ones(frame_rels.shape,nx.Float)*-10000
        headings = nx.ones(frame_rels.shape,nx.Float)*-10000
        mean_dist = nx.ones(frame_rels.shape,nx.Float)*-10000
        
        have_data = False
        for row in data3d.where( fstart <= data3d.cols.frame <= fend ):
            j = row['frame']-fstart
            #frame_rel = row['frame'] - trig_fno
            xs[j] = row['x']
            ys[j] = row['y']
            zs[j] = row['z']
            mean_dist[j] = row['mean_dist']
            
            # get angular position phi
            orientation = nx.array((-row['p2'],row['p4'],-row['p5']))
            if str(orientation[0]) != 'nan':
                yaw, pitch = PQmath.orientation_to_euler( PQmath.norm_vec(orientation) )
                headings[j] = yaw*RAD2DEG
            
            have_data = True

        if not have_data:
            print ('WARNING: trigger time within that saved, no data '
                   'from those frames saved.')
            continue

        count += 1
        if 0:
            if count == 20:
                quitnow = True
                break

        if 0:
            xm = M.masked_where(xs < -9999, xs)
            ym = M.masked_where(ys < -9999, ys)
            zm = M.masked_where(zs < -9999, zs)
        else:
            # This is a hack. I shouldn't have to do this.
            xm = M.masked_outside(xs, -9999, 9999)
            ym = M.masked_outside(ys, -9999, 9999)
            zm = M.masked_outside(zs, -9999, 9999)
        headings = M.masked_where(headings < -9999, headings)

        xm = M.masked_where(mean_dist > 10, xm)
        ym = M.masked_where(mean_dist > 10, ym)
        zm = M.masked_where(mean_dist > 10, zm)

        mean_dist = M.masked_where(mean_dist < -9999, mean_dist)
        mean_dist = M.masked_where(mean_dist > 10, mean_dist)

        xm_time_of_interest = M.masked_where( frame_rels >0, xm )
        zm_time_of_interest = M.masked_where( frame_rels >0, zm )
        xdiff = xm_time_of_interest[1:]-xm_time_of_interest[:-1]
        sum_xdiff = M.sum( xdiff )

        yvels = ym[1:]-ym[:-1]

        if type(sum_xdiff) != float:
            print 'HMM, WARNING'
##            print 'WARNING: at trig_time %s, repr(sum_xdiff)='%(repr(trig_time),)
##            print repr(sum_xdiff)
##            print 'xm=\n',xm
##            print 'xm_time_of_interest=\n',xm_time_of_interest
##            print 'xdiff=\n',xdiff
##            print
            continue
        else:
            print 'trig_time %s OK (%f Hz)'%(repr(trig_time),tf_hz),count

        mean_pretrig_z = mlab.mean( zm_time_of_interest.compressed())
        if type(mean_pretrig_z) != float:
            print 'HMM, WARNING 2'
        else:
            if mean_pretrig_z <= 70.0:
                print 'Mean pretrigger altitude <= 70mm, discarding'
                continue
        
        
        if sum_xdiff >= 0:
            upwind = True
        else:
            upwind = False

        if 0:
            # total non-directional displacement
            
            # find frame-to-frame distance in mm
            IFI_dist_mm = M.sqrt((xm[1:]-xm[:-1])**2 + (ym[1:]-ym[:-1])**2 + (zm[1:]-zm[:-1])**2)
        else:
            # x displacement
            IFI_dist_mm = xm[1:]-xm[:-1]

        IFI_dist_mm = IFI_dist_mm /1000 * 100 # convert to meters/sec
        nonmaskedlist = list( IFI_dist_mm.compressed() )

        if max(nonmaskedlist) > 10:
            print 'WARNING: skipping because >10 mm found between adjacent frames'
            continue

        if upwind:
            upwind_IFI_dist_mm.extend( nonmaskedlist )
        else:
            downwind_IFI_dist_mm.extend( nonmaskedlist )
        
        #print 'all_IFI_dist_mm',all_IFI_dist_mm
        
        # make copies to save in average
        xm_tmp = xm[:]
        ym_tmp = ym[:]
        zm_tmp = zm[:]
        yvels_tmp = yvels[:]
        headings_tmp = headings[:]

        xm_tmp.shape = 1,xm_tmp.shape[0]
        ym_tmp.shape = 1,ym_tmp.shape[0]
        zm_tmp.shape = 1,zm_tmp.shape[0]
        yvels_tmp.shape = 1,yvels_tmp.shape[0]
        headings_tmp.shape = 1,headings_tmp.shape[0]

        if upwind:
            upwind_xs.append( xm_tmp )
            upwind_ys.append( ym_tmp )
            upwind_zs.append( zm_tmp )
            upwind_yvels.append( yvels_tmp )
            upwind_heading.append( headings_tmp )
        else:
            downwind_xs.append( xm_tmp )
            downwind_ys.append( ym_tmp )
            downwind_zs.append( zm_tmp )
            downwind_yvels.append( yvels_tmp )
            downwind_heading.append( headings_tmp )

        good_count += 1

        if pylab_or_vtk == 'pylab':
            if upwind:
                fmt = 'r-'
            else:
                fmt = 'b-'
            ax_x.plot( 10.0*frame_rels, xm, fmt)
            ax_y.plot( 10.0*frame_rels, ym, fmt)
            ax_z.plot( 10.0*frame_rels, zm, fmt)
            ax_err.plot( 10.0*frame_rels, mean_dist, fmt)

            ax_yvel.plot( 10.0*frame_rels[:-1], yvels, fmt)
            ax_heading.plot( 10.0*frame_rels, headings, fmt)

        else:
            if upwind:
                if did_bbox:
                    bbox = False
                else:
                    bbox = True
                vtk_results.show_frames_vtk(results,renderers,fstart,fend,1,
                                            render_mode='ball_and_stick',
                                            labels=False,#True,
                                            orientation_corrected=False,
                                            use_timestamps=True,
                                            bounding_box=bbox,
                                            frame_no_offset=fstart+pre_frames,
                                            show_warnings=False,
                                            max_err=10)

print 'good_count',good_count
if pylab_or_vtk == 'pylab':

    leg_lines = {}
    leg_labels = {}
    for (val_list,ax) in [(upwind_xs, ax_x),
                          (upwind_ys, ax_y),
                          (upwind_zs, ax_z),
                          (upwind_yvels, ax_yvel),
                          (upwind_heading, ax_heading),
                          ]:
        
        if len(val_list) == 0:
            continue
        val_array = M.concatenate( val_list, axis=0 )
        va_sum = M.sum( val_array, axis = 0 )
        va_N = val_array.shape[0]-M.sum( M.getmask(val_array), axis=0 )
        va_mean = va_sum/va_N

        if val_list is upwind_yvels:
            xdata = 10.0*frame_rels[:-1]
        else:
            xdata = 10.0*frame_rels
        lines = ax.plot( xdata, va_mean, 'r-', lw=3)
        leg_lines.setdefault(ax,[]).append( lines[0] )
        leg_labels.setdefault(ax,[]).append('upwind')
        
    for (val_list,ax) in [(downwind_xs, ax_x),
                          (downwind_ys, ax_y),
                          (downwind_zs, ax_z),
                          (downwind_yvels, ax_yvel),
                          (downwind_heading, ax_heading),
                          ]:

        if len(val_list) == 0:
            continue
        val_array = M.concatenate( val_list, axis=0 )
        va_sum = M.sum( val_array, axis = 0 )
        va_N = val_array.shape[0]-M.sum( M.getmask(val_array), axis=0 )
        va_mean = va_sum/va_N

        if val_list is downwind_yvels:
            xdata = 10.0*frame_rels[:-1]
        else:
            xdata = 10.0*frame_rels
        lines=ax.plot( xdata, va_mean, 'b-', lw=3)
        leg_lines.setdefault(ax,[]).append( lines[0] )
        leg_labels.setdefault(ax,[]).append('downwind')

    for ax in leg_lines.keys():
        ax.legend( leg_lines[ax], leg_labels[ax] )
        
    ax_x.set_ylabel('x position (mm)')
    ax_y.set_ylabel('y position (mm)')
    pylab.setp(ax_y,'ylim',[110,180])
    ax_z.set_ylabel('z position (mm)')
    ax_z.set_xlabel('time (msec)')
    
    ax_yvel.set_ylabel('yvel')
    ax_heading.set_ylabel('heading')
    ax_heading.set_xlabel('time (msec)')
    
    pylab.setp(ax_z,'ylim',[50,350])

    fig_vel_hist = pylab.figure()
    ax_hist_upwind = fig_vel_hist.add_subplot(2,1,1)
    ax_hist_downwind = fig_vel_hist.add_subplot(2,1,2,sharex=ax_hist_upwind)
    #print 'all_IFI_dist_mm',all_IFI_dist_mm
    ax_hist_upwind.hist(upwind_IFI_dist_mm, bins=100)
    ax_hist_upwind.set_xlabel('IFI distance (mm)')
    ax_hist_downwind.hist(downwind_IFI_dist_mm, bins=100)
    ax_hist_downwind.set_xlabel('IFI distance (mm)')
    
    pylab.show()
else:
    vtk_results.interact_with_renWin(renWin,renderers)
    for renderer in renderers:
        vtk_results.print_cam_props(renderer.GetActiveCamera())



