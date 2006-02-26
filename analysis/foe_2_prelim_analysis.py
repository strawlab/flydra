import result_browser
import numpy
nx = numpy
ma = numpy.ma
import FOE_utils
import PQmath
import math
import time
time_fmt = '%Y-%m-%d %H:%M:%S %Z%z'

import glob

# wind
h5files = glob.glob('*.h5')
logfiles = ['accepted_triggers.txt']

(all_results, all_results_times, trigger_fnos,
 logfile_trig_times, tf_hzs) = FOE_utils.get_results_and_times(logfiles,h5files)
N_triggers = len(trigger_fnos)
print '%d FOE triggers'%N_triggers

analysis_file = open('strict_data.txt','w')

RAD2DEG = 180.0/math.pi

xs_dict = {} # sort by upwind
ys_dict = {}
zs_dict = {}
heading_dict = {}
xvels_dict = {}
yvels_dict = {}
zvels_dict = {}

for upwind in [True,False]:
    xs_dict[upwind] = {} # sort by tf
    ys_dict[upwind] = {}
    zs_dict[upwind] = {}
    heading_dict[upwind] = {}
    xvels_dict[upwind] = {}
    yvels_dict[upwind] = {}
    zvels_dict[upwind] = {}

all_IFI_dist_mm = []

count = 0
good_count = 0
strict_count = 0
for idx in range(N_triggers):
    tf_hz = tf_hzs[idx]
    trig_fno = int(trigger_fnos[idx])
    logfile_trig_time = logfile_trig_times[idx]
    found_results = None
    for results in all_results:
        data3d = results.root.data3d_best
        for row in data3d.where( data3d.cols.frame == int(trig_fno) ):
            if found_results is not None:
                raise ValueError('frame %d found in > 1 .h5 file'%trig_fno)
            found_results = results
    if found_results is None:
        # no data saved for this trigger
        print 'no data for trigger at frame',trig_fno
        continue
    results = found_results
    data3d = results.root.data3d_best
    camn2cam_id, cam_id2camns = result_browser.get_caminfo_dicts(results)
    
    pre_frames = 10
    post_frames = 30
    fstart = trig_fno-pre_frames
    fend = trig_fno+post_frames

    frame_rels = nx.arange(pre_frames+post_frames+1)-pre_frames
    
    tmp_array = nx.ones(frame_rels.shape,nx.Float)
    tmp_mask = tmp_array>0
    
    xs = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    ys = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    zs = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    headings = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    mean_dist = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    timestamps = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    del tmp_array
    del tmp_mask    

    camns_used = ['']*len(frame_rels)

    have_data = False
    for row in data3d.where( fstart <= data3d.cols.frame <= fend ):
        j = row['frame']-fstart
        # these unset the mask in the masked array:
        xs[j] = row['x']
        ys[j] = row['y']
        zs[j] = row['z']
        mean_dist[j] = row['mean_dist'] # 3d reconstruction error estimate
        timestamps[j] = row['timestamp']
        camns_used[j] = row['camns_used']

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

    if 0:
        data2d = results.root.data2d
        
        for i in range(len(frame_rels)):
            camns = map(int,camns_used[i].split())
            s_cams_used = [ camn2cam_id[camn] for camn in camns]
            s_cams_used.sort()
            fno = int(frame_rels[i]+trig_fno) # cast from numpy scalar
            if timestamps[i] == ma.masked:
                timestamp_str = 'nan'
            else:
                timestamp_str = repr(timestamps[i])
            if xs[i] == ma.masked:
                xs_str = 'nan'
            else:
                xs_str = '%f'%xs[i]
            if mean_dist[i] == ma.masked:
                mean_dist_str = 'nan'
            else:
                mean_dist_str = '%f'%mean_dist[i]
            try:
                print '%s %d (%d), x %s, (err %s), %s'%(timestamp_str,
                                                        fno, frame_rels[i], xs_str,
                                                        mean_dist_str,
                                                        ' '.join(s_cams_used))
            except TypeError,x:
                print 'type(xs[i]),xs[i]',type(xs[i]),xs[i]
                print 'type(mean_dist[i]),mean_dist[i]',type(mean_dist[i]),mean_dist[i]
                raise
            apparent_recon_latencies = []
            did_camns = []
            for row in data2d.where( data2d.cols.frame == fno):
                camn = row['camn']
                print '  ',camn2cam_id[camn],row
                if camn not in did_camns:
                    apparent_recon_latencies.append( timestamps[i] - row['timestamp'] )
                    did_camns.append( camn )
            print '  apparent latencies (msec):',numpy.array(apparent_recon_latencies)*1000.0

    count += 1
    if 1:
        # old names
        xm = xs
        ym = ys
        zm = zs

    xm_pretrig = ma.masked_where( frame_rels >0, xm )
    zm_pretrig = ma.masked_where( frame_rels >0, zm )
    xdiff = xm_pretrig[1:]-xm_pretrig[:-1]
    sum_xdiff = ma.sum( xdiff )

    delta_t = 0.01
    Px = xm/1000.0 # in meters
    Py = ym/1000.0 # in meters
    Pz = zm/1000.0 # in meters
    xvels= (Px[2:]-Px[:-2]) / (2*delta_t)
    yvels= (Py[2:]-Py[:-2]) / (2*delta_t)
    zvels= (Pz[2:]-Pz[:-2]) / (2*delta_t)

    if not isinstance(sum_xdiff,float):
        # not enough data
        print 'logfile_trig_time %s (fno %d) missing pretrigger tracking data'%(repr(logfile_trig_time),trig_fno)
        continue
    else:
        tts = time.strftime(time_fmt, time.localtime(logfile_trig_time))
        print 'logfile_trig_time %s (fno %d):'%(tts,trig_fno),
    mean_pretrig_z = numpy.mean( zm_pretrig.compressed())

    assert isinstance( mean_pretrig_z, float)

    if sum_xdiff >= 0:
        upwind = True
    else:
        upwind = False

    # find frame-to-frame distance in mm
    IFI_dist_mm = ma.sqrt((xm[1:]-xm[:-1])**2 + (ym[1:]-ym[:-1])**2 + (zm[1:]-zm[:-1])**2)
    nonmaskedlist = list( IFI_dist_mm.compressed() )

    if max(nonmaskedlist) > 20:
        print '>20 mm found between adjacent frames (%d frames found)'%(len(xm),)
        continue
    elif mean_pretrig_z <= 70.0:
        print 'Mean pretrigger altitude <= 70mm, discarding'
        continue
    else:
        print 'OK'

    all_IFI_dist_mm.extend( nonmaskedlist )

    #print 'all_IFI_dist_mm',all_IFI_dist_mm

    # make copies to save in average
    xm_tmp = xm[:]
    ym_tmp = ym[:]
    zm_tmp = zm[:]
    xvels_tmp = xvels[:]
    yvels_tmp = yvels[:]
    zvels_tmp = zvels[:]

    headings_tmp = headings[:]

    xm_tmp.shape = 1,xm_tmp.shape[0]
    ym_tmp.shape = 1,ym_tmp.shape[0]
    zm_tmp.shape = 1,zm_tmp.shape[0]
    xvels_tmp.shape = 1,xvels_tmp.shape[0]
    yvels_tmp.shape = 1,yvels_tmp.shape[0]
    zvels_tmp.shape = 1,zvels_tmp.shape[0]
    headings_tmp.shape = 1,headings_tmp.shape[0]

    ultra_strict = True
    if xm_tmp.shape[1]==xm_tmp.count():
        # meets ultra_strict requirements
        strict_count += 1
        print >> analysis_file, upwind, fstart, trig_fno, fend, results.filename, tf_hz
    if not ultra_strict or xm_tmp.shape[1]==xm_tmp.count():
        xs_dict[upwind].setdefault( tf_hz, []).append( xm_tmp )
    if not ultra_strict or ym_tmp.shape[1]==ym_tmp.count():
        ys_dict[upwind].setdefault( tf_hz, []).append( ym_tmp )
    if not ultra_strict or zm_tmp.shape[1]==zm_tmp.count():
        zs_dict[upwind].setdefault( tf_hz, []).append( zm_tmp )

    if not ultra_strict or xvels_tmp.shape[1]==xvels_tmp.count():
        xvels_dict[upwind].setdefault( tf_hz, []).append( xvels_tmp )
    if not ultra_strict or yvels_tmp.shape[1]==yvels_tmp.count():
        yvels_dict[upwind].setdefault( tf_hz, []).append( yvels_tmp )
    if not ultra_strict or zvels_tmp.shape[1]==zvels_tmp.count():
        zvels_dict[upwind].setdefault( tf_hz, []).append( zvels_tmp )

    if not ultra_strict or headings_tmp.shape[1]==headings_tmp.count():
        heading_dict[upwind].setdefault( tf_hz, []).append( headings_tmp )
    good_count += 1
    
print >> analysis_file, '# %d triggers in this file because they met all strictness criteria'%strict_count
print >> analysis_file, '# %d triggers would be possible if willing to accept missing data'%good_count
analysis_file.close()

print '%d triggers in this file because they met all strictness criteria'%strict_count
print '%d triggers would be possible if willing to accept missing data'%good_count

if 0:
    import pylab
    for yvals_upwind_and_downwind in [xs_dict,
                                      ys_dict,
                                      zs_dict,
                                      xvels_dict,
                                      yvels_dict,
                                      zvels_dict,
                                      heading_dict]:
        for doing_upwind in (True,False):
            fig_yvel = pylab.figure()
            ax_yvel = fig_yvel.add_subplot(1,1,1)

            yvals_dict = yvals_upwind_and_downwind[doing_upwind]
            tf_hzs = yvals_dict.keys()

            for tf_hz in tf_hzs:

                #if tf_hz == 1.0:
                #    continue

                val_list = yvals_dict[tf_hz]
                val_array = ma.concatenate( val_list, axis=0 )
                # each column is a timepoint
                means = nx.zeros( (val_array.shape[1],), nx.Float )
                stds = nx.zeros( (val_array.shape[1],), nx.Float )
                minN = 1e30
                maxN = 0
                for j in range( val_array.shape[1] ):
                    col = val_array[:,j]
                    if isinstance(col,ma.MaskedArray):
                        col_compressed = val_array[:,j].compressed()
                    else:
                        col_compressed = col
                    N = len(col_compressed)
                    if N < minN: minN = N
                    if N > maxN: maxN = N
                    if N==0:
                        print 'WARNING: mean not computed, N==0'
                        continue
                    means[j] = numpy.mean( col_compressed )
                    if N==1:
                        continue
                    stds[j] = numpy.std( col_compressed )

                if (yvals_upwind_and_downwind is xvels_dict or
                    yvals_upwind_and_downwind is yvels_dict or
                    yvals_upwind_and_downwind is zvels_dict):
                    # velocities have fewer data points
                    xdata = 10.0*frame_rels[1:-1]
                else:
                    xdata = 10.0*frame_rels

                if doing_upwind: key = 'upwind flight, '
                else: key = 'downwind flight, '

                Nstr = 'n=%d'%(minN,)
                if minN!=maxN:
                    # if ultra_strict = False, this can happen
                    print 'WARNING: minN!=MaxN (%d, %d)'%(minN,maxN)
                    Nstr = 'n=%d-%d'%(minN,maxN)

                if tf_hz == 0.0:
                    key += 'no expansion, '
                elif tf_hz == 1.0:
                    key += '0.1 m/sec expanding pattern, '
                elif tf_hz == 5.0:
                    key += '0.5 m/sec expanding pattern, '
                key += Nstr

                line,=ax_yvel.plot( xdata, means, lw=2, label=key )

                x1 = xdata
                y1 = means+stds
                y2 = means-stds

                # reverse x and y2 so the polygon fills in order
                x = nx.concatenate( (x1,x1[::-1]) )
                y = nx.concatenate( (y1,y2[::-1]) )

                p = ax_yvel.fill(x, y, facecolor=pylab.getp(line,'color'), linewidth=0)
                pylab.setp(p, alpha=0.5)

            pylab.setp(ax_yvel, 'xlabel','Time relative to expansion onset (msec)')
            pylab.setp(ax_yvel, 'xlim',[-pre_frames*10,post_frames*10])

            if yvals_upwind_and_downwind is xvels_dict:
                pylab.setp(ax_yvel, 'ylabel','Longitudinal speed of fly in WT (m/sec)')
            elif yvals_upwind_and_downwind is yvels_dict:
                pylab.setp(ax_yvel, 'ylabel','Lateral speed of fly in WT (m/sec)')
            elif yvals_upwind_and_downwind is zvels_dict:
                pylab.setp(ax_yvel, 'ylabel','Vertical speed of fly in WT (m/sec)')

            elif yvals_upwind_and_downwind is xs_dict:
                pylab.setp(ax_yvel, 'ylabel','X pos fly in WT (mm)')
            elif yvals_upwind_and_downwind is ys_dict:
                pylab.setp(ax_yvel, 'ylabel','Y pos fly in WT (mm)')
            elif yvals_upwind_and_downwind is zs_dict:
                pylab.setp(ax_yvel, 'ylabel','Z pos fly in WT (mm)')

            elif yvals_upwind_and_downwind is heading_dict:
                pylab.setp(ax_yvel, 'ylabel','heading of fly (?)')
            else:
                pylab.setp(ax_yvel, 'ylabel','unknown variable')
            pylab.legend()
            ax_yvel.grid(True)
        
    pylab.show()

