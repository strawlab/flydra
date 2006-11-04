import result_browser
import numpy
nx = numpy
ma = numpy.ma
import FOE_utils
import PQmath
import math
import time
time_fmt = '%Y-%m-%d %H:%M:%S %Z%z'

(all_results, all_results_times, trigger_fnos,
 logfile_trig_times, condition_floats) = FOE_utils.get_results_and_times(logfiles,h5files)
N_triggers = len(trigger_fnos)
print '%d FOE triggers'%N_triggers

strict_file = open('strict_data.txt','w')
lax_file = open('lax_data.txt','w')
for analysis_file in [strict_file, lax_file]:
    print >> analysis_file, '#pre_frames = %d; post_frames = %d; max_IFI_dist_mm = %f'%(
        pre_frames, post_frames, max_IFI_dist_mm)
    print >> analysis_file, '#landed_check_OK=%s; landed_max_z=%f'%(
        landed_check_OK,landed_max_z)
    print >> analysis_file, '#h5files = %s'%(repr(h5files),)
    print >> analysis_file, '#logfiles = %s'%(repr(logfiles),)
    print >> analysis_file, '#'
    print >> analysis_file, '#upwind, fstart, trig_fno, fend, results.filename, condition_float'

RAD2DEG = 180.0/math.pi

def my_masked_nonzero( m ):
    assert len(m.shape)==1 # not working on ndarrays yet
    nz = nx.nonzero(m.data)
    notmasked = ~m.mask[nz]
    newnz = nz[notmasked]
    return newnz

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

count = 0
good_count = 0
strict_count = 0
for idx in range(N_triggers):
    condition_float = condition_floats[idx]
    trig_fno = int(trigger_fnos[idx])
##    if trig_fno != 447748:
##        continue
    
    logfile_trig_time = logfile_trig_times[idx]
    found_results = []
    found_timestamps = []
    for results in all_results:
        data3d = results.root.data3d_best
        for row in data3d.where( data3d.cols.frame == trig_fno ):
            found_timestamps.append( row['timestamp'] )
            found_results.append( results )
    if not len(found_results):
        # no data saved for this trigger
        print 'no data for trigger at frame',trig_fno
        continue
    elif len(found_results) > 1:
        fts = nx.array(found_timestamps)
        diff_timestamps = abs( fts-logfile_trig_time)
        idx = nx.argmin(diff_timestamps)
        results = found_results[idx]
        #print 'using %s'%results.filename, fts
        #raise ValueError('frame %d found in > 1 .h5 file'%trig_fno)
    else:
        results = found_results[0]
    data3d = results.root.data3d_best
    camn2cam_id, cam_id2camns = result_browser.get_caminfo_dicts(results)
    
    fstart = trig_fno-pre_frames
    fend = trig_fno+post_frames

    frame_rels = nx.arange(pre_frames+post_frames+1)-pre_frames
    
    tmp_array = nx.ones(frame_rels.shape,nx.Float)
    tmp_mask = tmp_array>0
    
    xm = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    ym = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
    zm = ma.masked_array( tmp_array.copy(), mask=tmp_mask.copy())
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
        xm[j] = row['x']
        ym[j] = row['y']
        zm[j] = row['z']
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
        print ('WARNING: trigger time within that saved, but no data '
               'from those frames saved.')
        continue

    if 0:
        # display h5 file information
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
            if xm[i] == ma.masked:
                xm_str = 'nan'
            else:
                xm_str = '%f'%xm[i]
            if mean_dist[i] == ma.masked:
                mean_dist_str = 'nan'
            else:
                mean_dist_str = '%f'%mean_dist[i]
            try:
                print '%s %d (%d), x %s, (err %s), %s'%(timestamp_str,
                                                        fno, frame_rels[i], xm_str,
                                                        mean_dist_str,
                                                        ' '.join(s_cams_used))
            except TypeError,x:
                print 'type(xm[i]),xm[i]',type(xm[i]),xm[i]
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

    post_trig_cond = frame_rels>0
    xm_pretrig = ma.masked_where( post_trig_cond, xm )
    zm_pretrig = ma.masked_where( post_trig_cond, zm )
    xdiff = xm_pretrig[1:]-xm_pretrig[:-1]
    sum_xdiff = ma.sum( xdiff )

    tts = time.strftime(time_fmt, time.localtime(logfile_trig_time))
    print 'logfile_trig_time %s (fno %d):'%(tts,trig_fno),
    
    if not isinstance(sum_xdiff,float):
        # not enough data in pretrigger time
        print 'missing pretrigger tracking data'
        continue
        
    mean_pretrig_z = numpy.mean( zm_pretrig.compressed())
    assert isinstance( mean_pretrig_z, float)
    if mean_pretrig_z <= 70.0:
        print 'Mean pretrigger altitude <= 70mm, discarding'
        continue

    if sum_xdiff >= 0:
        upwind = True
    else:
        upwind = False

    if landed_check_OK:
        # heuristic to check for landed flies
        landed_cond = zm < landed_max_z
        if 0:
            # check for 3 consecutive frames
            # buggy because tmp is masked
            tmp = landed_cond.astype(numpy.int_)
            conseq = tmp[:-2] + tmp[1:-1] + tmp[2:]
            landed = conseq == 3 # indexes first frame of above in zm array
        else:
            landed = landed_cond
        landed_idx = my_masked_nonzero( landed )
        # discard potential hits before trigger
        landed_idx = landed_idx[landed_idx>pre_frames]
##        if trig_fno == 447748:
##            print 'zm',zm
##            print 'landed_cond',landed_cond
##            print 'landed',landed
##            print 'my_masked_nonzero( landed )',my_masked_nonzero( landed )
##            print 'landed_idx',landed_idx
        
        if landed_idx.shape[0] > 0:
            print 'landed',
            # landed at some point
            first_landed_idx = landed_idx[0]
##            if trig_fno == 447748:
##                print 'first_landed_idx',first_landed_idx
##            if trig_fno == 447748:
##                print 'xm.shape[0]',xm.shape[0]
##                print 'fstart',fstart
##                print 'fend',fend
##                print 'fend-fstart',fend-fstart
        else:
            first_landed_idx = None # no landing
    else:
        first_landed_idx = None # didn't check for landing
            
    # find frame-to-frame distance in mm
    IFI_dist_mm = ma.sqrt((xm[1:]-xm[:-1])**2 + (ym[1:]-ym[:-1])**2 + (zm[1:]-zm[:-1])**2)
    large_dist_cond = IFI_dist_mm > max_IFI_dist_mm
    large_dist_idx = my_masked_nonzero( large_dist_cond )
    if first_landed_idx is None:
        if len(large_dist_idx):
            print '>%f mm found between adjacent frames (%d frames found)'%(
                max_IFI_dist_mm,len(xm))
            continue
    else:
        if not len(large_dist_idx):
            # no frames have jump
            last_landed_idx = len(xm)-1
        else:
            # there is a jump in data, go to last possible landed data before jump
            if large_dist_idx[0] >= first_landed_idx:
                last_landed_idx = large_dist_idx[0]
                print 'large IFI distance after landing, truncating data',
            else:
                print '>%f mm found between adjacent frames (%d frames found)'%(
                    max_IFI_dist_mm,len(xm))
                continue
    
    if first_landed_idx is not None:
        xm = xm[:last_landed_idx]
        ym = ym[:last_landed_idx]
        zm = zm[:last_landed_idx]
        headings = headings[:last_landed_idx]
        mean_dist = mean_dist[:last_landed_idx]
        timestamps = timestamps[:last_landed_idx]
        fend = fstart+last_landed_idx
            
##    if trig_fno == 447748:
##        print 'xm',xm

    print 'OK'

    delta_t = 0.01
    Px = xm/1000.0 # in meters
    Py = ym/1000.0 # in meters
    Pz = zm/1000.0 # in meters
    xvels= (Px[2:]-Px[:-2]) / (2*delta_t)
    yvels= (Py[2:]-Py[:-2]) / (2*delta_t)
    zvels= (Pz[2:]-Pz[:-2]) / (2*delta_t)

    # make copies to save in average

    # reshape for averaging
    xm.shape = 1,xm.shape[0]
    ym.shape = 1,ym.shape[0]
    zm.shape = 1,zm.shape[0]
    xvels.shape = 1,xvels.shape[0]
    yvels.shape = 1,yvels.shape[0]
    zvels.shape = 1,zvels.shape[0]
    headings.shape = 1,headings.shape[0]

    ultra_strict = True
    if xm.shape[1]==xm.count():
        # meets ultra_strict requirements
        strict_count += 1
        print >> strict_file, upwind, fstart, trig_fno, fend, results.filename, condition_float
    strict_file.flush()
    print >> lax_file, upwind, fstart, trig_fno, fend, results.filename, condition_float
    lax_file.flush()
    if not ultra_strict or xm.shape[1]==xm.count():
        xs_dict[upwind].setdefault( condition_float, []).append( xm )
    if not ultra_strict or ym.shape[1]==ym.count():
        ys_dict[upwind].setdefault( condition_float, []).append( ym )
    if not ultra_strict or zm.shape[1]==zm.count():
        zs_dict[upwind].setdefault( condition_float, []).append( zm )

    if not ultra_strict or xvels.shape[1]==xvels.count():
        xvels_dict[upwind].setdefault( condition_float, []).append( xvels )
    if not ultra_strict or yvels.shape[1]==yvels.count():
        yvels_dict[upwind].setdefault( condition_float, []).append( yvels )
    if not ultra_strict or zvels.shape[1]==zvels.count():
        zvels_dict[upwind].setdefault( condition_float, []).append( zvels )

    if not ultra_strict or headings.shape[1]==headings.count():
        heading_dict[upwind].setdefault( condition_float, []).append( headings )
    good_count += 1
    
print >> strict_file, '# %d triggers in this file because they met all strictness criteria'%strict_count
print >> strict_file, '# %d triggers would be possible if willing to accept missing data'%good_count
strict_file.close()

print >> lax_file, '# lax data -- all accepted'
lax_file.close()

print '%d triggers in this file because they met all strictness criteria'%strict_count
print '%d triggers would be possible if willing to accept missing data'%good_count
