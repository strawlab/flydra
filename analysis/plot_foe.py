import result_browser
#import matplotlib.numerix.ma as M
#from matplotlib.numerix.ma import array
import matplotlib.numerix.mlab as mlab
#import matplotlib.numerix as nx
import numpy
nx = numpy
M = numpy.ma
import FOE_utils
import PQmath
import math

import glob
import pylab

if 0:
    # still air
    h5files = [
        
        ]

    logfiles = [
                ]
else:
    # wind
    h5files = glob.glob('*.h5')
    logfiles = glob.glob('escape_wall*.log')

if 1:
    (all_results, all_results_times, trig_fnos,
     trig_times, tf_hzs) = FOE_utils.get_results_and_times(logfiles,h5files)
    print '%d FOE triggers'%len(trig_times)
else:
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
quitnow = False
for trig_time in trig_times:
    if quitnow:
        break
    tf_hz = tf_hzs[trig_time]
    for i in range(len(all_results)):
        results = all_results[i]
        results_start, results_stop = all_results_times[i]
        if results_start < trig_time < results_stop:
            trig_fno = trig_fnos[trig_time]
        else:
            continue

        print
        pre_frames = 10
        post_frames = 50
##        pre_frames = 50
##        post_frames = 150
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
            if count == 50:
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

        delta_t = 0.01
        Px = xm/1000.0 # in meters
        Py = ym/1000.0 # in meters
        Pz = zm/1000.0 # in meters
        xvels= (Px[2:]-Px[:-2]) / (2*delta_t)
        yvels= (Py[2:]-Py[:-2]) / (2*delta_t)
        zvels= (Pz[2:]-Pz[:-2]) / (2*delta_t)

        if not isinstance(sum_xdiff,float):
            # not enough data
            print 'trig_time %s (fno %d) missing tracking data %d'%(repr(trig_time),trig_fno,count)
##            print 'HMM, WARNING'
##            print 'WARNING: at trig_time %s, repr(sum_xdiff)='%(repr(trig_time),)
##            print repr(sum_xdiff)
##            print 'type(sum_xdiff)',type(sum_xdiff)
##            print 'xm=\n',xm
##            print 'xm_time_of_interest=\n',xm_time_of_interest
##            print 'xdiff=\n',xdiff
##            print
            continue
        else:
            print 'trig_time %s (fno %d) OK %d'%(repr(trig_time),trig_fno,count)

        mean_pretrig_z = mlab.mean( zm_time_of_interest.compressed())
        if not isinstance( mean_pretrig_z, float):
            print 'HMM, WARNING 2'
        else:
            if mean_pretrig_z <= 70.0:
                print 'Mean pretrigger altitude <= 70mm, discarding'
                continue
        
        if sum_xdiff >= 0:
            upwind = True
        else:
            upwind = False

        # find frame-to-frame distance in mm
        IFI_dist_mm = M.sqrt((xm[1:]-xm[:-1])**2 + (ym[1:]-ym[:-1])**2 + (zm[1:]-zm[:-1])**2)
        nonmaskedlist = list( IFI_dist_mm.compressed() )

        if max(nonmaskedlist) > 10:
            print 'WARNING: skipping because >10 mm found between adjacent frames (%d frames found)'%(len(xm),)
            continue
        
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
        
print 'good_count',good_count,'(includes Ns excluded by ultra_strict)'

del upwind # make sure we don't use this
del xs
del ys
del zs
del xvels
del yvels
del zvels
del headings
del mean_dist

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
            val_array = M.concatenate( val_list, axis=0 )
            # each column is a timepoint
            means = nx.zeros( (val_array.shape[1],), nx.Float )
            stds = nx.zeros( (val_array.shape[1],), nx.Float )
            minN = 1e30
            maxN = 0
            for j in range( val_array.shape[1] ):
                col = val_array[:,j]
                if isinstance(col,numpy.ma.MaskedArray):
                    col_compressed = val_array[:,j].compressed()
                else:
                    col_compressed = col
                N = len(col_compressed)
                if N < minN: minN = N
                if N > maxN: maxN = N
                if N==0:
                    print 'WARNING: mean not computed, N==0'
                    continue
                means[j] = mlab.mean( col_compressed )
                if N==1:
                    print 'WARNING: std not computed, N==1'
                    continue
                stds[j] = mlab.std( col_compressed )

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

