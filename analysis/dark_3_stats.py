import pylab
import result_browser
import numpy as nx
import PQmath
import math, glob, time, sys, os
import mplsizer

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

view = 'top'
conditions=['light',
            'dark',
            'blinking']
dirs = ['upwind','downwind']
fig = pylab.figure(figsize=(10,8))

h5files = {}
rough_timestamps = {}

frame = mplsizer.MplSizerFrame( fig )
sizer = mplsizer.MplGridSizer(cols=len(conditions),append_horiz=False)
frame.SetSizer(sizer,expand=1,left=1,bottom=1,top=1,border=0.5)

ax = None
for col,condition in enumerate(conditions):
    for row,dir in enumerate(dirs):
        count = 0
        max_n_frames = 0
        ax = fig.add_axes([0,0,1,1],label='row %d col %d'%(row,col),
                          sharex=ax,sharey=ax)
        sizer.Add(ax,expand=1,border=.2,all=1)
        if row==0:
            ax.set_title(condition)
        else:
            ax.set_xlabel('X position (mm)')
        if col==0:
            if view == 'side':
                ax.set_ylabel('height (mm)')
            elif view == 'top':
                ax.set_ylabel('lateral position (mm)')
        for line_no,line in enumerate(f_segments):
            upwind_orig, fstart, trig_fno, fend, h5filename, condition_float = line

            if upwind_orig == 'False':
                upwind = False
            elif upwind_orig == 'True':
                upwind = True
            else:
                raise ValueError('hmm')

            if dir=='upwind' and not upwind:
                continue

            if dir=='downwind' and upwind:
                continue

            fstart = int(fstart)
            trig_fno = int(trig_fno)
            fend = int(fend)
            n_frames = fend-fstart
            max_n_frames = max(n_frames,max_n_frames)

            condition_float = float(condition_float)

            if condition == 'light':
                if condition_float!=1.0:
                    continue
            elif condition == 'dark':
                if condition_float!=0.0:
                    continue
            elif condition == 'blinking':
                if condition_float!=0.5:
                    continue

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

            if (rough_timestamp < datetime.datetime(2006, 2, 24, 18, 0, 0,
                                                    tzinfo=pacific)):
                print 'old scheme'
                raise NotImplementedError('')

            data3d = results.root.data3d_best
            ts=[]
            xs=[]
            ys=[]
            zs=[]
            fnos=[]
            for row in data3d.where( fstart <= data3d.cols.frame <= fend ):
                ts.append(row['timestamp'])
                xs.append(row['x'])
                ys.append(row['y'])
                zs.append(row['z'])
                fnos.append(row['frame'])
            ts=nx.array(ts)
            xs=nx.array(xs)
            ys=nx.array(ys)
            zs=nx.array(zs)
            fnos=nx.array(fnos)

            pre_trig_cond = ~(fnos > trig_fno)
            post_trig_cond = fnos > trig_fno
            trig_idx = nx.nonzero(fnos==trig_fno)[0]
            x0 = xs[trig_idx]
            y0 = ys[trig_idx]
            z0 = zs[trig_idx]

            if view=='side':
                #fmt = 'k-'
                #ax.plot( xs[post_trig_cond]-x0, zs[post_trig_cond]-z0, fmt )
                fmt = 'r-'
                ax.plot( xs[post_trig_cond]-x0, zs[post_trig_cond]-z0, fmt )
            elif view=='top':
                #fmt = 'k-'
                #ax.plot( xs[post_trig_cond]-x0, ys[post_trig_cond]-y0, fmt )
                fmt = 'r-'
                ax.plot( xs[post_trig_cond]-x0, ys[post_trig_cond]-y0, fmt )
            
            count += 1

        ax.text(0.01,0.98,'%s (n=%d)'%(dir,count),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top')
        ax.set_xlim((-110,110))
        ax.set_ylim((-155,155))
frame.Layout()
pylab.show()
#print 'examining %d traces'%count
