from __future__ import division
import result_browser
import pylab
import matplotlib.axes
import numpy as nx
import math, glob, time, sys
import circstats
from math import pi
import mplsizer

try:
    import rpy1
    have_rpy = True
except ImportError:
    have_rpy = False

if have_rpy:
    R = rpy.r
    R.library("circular")

R2D = 180/pi
D2R = pi/180
# find segments to use
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'strict_data.txt'
print 'opening',fname
analysis_file = open(fname,'r')

execfile('triggered_loaddata.py')
#tf_hz = condition_float

h5filenameslist = []
for key in h5filenames.keys():
    for h5f in h5filenames[key]:
        if h5f not in h5filenameslist:
            h5filenameslist.append( h5f )
print h5filenameslist
        
if 1:
    col_info = [(0.0,0,'no FOE'),
                (5.0,1,'FOE'),
                ]
    fig = pylab.figure()
    frame = mplsizer.MplSizerFrame( fig )
    sizer = mplsizer.MplGridSizer(cols=len(col_info),append_horiz=False)
    frame.SetSizer(sizer,expand=1,left=1,bottom=1,top=1,border=0.5)
    ax = None
    for tf_hz, col, title in col_info:
        downwind_cond = early_xvel[tf_hz] < -0.2 # m/sec
        slow_cond = (-0.2 <= early_xvel[tf_hz]) & (early_xvel[tf_hz] < 0.2) # m/sec
        upwind_cond = (0.2 <= early_xvel[tf_hz])

        downwind_heading_late = heading_late[tf_hz][downwind_cond]
        slow_heading_late = heading_late[tf_hz][slow_cond]
        upwind_heading_late = heading_late[tf_hz][upwind_cond]


        ax = fig.add_axes([0,0,1,1],label='%s, %f, %d'%(title, tf_hz, col),
                          sharex=ax,sharey=ax,frameon=True)
        ax.axesPatch.set_edgecolor('white')
        sizer.Add(ax,expand=0,border=.2,all=1,minsize=(2,2))
        circstats.raw_data_plot(ax,downwind_heading_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(downwind_heading_late)['mu']
        #circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)
        ax.text(0.01,0.98, 'n=%d'%len(downwind_heading_late),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                )
        if col==0:
            ax.set_ylabel('fast downwind',fontsize=14)
        ax.set_title(title,fontsize=14)

        ax = fig.add_axes([0,0,1,1],label='%s, %f, %d'%(title, tf_hz, col),
                          sharex=ax,sharey=ax,frameon=True)
        ax.axesPatch.set_edgecolor('white')
        sizer.Add(ax,expand=0,border=.2,all=1,minsize=(2,2))
        circstats.raw_data_plot(ax,slow_heading_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(slow_heading_late)['mu']
        #circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)
        ax.text(0.01,0.98, 'n=%d'%len(slow_heading_late),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                )
        if col==0:
            ax.set_ylabel('slow flight',fontsize=14)


        ax = fig.add_axes([0,0,1,1],label='%s, %f, %d'%(title, tf_hz, col),
                          sharex=ax,sharey=ax,frameon=True)
        ax.axesPatch.set_edgecolor('white')
        sizer.Add(ax,expand=0,border=.2,all=1,minsize=(2,2))
        circstats.raw_data_plot(ax,upwind_heading_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(upwind_heading_late)['mu']
        #circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)
        ax.text(0.01,0.98, 'n=%d'%len(upwind_heading_late),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                )
        if col==0:
            ax.set_ylabel('fast upwind',fontsize=14)
    frame.Layout()
    pylab.figtext(0.99,0.01,'\n'.join(h5filenameslist),
                  horizontalalignment='right',
                  verticalalignment='bottom')
    
if 0:
    class ToR:
        def __init__(self):
            self.toR_fds = {}
        def __call__(self,name,arr,rfile,circular=None):
            # remember if we did manipulations to R data file
            if rfile not in self.toR_fds:
                self.toR_fds[rfile] = {}

            valstr = 'c(%s)'%(', '.join(map(repr,arr)),)
            if circular is not None:
                if not self.toR_fds[rfile].get('loaded circular',False):
                    print >>rfile,'library("circular")'
                    self.toR_fds[rfile]['loaded circular'] = True
                if circular.lower().startswith('deg'):
                    valstr = 'circular(%s,units="degrees")'%valstr
                elif circular.lower().startswith('rad'):
                    valstr = 'circular(%s,units="radians")'%valstr
                else:
                    raise ValueError("unknown circular format")
            print >>rfile,'%s <- %s'%(name,valstr)
            
    toR = ToR()
    tf_hz = 0.0
    name = {0.0:'still',
            5.0:'fast',
            }
    rfile = open('data.r','w')
    for tf_hz in [0.0,5.0]:
        toR('heading_early_%s'%name[tf_hz],heading_early[tf_hz],rfile,circular='rad')
        toR('heading_late_%s'%name[tf_hz],heading_late[tf_hz],rfile,circular='rad')
        toR('xvel_early_%s'%name[tf_hz],early_xvel[tf_hz],rfile)
    rfile.close()

if 0:
    pylab.figure()
    for tf_hz, fmt, fmt_fit in [(0.0,'r.','r-'),
                                (5.0,'b.','b-'),
                                ]:
        pylab.subplot(4,2,1)
        pylab.plot( early_xvel[tf_hz], turn_angle[tf_hz], fmt )
        pylab.xlabel('initial x vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('turn angle (deg)')
        pylab.axhline(0.0,color='k')
        pylab.axhline(360.0,color='k')

        pylab.subplot(4,2,2)
        pylab.plot( heading_early[tf_hz], heading_late[tf_hz], fmt )
        pylab.xlabel('initial heading (rad)')
        pylab.axvline(-180.0,color='k')
        pylab.axvline(0.0,color='k')
        pylab.axvline(180.0,color='k')
        pylab.ylabel('late heading (rad)')
        pylab.axhline(-180.0,color='k')
        pylab.axhline(0.0,color='k')
        pylab.axhline(180.0,color='k')

        pylab.subplot(4,2,3)
        pylab.plot( early_xvel[tf_hz], heading_early[tf_hz], fmt )
        pylab.xlabel('initial x vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('initial heading (rad)')
        pylab.axhline(-180.0,color='k')
        pylab.axhline(0.0,color='k')
        pylab.axhline(180.0,color='k')

        pylab.subplot(4,2,4)
        pylab.plot( early_xvel[tf_hz], heading_late[tf_hz], fmt )
        pylab.xlabel('initial x vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('late heading (rad)')
        pylab.axhline(-180.0,color='k')
        pylab.axhline(0.0,color='k')
        pylab.axhline(180.0,color='k')
        if have_rpy:
            # do regression
            regr = R.lm_circular( nx.array(heading_late[tf_hz])*D2R,
                                  early_xvel[tf_hz], [1.0],
                                  type="c-l", verbose=True )
            mu = regr['mu']
            beta = regr['coefficients']
            
            # fit
            x = pylab.linspace(-1,.6,100)
            yfit = mu + 2*nx.arctan( x*beta )
            pylab.plot( x,yfit*R2D,fmt_fit)

        pylab.subplot(4,2,5)
        pylab.plot( early_xvel[tf_hz], early_yvel[tf_hz], fmt )
        pylab.xlabel('initial x vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('initial y vel (m/sec)')
        pylab.axhline(0.0,color='k')

        pylab.subplot(4,2,6)
        pylab.plot( early_xvel[tf_hz], late_yvel[tf_hz], fmt )
        pylab.xlabel('initial x vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('late y vel (m/sec)')
        pylab.axhline(0.0,color='k')

        pylab.subplot(4,2,7)
        pylab.plot( early_yvel[tf_hz], late_yvel[tf_hz], fmt )
        pylab.xlabel('initial y vel (m/sec)')
        pylab.axvline(0.0,color='k')
        pylab.ylabel('late y vel (m/sec)')
        pylab.axhline(0.0,color='k')

        pylab.subplot(4,2,8)
        pylab.plot( [0],[0],fmt,label='TF %.0f Hz'%tf_hz)
    pylab.legend()

if 0:
    pylab.figure()
    for tf_hz, fmt, fmt2 in [(0.0,'r-','r.'),
                             (5.0,'b-','b.'),
                             ]:
        pylab.subplot(1,1,1)
        for ex, lx, ey, ly in zip(early_xvel[tf_hz],late_xvel[tf_hz],
                                  early_yvel[tf_hz],late_yvel[tf_hz]):
            pylab.plot( [ex,lx],[ey,ly],fmt)
            pylab.plot( [lx],[ly],fmt2)
pylab.show()
