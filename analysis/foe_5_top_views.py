from __future__ import division
import result_browser
import pylab
import numpy as nx
import math, glob, time
from math import pi

R2D = 180/pi
D2R = pi/180

# find segments to use
analysis_file = open('strict_data.txt','r')
execfile('triggered_loaddata.py')
tf_hz = condition_float

# top views
for horsetail in [True,False]:
    for slowest_xvel, fastest_xvel, frame_title in [
        (-.2,.2,'slow (up-tunnel, hovering, down-tunnel)'),
        
        (.2,1000,'fast up-tunnel'),
        (.05,.2,'slow up-tunnel'),
        (-.05,.05,'hovering'),
        (-.2,-0.05,'slow down-tunnel'),
        (-1000,-.2,'fast down-tunnel'),
        ]:
        pylab.figure()#figsize=(8,8/(2/3)))
        ax = None
        for tf_hz, col, axtitle in [(0.0,0,'static'),
                                    (5.0,1,'FOE'),
                                    ]:
            ax=pylab.subplot(2,1,col+1,sharex=ax,sharey=ax)
            if col==0:
                pylab.title(frame_title)
            if tf_hz not in early_xvel:
                continue
            for i in range(len(early_xvel[tf_hz])):
                this_early_xvel = early_xvel[tf_hz][i]
                if not (slowest_xvel<=this_early_xvel<fastest_xvel):
                    continue
                X = xs_pre[tf_hz][i]
                Y = ys_pre[tf_hz][i]
                if horsetail:
                    X0 = X[-1]
                    Y0 = Y[-1]
                else:
                    X0 = 0.0
                    Y0 = 0.0
                pylab.plot(X-X0,Y-Y0,'k-')

                X = xs_post[tf_hz][i]
                Y = ys_post[tf_hz][i]
                if 0:
                    if tf_hz == 0.0:
                        fmt = 'k-'
                    else:
                        fmt = 'r-'
                else:
                    fmt = 'r-'
                pylab.plot(X-X0,Y-Y0,fmt)
            if not horsetail:
                ax.set_xlim([500,850])
                ax.set_ylim([30,330])
            else:
                ax.set_xlim([-75,75])
                ax.set_ylim([-50,50])
            pylab.text(0.01,0.98,axtitle,
                       transform = ax.transAxes,
                       horizontalalignment='left',
                       verticalalignment='top',
                       )
            
if 0:
    pylab.figure()#figsize=(8,8/(2/3)))
    for tf_hz, col, title in [(0.0,0,'no FOE'),
                              (5.0,1,'FOE'),
                              ]:
        downwind_cond = early_xvel[tf_hz] < -0.2 # m/sec
        slow_cond = (-0.2 <= early_xvel[tf_hz]) & (early_xvel[tf_hz] < 0.2) # m/sec
        upwind_cond = (0.2 <= early_xvel[tf_hz])

        downwind_headling_late = heading_late[tf_hz][downwind_cond]
        slow_headling_late = heading_late[tf_hz][slow_cond]
        upwind_headling_late = heading_late[tf_hz][upwind_cond]

        ax = pylab.subplot(3,2,col+1,frameon=False)#,polar=True)
        ax.set_title(title)
        circstats.raw_data_plot(ax,downwind_headling_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(downwind_headling_late)['mu']
        circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)

        ax = pylab.subplot(3,2,col+3,frameon=False)#,polar=True)
        circstats.raw_data_plot(ax,slow_headling_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(slow_headling_late)['mu']
        circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)

        ax = pylab.subplot(3,2,col+5,frameon=False)#,polar=True)
        circstats.raw_data_plot(ax,upwind_headling_late,marker='.',linestyle='None')
        mu = circstats.mle_vonmises(upwind_headling_late)['mu']
        circstats.raw_data_plot(ax,[mu],marker='*',linestyle='None',r=0.8)
    
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
