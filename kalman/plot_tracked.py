import tables
import numpy

class TrackedObject: pass

filename = 'DATA20060719_180955.tracked.h5'
results = tables.openFile(filename,mode='r')
idx = 0
tros = []
while 1:
    try:
        xhats_table = getattr(results.root,'kalman_%d'%idx)
    except tables.exceptions.NoSuchNodeError:
        break
    obs_table = getattr(results.root,'observations_%d'%idx)
    idx+=1

    tro = TrackedObject()
    tro.observations_frames = numpy.asarray(obs_table.read(flavor='numpy',field='frame'))
    if len(tro.observations_frames) > 10:
        tro.observations_data = numpy.hstack((numpy.asarray(obs_table.read(flavor='numpy',field='x'))[:,numpy.newaxis],
                                              numpy.asarray(obs_table.read(flavor='numpy',field='y'))[:,numpy.newaxis],
                                              numpy.asarray(obs_table.read(flavor='numpy',field='z'))[:,numpy.newaxis]))
        tro.frames = numpy.asarray(xhats_table.read(flavor='numpy',field='frame'))
        tro.xhats = numpy.hstack((numpy.asarray(xhats_table.read(flavor='numpy',field='x'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='y'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='z'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='xvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='yvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='zvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='xaccel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='yaccel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='zaccel'))[:,numpy.newaxis]))
        tros.append(tro)
    print 'loaded',idx
    if idx==300:
        break
    
if 1:
    import pylab

    colorlist = 'b','g','r','k','c','m'

    ss = 9
    nobs = 3
    pylab.figure()
    ax = None
    varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
    for i in range(ss):
        ax = pylab.subplot(ss,1,i+1,sharex=ax)
        var = varnames[i]
        for tro_idx,tro in enumerate(tros):
            color_idx = tro_idx%len(colorlist)
            this_color = colorlist[color_idx]

            if i==0: print 'tro',this_color,tro
            if i<nobs:
                if i==0: print tro.observations_frames[0],'-',tro.observations_frames[-1]
                ax.plot(tro.observations_frames,tro.observations_data[:,i],this_color+'+',label='observations of %s'%var)
            ax.plot(tro.frames,tro.xhats[:,i],this_color+'-',label='estimates of %s'%var)
        pylab.ylabel(var)       
    pylab.xlabel('frame')

if 0:

    pylab.figure()
    ax = None
    varnames = ['X','Y','Z','X vel','Y vel','Z vel','X accel','Y accel','Z accel']
    for i in range(ss):
        ax = pylab.subplot(ss,1,i+1,sharex=ax)
        var = varnames[i]
        for tro_idx,tro in enumerate(tracker.dead_tracked_objects):
            color_idx = tro_idx%len(colorlist)
            this_color = colorlist[color_idx]

            tro.Ps = numpy.asarray(tro.Ps)
            ax.plot(tro.frames,numpy.sqrt(tro.Ps[:,i,i]),this_color+'-',label='estimates of %s'%var)
        pylab.ylabel(var)       
    pylab.xlabel('frame')

pylab.show()
