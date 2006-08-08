import tables
import numpy

class TrackedObject: pass

filename = 'DATA20060719_180955.tracked.resaved-tracked.h5'
results = tables.openFile(filename,mode='r')
xhats_table = getattr(results.root,'kalman_estimates')
obs_table = getattr(results.root,'observations')
flavor = 'numarray' # numpy not working yet

obj_ids = xhats_table.read(field='obj_id',flavor=flavor)

idx = 1
print 'XXX WARNING: skipping first track for temporary reasons!'
tros = []
while 1:
    xhat_coords = xhats_table.getWhereList(xhats_table.cols.obj_id==idx,flavor=flavor)
    obs_coords = obs_table.getWhereList(obs_table.cols.obj_id==idx,flavor=flavor)

    if len(obs_coords)==0:
        break
    
    idx+=1

    obs_recarray = obs_table.readCoordinates(obs_coords,flavor=flavor)
    
    tro = TrackedObject()
    tro.observations_frames = obs_recarray.field('frame')
    if len(tro.observations_frames) > 10:
        tro.observations_data = numpy.hstack((obs_recarray.field('x')[:,numpy.newaxis],
                                              obs_recarray.field('y')[:,numpy.newaxis],
                                              obs_recarray.field('z')[:,numpy.newaxis]))
        
        xhat_recarray = xhats_table.readCoordinates(xhat_coords,flavor=flavor)
        tro.frames = xhat_recarray.field('frame')
        tro.xhats = numpy.hstack((xhat_recarray.field('x')[:,numpy.newaxis],
                                  xhat_recarray.field('y')[:,numpy.newaxis],
                                  xhat_recarray.field('z')[:,numpy.newaxis],
                                  xhat_recarray.field('xvel')[:,numpy.newaxis],
                                  xhat_recarray.field('yvel')[:,numpy.newaxis],
                                  xhat_recarray.field('zvel')[:,numpy.newaxis],
                                  xhat_recarray.field('xaccel')[:,numpy.newaxis],
                                  xhat_recarray.field('yaccel')[:,numpy.newaxis],
                                  xhat_recarray.field('zaccel')[:,numpy.newaxis]))
                                  
        tros.append(tro)
    print 'loaded',idx
##    if idx==300:
##        break
    
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
