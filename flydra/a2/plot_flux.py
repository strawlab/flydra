from __future__ import division
import pickle
import numpy
import pylab
import sys

def plot_top_view(ax,datadict,**kws):
    plot_view(ax,datadict,'top',**kws)
    
def plot_side_view(ax,datadict,**kws):
    plot_view(ax,datadict,'side',**kws)
    
def plot_view(ax,datadict,view,pdf=False,
              vmin=None,
              vmax=None,
              ):
    to_counts = datadict['to_counts']
    x_boundaries = datadict['x_boundaries']
    y_boundaries = datadict['y_boundaries']
    z_boundaries = datadict['z_boundaries']

    if view == 'top':
        narrow_z = (z_boundaries >= 0.05) & (z_boundaries <= 0.25)
        top_view = numpy.sum(to_counts[1:-1,1:-1,narrow_z],axis=2)
        view = top_view
        xlim = x_boundaries[1],x_boundaries[-2]
        ylim = y_boundaries[1],y_boundaries[-2]
        xlabel = 'x'
        ylabel = 'y'
    elif view == 'side':
        narrow_y = (y_boundaries >= 0.1) & (y_boundaries <= 0.3)
        side_view = numpy.sum(to_counts[1:-1,narrow_y,1:-1],axis=1)
        view = side_view
        xlim = x_boundaries[1],x_boundaries[-2]
        ylim = z_boundaries[1],z_boundaries[-2]
        xlabel = 'x'
        ylabel = 'z'

    if pdf:
        view_sum = numpy.sum(numpy.sum(view))
        view = view/view_sum
        max_prob = view.max()
        #print 'max_prob',max_prob
    
    pylab.imshow(view.swapaxes(0,1), # images are plotted y then x
                 interpolation='nearest',
                 origin='lower',
                 extent=[xlim[0],xlim[1],ylim[0],ylim[1]],
                 vmin=vmin,
                 vmax=vmax,
                 aspect='equal')
    #pylab.xlabel(xlabel+' (m)')
    #pylab.ylabel(ylabel+' (m)')

    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim((.1,.8))
    
def main():
    filename = sys.argv[1]
    fname = filename+'.pkl'
    fd = open(fname,mode='rb')
    datadict = pickle.load(fd)
    fd.close()

    if 0:
        pylab.figure()
        pylab.title( 'top view '+filename)
        ax = pylab.subplot(1,1,1)
        plot_top_view(ax,datadict)

        pylab.figure()
        pylab.title( 'side view '+filename)
        ax = pylab.subplot(1,1,1)
        plot_side_view(ax,datadict)
    else:
        pylab.figure()
        pylab.title( filename )
        ax = pylab.subplot(2,1,1)
        plot_top_view(ax,datadict)
        ax = pylab.subplot(2,1,2,sharex=ax)
        plot_side_view(ax,datadict)
        
    pylab.show()
    

if __name__=='__main__':
    main()
