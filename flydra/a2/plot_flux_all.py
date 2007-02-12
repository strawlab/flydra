from conditions import files, condition_names
import flux, plot_flux
import pylab
import pickle, os
import numpy

if 1:
    n_cols = len(files)
    n_rows = 2
    pylab.figure( figsize=(n_cols*3.0,n_rows*2.0), dpi=75 )
    ax = None
    top_view_ax = None
    side_view_ax = None


                       
        
    for col_num,condition_name in enumerate(condition_names):
    #for col_num,condition_name in enumerate(files):
        filenames = files[condition_name]

        to_counts = None
        
        data_shape = None
        x_boundaries = None
        y_boundaries = None
        z_boundaries = None
        for filename in filenames:
            fname = filename+'.pkl'
            
            if not os.path.exists(fname):
                print 'generating .pkl file',fname
                flux.doit(filename)
                
            #print 'loading',fname
            fd = open(fname,mode='rb')
            datadict = pickle.load(fd)
            fd.close()

            if x_boundaries is None:
                x_boundaries = datadict['x_boundaries']
            assert numpy.allclose(x_boundaries,datadict['x_boundaries'])
            if y_boundaries is None:
                y_boundaries = datadict['y_boundaries']
            assert numpy.allclose(y_boundaries,datadict['y_boundaries'])
            if z_boundaries is None:
                z_boundaries = datadict['z_boundaries']
            assert numpy.allclose(z_boundaries,datadict['z_boundaries'])

            if data_shape is None:
                to_counts = datadict['to_counts']
                data_shape = to_counts.shape
            else:
                to_counts += datadict['to_counts']
            assert data_shape == datadict['to_counts'].shape

        datadict = {'to_counts':to_counts,
                    'x_boundaries':x_boundaries,
                    'y_boundaries':y_boundaries,
                    'z_boundaries':z_boundaries,
                    }

        row_num = 0
##        print 'row_num,col_num',row_num,col_num+1
##        print 'TOP  row_num*n_rows+col_num+1',row_num*n_rows+col_num+1
        ax = pylab.subplot(n_rows,n_cols,row_num*n_cols+col_num+1,sharex=ax,
                           sharey=top_view_ax,
                           frameon=False)
        if top_view_ax is None:
            top_view_ax = ax
        plot_flux.plot_top_view(ax,datadict,pdf=True,
                                vmin=0.0,
                                vmax=0.00177147918512,
                                )
        pylab.title(condition_name)
        if col_num==0:
            pylab.ylabel('top view')
        
        row_num = 1
##        print 'row_num,col_num',row_num,col_num+1
##        print 'SIDE row_num*n_rows+col_num+1',row_num*n_rows+col_num+1
##        print
        ax = pylab.subplot(n_rows,n_cols,row_num*n_cols+col_num+1,sharex=ax,
                           sharey=side_view_ax,
                           frameon=False)
        if side_view_ax is None:
            side_view_ax = ax
        plot_flux.plot_side_view(ax,datadict,pdf=True,
                                 vmin=0.0,
                                 vmax=0.00254583364532)
        if col_num==0:
            pylab.ylabel('side view')

        if row_num == 1 and col_num == 0:
            yval = -.05
            pylab.axhline( y = yval, xmin=0.2, xmax=0.3, color='k', lw=2 )
            pylab.text(.25,yval, '10 cm',
                       horizontalalignment='center',
                       verticalalignment='top',
                       transform = ax.transAxes)


    pylab.show()
        
        
