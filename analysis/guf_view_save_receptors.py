import numpy
import pylab
import scipy.io
import fsee.plot_utils as plot_utils

fname = './guf_data.mat'
#fname = 'guf_data'
if 1:
    print 'reading',fname
    results = scipy.io.loadmat(fname,appendmat=0)

    keys = results.keys()
    for key in keys:
        if len(results[key].shape) == 1:
            results[key] = results[key][numpy.NewAxis,:]

    wrapped_basemap_instance = plot_utils.BasemapInstanceWrapper(proj='moll')
    
    n_steps = results['EMDs'].shape[0]
    for i in range(0,n_steps,10):
        new_filename = fname + '_receptors_%05d'%((i+1),) + '.ps'
        #new_filename = fname + '_receptors_%05d'%((i+1),) + '.svg'
        print 'saving',new_filename
        R=results['R'][i]
        G=results['G'][i]
        B=results['B'][i]
        emds=results['EMDs'][i]

        plot_utils.plot_receptor_and_emd_fig( R,G,B,emds, scale=1e-4, save_fname=new_filename )
        #pylab.close('all')
