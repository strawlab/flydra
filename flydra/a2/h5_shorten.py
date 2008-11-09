import tables
from optparse import OptionParser
import warnings
import numpy as np

def get_start_stop(src_h5,name,start,stop):
    print '  finding start and stop row of %s'%name
    input_node = src_h5.root.data2d_distorted
    frames = input_node[:]['frame']
    if start is not None:
        valid_cond = frames >= start
    else:
        valid_cond = np.ones( frames.shape, dtype=np.bool )
    if stop is not None:
        valid_cond &= frames <= stop
    valid_idx = np.nonzero(valid_cond)[0]

    # check that start and stop accurately characterize data
    di = valid_idx[1:]-valid_idx[:-1]
    if np.max(di) != 1 or np.min(di) != 1:
        raise ValueError('data are not contiguous')

    return valid_idx[0], valid_idx[-1]

def copy_selective(src_h5,input_node,output_group,options):
    if input_node.name in ['data2d_distorted',
                           'kalman_estimates',
                           'kalman_observations',
                           ]:

        startrow,stoprow=get_start_stop(src_h5,input_node.name,
                                        options.start,options.stop)
        print '  copying data'
        input_node._f_copy(output_group,
                           start=startrow,
                           stop=stoprow)
    elif input_node.name=='kalman_observations_2d_idxs':
        # kalman_observations and kalman_observations_2d_idxs must be
        # reduced with knowledge of each other. Column obs_2d_idx of
        # kalman_observations must point to row number of
        # corresponding kalman_observations_2d_idxs. Therefore, if
        # rows are dropped from start of kalman_observations_2d_idxs,
        # kalman_observations must be updated.

        if options.start is not None:
            warnings.warn('filtering initial data in table '
                          'kalman_observations_2d_idxs not implemented')

        startrow,stoprow=get_start_stop(src_h5,'kalman_observations',
                                        options.start,options.stop)
        my_stoprow = int(src_h5.root.kalman_observations[stoprow]['obs_2d_idx'])
        input_node._f_copy(output_group,
                           start=0,
                           stop=my_stoprow+1)

    else:
        warnings.warn('no filtering implemented, copying entire '
                      'table %s'%input_node.name)
        input_node._f_copy(output_group,recursive=True)

def doit(input_filename, output_filename, options):
    h5 = tables.openFile( input_filename,mode='r')
    output_h5 = tables.openFile( output_filename,mode='w')
    for input_node in h5.root._f_iterNodes():
        print 'copying',input_node
        if (hasattr(input_node,'name') and
            input_node.name in ['data2d_distorted',
                                'kalman_estimates',
                                'kalman_observations',
                                'kalman_observations_2d_idxs']):
            copy_selective(h5,input_node,output_h5.root,options)
        else:
            # copy everything from source to dest
            input_node._f_copy(output_h5.root,recursive=True)
    h5.close()
    output_h5.close()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)
    parser.add_option("--input", type='string')
    parser.add_option("--output", type='string')
    parser.add_option("--start", type='int', default=None)
    parser.add_option("--stop", type='int', default=None)

    (options, args) = parser.parse_args()

    input = options.input
    output = options.output
    doit(input,output,options)

if __name__=='__main__':
    main()

