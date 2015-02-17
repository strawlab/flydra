#!/usr/bin/env python
import h5py
import sys

def remove_field_name(a, name):
    # See http://stackoverflow.com/a/15577562/1633026
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b

def revert_schema12_to_schema11(fname):
    expected_first = '{"schema": "http://strawlab.org/schemas/flydra/1.2"}'
    new_first      = '{"schema": "http://strawlab.org/schemas/flydra/1.1"}'
    lef = len(expected_first)
    assert len(new_first)==lef

    with open(fname,mode='rb') as fd:
        first = fd.read(lef)
        print 'found %r'%(first,)
        if first!=expected_first:
            print 'schema not 1.2'
            return
        print 'schema 1.2 found, reverting'

    # now we know we have a schema 1.2 file
    with h5py.File(fname,'r+') as f:
        trajs = f['trajectories'][:]
        print trajs.dtype
        trajs = remove_field_name(trajs, 'covariance_x')
        trajs = remove_field_name(trajs, 'covariance_y')
        trajs = remove_field_name(trajs, 'covariance_z')
        print trajs.dtype

        del f['trajectories']
        #f['trajectories'] = trajs
        f.create_dataset( 'trajectories', data=trajs,
                          compression='gzip',
                          compression_opts=9)

    with open(fname,mode='r+b') as fd:
        fd.seek(0)
        fd.write(new_first)

if __name__=='__main__':
    fname = sys.argv[1]
    revert_schema12_to_schema11(fname)
