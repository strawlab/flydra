import os, glob, time
from .auto_discover_ufmfs import get_h5_start_stop

DEFAULT_MOVIE_SUBDIR = '~/FLYDRA_MOVIES'

def find_movies(h5_fname,verbose=True):
    '''find movies (.ufmf or .fmf) which are in canonical location'''
    h5_start, h5_stop = get_h5_start_stop(h5_fname)
    if verbose:
        print 'h5_start, h5_stop',h5_start, h5_stop
    test_path = os.path.expanduser(DEFAULT_MOVIE_SUBDIR)
    maybe_dirnames = glob.glob( os.path.join(test_path,'*') )
    candidate = None

    for maybe_dirname in maybe_dirnames:
        if not os.path.isdir(maybe_dirname):
            continue # not a directory
        basename = os.path.split(maybe_dirname)[-1]

        struct_time = time.strptime(basename,'%Y%m%d_%H%M%S')
        approx_start = time.mktime(struct_time)
        if verbose:
            print '  option: %s: %s'%(maybe_dirname, approx_start)

        if (h5_start <= approx_start) and (h5_stop >= approx_start):
            assert candidate is None
            candidate = (maybe_dirname, basename)

    if candidate is None:
        return []

    (dirname, basename) = candidate
    cam_ids = os.listdir(dirname)
    results = []
    mode = None
    for cam_id in cam_ids:
        camdir = os.path.join(dirname,cam_id)
        camfiles = os.listdir(camdir)
        if mode=='fmf' or mode=='ufmf':
            fname = basename + '.' + mode
            assert fname in camfiles
            results.append( os.path.join(camdir,fname))
        else:
            fname = basename + '.fmf'
            if fname in camfiles:
                mode='fmf'
                results.append( os.path.join(camdir,fname))
            else:
                fname = basename + '.ufmf'
                if fname in camfiles:
                    mode='ufmf'
                    results.append( os.path.join(camdir,fname))
    if verbose:
        for r in results:
            print r
    return results
