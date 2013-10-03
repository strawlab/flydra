import os, glob, time
from .auto_discover_ufmfs import get_h5_start_stop, get_uuid

DEFAULT_MOVIE_SUBDIR = '~/FLYDRA_MOVIES'

def find_movies(h5_fname,verbose=True,candidate_index=0):
    '''find movies (.ufmf or .fmf) which are in canonical location'''
    h5_start, h5_stop = get_h5_start_stop(h5_fname)
    uuid = get_uuid(h5_fname)
    if verbose:
        print 'h5_start, h5_stop',h5_start, h5_stop
    test_path = os.path.expanduser(DEFAULT_MOVIE_SUBDIR)

    if os.path.exists(os.path.join(test_path,uuid)):
        # new code path - movies saved in "<DEFAULT_MOVIE_SUBDIR>/<uuid>/<cam_id>/<time>.fmf"
        return find_movies_uuid(h5_fname,verbose=verbose)

    maybe_dirnames = glob.glob( os.path.join(test_path,'*') )

    candidates = []

    for maybe_dirname in maybe_dirnames:
        if not os.path.isdir(maybe_dirname):
            continue # not a directory
        basename = os.path.split(maybe_dirname)[-1]

        struct_time = time.strptime(basename,'%Y%m%d_%H%M%S')
        approx_start = time.mktime(struct_time)
        if verbose:
            print '  option: %s: %s'%(maybe_dirname, approx_start)

        if (h5_start <= approx_start) and (h5_stop >= approx_start):
            if verbose:
                print '    valid!'
            candidates.append( (maybe_dirname, basename) )

    if len(candidates) == 0:
        return []

    candidate = candidates[candidate_index]
    (dirname, basename) = candidate

    if len(candidates) > 1:
        print '%d candidate movies available, choosing %s/%s'%(len(candidates),
                                                               dirname, basename)

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

def find_movies_uuid(h5_fname,verbose=True):
    uuid = get_uuid(h5_fname)

    test_path = os.path.expanduser(DEFAULT_MOVIE_SUBDIR)

    candidates = []

    cam_ids = glob.glob( os.path.join(test_path,uuid,'*') )
    for cam_id in cam_ids:
        cam_dir = os.path.join( os.path.join(test_path,uuid, cam_id ) )

        mean_fmf = glob.glob( os.path.join( cam_dir, '*_mean.fmf' ))
        mean_ufmf = glob.glob( os.path.join( cam_dir, '*_mean.ufmf' ))
        mean_files = mean_fmf + mean_ufmf

        total_length = len(mean_files)
        if not total_length==1:
            raise NotImplementedError('need to figure out what time is best...')
        if len(mean_fmf)==1:
            mode='fmf'
        else:
            mode='ufmf'

        mean_fname = mean_files[0]
        basename = os.path.split(mean_fname)[-1]
        struct_time = time.strptime(basename,'%Y%m%d_%H%M%S_mean.'+mode)
        approx_start = time.mktime(struct_time)
        main_file = time.strftime('%Y%m%d_%H%M%S.'+mode, struct_time)
        candidates.append( os.path.join(cam_dir,main_file ))
    return candidates
