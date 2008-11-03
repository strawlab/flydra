from optparse import OptionParser
import tables
import numpy as np
import glob, os, re, time
import flydra.analysis.result_utils as result_utils
import motmot.ufmf.ufmf

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

def find_ufmfs(filename,ufmf_dir=None):

    ufmf_template = 'small_%(date_time)s_%(cam_id)s.ufmf$'
    date_time_re = '([0-9]{8}_[0-9]{6})'
    ufmf_template_re = ufmf_template.replace('%(date_time)s',date_time_re)

    if ufmf_dir is None:
        ufmf_dir = os.path.split( os.path.abspath( filename ))[0]
    all_ufmfs = glob.glob(os.path.join(ufmf_dir,'*.ufmf'))
    all_ufmfs.sort()

    h5 = tables.openFile(filename,mode='r')
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
    cam_ids = cam_id2camns.keys()

    h5_start = np.inf
    h5_stop = -np.inf
    for row in h5.root.data2d_distorted:
        ts = row['timestamp']
        h5_start = min(ts,h5_start)
        h5_stop = max(ts,h5_stop)
    h5.close()

    possible_ufmfs = []

    for cam_id in cam_ids:
        # find likely ufmfs for each cam_id in .h5 file
        this_cam_re = ufmf_template_re.replace('%(cam_id)s',cam_id)
        prog = re.compile(this_cam_re)

        approx_starts = []
        this_cam_id_fnames = []
        for ufmf_filename in all_ufmfs:
            dir,node = os.path.split(ufmf_filename)
            match_object = prog.match(node)
            if match_object is None:
                continue
            date_time = match_object.group(1)
            struct_time = time.strptime(date_time,'%Y%m%d_%H%M%S')
            ufmf_approx_start = time.mktime(struct_time)
            #print node,time.asctime(struct_time),ufmf_approx_start

            approx_starts.append( ufmf_approx_start )
            this_cam_id_fnames.append( ufmf_filename )

        approx_stops = approx_starts[1:]
        approx_stops.append(np.inf)

        ufmf_approx_starts = np.array( approx_starts )
        ufmf_approx_stops = np.array( approx_stops )

        eps = 1.0 # 1 second slop
        bad_cond = (ufmf_approx_stops-eps) < h5_start
        bad_cond |= (ufmf_approx_starts+eps) > h5_stop
        good_cond = ~bad_cond
        good_idx = np.nonzero(good_cond)[0]
        ## for idx in good_idx:
        ##     print this_cam_id_fnames[idx]
        possible_ufmfs.extend( [ this_cam_id_fnames[idx] for idx in good_idx] )

    results = []
    for ufmf_filename in possible_ufmfs:
        ufmf = motmot.ufmf.ufmf.FlyMovieEmulator(ufmf_filename)
        ufmf_timestamps = ufmf.get_all_timestamps()
        ufmf_start = ufmf_timestamps[0]
        ufmf_stop = ufmf_timestamps[-1]

        if h5_start <= ufmf_start <= h5_stop:
            results.append( ufmf_filename )
        elif h5_start <= ufmf_stop <= h5_stop:
            results.append( ufmf_filename )
        elif ufmf_start <= h5_start <= ufmf_stop:
            results.append( ufmf_filename )

        ufmf.close()

    return results

def main():
    usage = """%prog [options] FILENAME"""
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    if len(args)!=1:
        raise ValueError('a (single) filename must be given')
    filename = args[0]
    ufmfs = find_ufmfs(filename)
    for ufmf in ufmfs:
        print ufmf
