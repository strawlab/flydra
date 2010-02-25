import json
from optparse import OptionParser
from couchdb.client import Server
import os, stat, re
import flydra.analysis.result_utils as result_utils
import flydra.reconstruct
import pprint
import numpy as np
import motmot.ufmf.ufmfstats as ufmfstats

from defaults import default_preferences

def fix_stat_dict(d):
    for k in ['data_start_time_float','data_stop_time_float']:
        newkey = k[:-6]
        if k in d:
            d[newkey] = result_utils.timestamp2string(d[k])
            del d[k]
        else:
            d[newkey] = None
    for kk,v in d['frame_info'].iteritems():
        d2 = v
        for k in ['start_time_float','stop_time_float']:
            newkey = k[:-6]
            d2[newkey] = result_utils.timestamp2string( d2[k] )
            del d2[k]
        if kk=='raw':
            # copy start/stop times from raw frames (even if no data)
            d['start_time'] = d2['start_time']
            d['stop_time'] = d2['stop_time']

def upload_ufmf(filenames,config):
    if config['couch_user'] is not None or config['couch_password'] is not None:
        credentials = '%s:%s@'%(config['couch_user'],config['couch_password'])
    else:
        credentials = ''
    server = Server(credentials+config['couch_url'])
    db = server[config['couch_db']]
    docs = []

    ufmf_fname_re = re.compile(r'^small_(\d+)_(\d+)_(\w+)\.ufmf$')
    
    for filename in filenames:
        doc = {}
        dataset_short = config['dataset']
        filename_short = os.path.split(filename)[-1]

        match = ufmf_fname_re.search(filename_short)
        if match is None:
            print >> sys.stderr, ('WARNING: could not parse cam_id from '
                                  'filename %s, skipping.'%filename_short)
            continue
        ufmf_date, ufmf_time, cam_id = match.groups()
        stats = ufmfstats.collect_stats(filename)
        fix_stat_dict(stats)

        doc['_id'] = 'ufmf:%s:%s'%(dataset_short,filename_short)
        doc['cam_id'] = cam_id
        doc['dataset'] = 'dataset:'+dataset_short
        doc['filename'] = filename_short
        doc['filesize'] = os.stat( filename )[stat.ST_SIZE]

        doc['type'] = 'ufmf'

        doc.update(stats)

        docs.append(doc)
    results = db.update(docs)
    counts = [r[0] for r in results]
    count = np.sum(counts)
    print 'uploaded %d of %d files to database "%s", dataset "%s"'%(
        count,len(docs),config['couch_db'],dataset_short)

def main():
    # load prefs
    config = default_preferences.copy()
    pref_fname = os.path.expanduser('~/.flydra/uploaderrc')
    if os.path.exists(pref_fname):
        prefs = json.loads( open(pref_fname,mode='r').read() )
        config.update(prefs)

    usage = '%prog FILE [options]'
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filenames = args

    upload_ufmf( filenames, config )

if __name__=='__main__':
    main()
