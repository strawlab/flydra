import json
import tables
from optparse import OptionParser
from couchdb.client import Server
import os, stat
import flydra.analysis.result_utils as result_utils
import flydra.reconstruct
import pprint
import numpy as np

default_preferences = {'dataset':'default dataset',
                       'couch_url':'http://localhost:5984/',
                       'couch_user':None,
                       'couch_password':None,
                       'couch_db':'flydra',
                       }

def doit(filenames,config):
    if config['couch_user'] is not None or config['couch_password'] is not None:
        credentials = '%s:%s@'%(config['couch_user'],config['couch_password'])
    else:
        credentials = ''
    server = Server(credentials+config['couch_url'])
    db = server[config['couch_db']]
    docs = []
    for filename in filenames:
        doc = {}
        dataset_short = config['dataset']
        filename_short = os.path.split(filename)[-1]
        h5 = tables.openFile(filename,mode='r')
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        try:
            reconstructor = flydra.reconstruct.Reconstructor( h5 )
            has_calib = True
        except tables.exceptions.NoSuchNodeError:
            has_calib = False

        frames = h5.root.data2d_distorted.read(field='frame')
        assert len(frames)
        start_frame = long(np.min(frames))
        stop_frame = long(np.max(frames))

        timestamps = h5.root.data2d_distorted.read(field='timestamp')
        assert len(timestamps)

        start_time = result_utils.timestamp2string(np.min(timestamps))
        stop_time = result_utils.timestamp2string(np.max(timestamps))

        doc['_id'] = 'h5:%s:%s'%(dataset_short,filename_short)
        doc['cam_ids'] = sorted(cam_id2camns.keys())
        doc['camns'] = sorted(camn2cam_id.keys())
        doc['comments'] = ''
        doc['dataset'] = 'dataset:'+dataset_short
        doc['filename'] = filename_short
        doc['filesize'] = os.stat( filename )[stat.ST_SIZE]

        doc['has_2d_position'] = hasattr(h5.root, 'data2d_distorted')
        doc['has_2d_orientation'] = False # XXX TODO FIXME

        doc['has_3d_position'] = hasattr(h5.root, 'kalman_estimates')
        doc['has_3d_orientation'] = False # XXX TODO FIXME

        doc['has_calibration'] = has_calib

        doc['source'] = 'original data' # XXX TODO FIXME

        doc['start_frame'] = start_frame
        doc['start_time'] = start_time
        doc['stop_frame'] = stop_frame
        doc['stop_time'] = stop_time

        doc['type'] = 'h5'
        import time
        doc['uuid'] = time.time()

        h5.close()

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

    doit( filenames, config )

if __name__=='__main__':
    main()

