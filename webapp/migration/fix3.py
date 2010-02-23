#!/usr/bin/env python
import sys, pprint, json
from multiprocessing import Pool
import glob, os
import subprocess
import motmot.ufmf.ufmfstats as ufmfstats
import datetime
import pytz # from http://pytz.sourceforge.net/


server_name = 'http://localhost:3388/'

upload_dir = '/media/humdra-disk/humdra/2008_september/upload'
tmp_dir = '/media/humdra-disk/humdra/tmp'

files = glob.glob(os.path.join(upload_dir,'*ufmf.lzma'))
files.sort()

fshort = []
for fname in files:
    trim = os.path.split(fname)[-1]
    dst = os.path.join( tmp_dir, trim )
    if not os.path.exists(dst):
        os.link( fname, dst )
    fshort.append(trim)

if 0:
    # filter by already done
    orig_fshort = fshort
    fshort = []
    for target in orig_fshort:
        id = 'ufmf:humdra_200809:'+target
        db_name = 'altshuler'
        print 'id',id
        url = server_name + db_name + '/' + id
        result = get_couch(url)
        doc = json.loads(result)
        if doc['start_time']=='xxx':
            print 'doing %s again'%target
            fshort.append( target )
        else:
            print 'already did %s'%target

def timestamp2string(ts_float,timezone='US/Pacific'):
    pacific = pytz.timezone(timezone)
    dt_ts = datetime.datetime.fromtimestamp(ts_float,pacific)
    # dt_ts.ctime()
    return dt_ts.isoformat()

def get_couch(url):
    p = subprocess.Popen(['/usr/bin/curl',url],stdout=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('exitcode error')
    return p.stdout.read()

def put_couch(url,data):
    p = subprocess.Popen(['/usr/bin/curl','-X','PUT',url,'-d',data],stdout=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('exitcode error')
    return p.stdout.read()

def fix_stat_dict(d):
    for k in ['data_start_time_float','data_stop_time_float']:
        newkey = k[:-6]
        if k in d:
            d[newkey] = timestamp2string(d[k])
            del d[k]
        else:
            d[newkey] = None
    for kk,v in d['frame_info'].iteritems():
        d2 = v
        for k in ['start_time_float','stop_time_float']:
            newkey = k[:-6]
            d2[newkey] = timestamp2string( d2[k] )
            del d2[k]
        if kk=='raw':
            # copy start/stop times from raw frames (even if no data)
            d['start_time'] = d2['start_time']
            d['stop_time'] = d2['stop_time']

def doit(fname):
    try:
        fname = os.path.join(tmp_dir,fname)
        assert fname.endswith('.lzma')
        target = fname[:-5]
        cmd = ['/usr/bin/unlzma','--keep',fname]
        print 'uncompressing to',target
        if not os.path.exists(target):
            subprocess.check_call(cmd)
        try:

            id = 'ufmf:humdra_200809:'+os.path.split(target)[-1]
            # get databases -----------------------------------------------
            db_name = 'altshuler'
            print 'id',id
            url = server_name + db_name + '/' + id
            result = get_couch(url)
            doc = json.loads(result)

            pprint.pprint(doc)
            if doc['start_time']=='xxx':
                stats = ufmfstats.collect_stats(target)
                pprint.pprint(stats)
                fix_stat_dict(stats)
                print
                pprint.pprint(stats)

            doc.update(stats)
            put_couch(url,json.dumps(doc))
        finally:
            os.unlink(target)
            pass
    except:
        import traceback
        print 'vvvvv in process for',fname
        traceback.print_exc()
        print '^^^^^ in process for',fname
        raise

if 1:
    pool = Pool(processes=7)
    print len(fshort)
    pool.map( doit, fshort)
else:
    map( doit, fshort)
## # migrate h5 documents
## allfiles = []
## for db_name in db_names:
##         if not 'type' in doc:
##             print 'fail 1:',id
##             if 'ufmf' in id:
##                 print doc.keys()
##             continue
##         if doc['type']!='ufmf':
##             print 'fail 2:',id
##             continue

##         doc = db[id]
##         info = (id,doc['filename'])
##         allfiles.append(info)


