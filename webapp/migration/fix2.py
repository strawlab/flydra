#!/usr/bin/env python
import sys, pprint, json
from multiprocessing import Pool
import glob, os
import subprocess

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

def doit(fname):
    try:
        assert fname.endswith('.lzma')
        target = fname[:-5]
        cmd = ['/usr/bin/unlzma','--keep',fname]
        #print 'uncompressing',target
        #subprocess.check_call(cmd)
        try:
            id = 'ufmf:humdra_200809:'+target
            # get databases -----------------------------------------------
            db_name = 'altshuler'
            if 1:
                print 'id',id
                url = server_name + db_name + '/' + id
                result = get_couch(url)
                doc = json.loads(result)
                #pprint.pprint(doc)
                doc['start_time']='xxx'
                doc['stop_time']='yyy'
                put_couch(url,json.dumps(doc))
            else:
                print id
                1/0
        finally:
            #os.unlink(target)
            pass
    except:
        import traceback
        print 'vvvvv in process for',fname
        traceback.print_exc()
        print '^^^^^ in process for',fname
        raise

if 0:
    pool = Pool(processes=4)
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


