#!/usr/bin/env python
from couchdb.client import Server
import sys, pprint, re

server_name = 'http://localhost:5984/'
server = Server(server_name)
metadb_name = 'flydraweb_metadata'

# get databases -----------------------------------------------
def get_db_names( server, metadb_name ):
    metadb = server[metadb_name]
    db_names = [row.key for row in metadb.view('meta/databases')]
    return db_names
db_names = get_db_names( server, metadb_name )

reufmf = re.compile( r'^small_(\d+_\d+)_(.*_.*)\.ufmf$' )

# migrate h5 documents
for db_name in db_names:
    db = server[db_name]
    for row in db.view('fw/ufmfs'):
        (dataset, my_enum) = row.key
        if my_enum==0:
            # this is just a dataset row, skip to h5 row
            continue
        doc = db[row.id]
        #pprint.pprint(dict(doc))

        if 'cam_id' in doc:
            # already done
            continue

        try:

            matchobj = reufmf.search(doc['filename'])
            assert matchobj is not None
            datetime, cam_id = matchobj.groups()
            doc['cam_id'] = cam_id

            #pprint.pprint(dict(doc))
            #print

            # update the view
            db[row.id] = doc

        except Exception,err:
            print 'error %s on: %s'%(err,row.id)
            continue
