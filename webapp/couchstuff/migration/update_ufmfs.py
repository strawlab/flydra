#!/usr/bin/env python
from couchdb.client import Server
import sys, pprint

server_name = 'http://localhost:5984/'
server = Server(server_name)
metadb_name = 'flydraweb_metadata'

# get databases -----------------------------------------------
def get_db_names( server, metadb_name ):
    metadb = server[metadb_name]
    db_names = [row.key for row in metadb.view('meta/databases')]
    return db_names
db_names = get_db_names( server, metadb_name )

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

        if 'start_time' in doc:
            # already done
            continue

        try:
            doc['start_time']=doc['data_start_time']
            doc['stop_time']=doc['data_stop_time']

            del doc['data_start_time']
            del doc['data_stop_time']

            #pprint.pprint(dict(doc))
            #print

            # update the view
            db[row.id] = doc

        except Exception,err:
            print 'error %s on: %s'%(err,row.id)
            continue
