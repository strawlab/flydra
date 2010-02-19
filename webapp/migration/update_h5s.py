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
    for row in db.view('fw/original_h5s'):
        (dataset, my_enum) = row.key
        if my_enum==0:
            # this is just a dataset row, skip to h5 row
            continue
        doc = db[row.id]
        #pprint.pprint(dict(doc))

        assert doc['has_2d_data']==True
        assert doc['has_3d_data']==False

        del doc['has_2d_data']
        doc['has_2d_position']=True
        doc['has_2d_orientation']=False

        del doc['has_3d_data']
        doc['has_3d_position']=False
        doc['has_3d_orientation']=False

        #pprint.pprint(dict(doc))
        #print

        # update the view
        db[row.id] = doc


