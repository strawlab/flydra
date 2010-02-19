#!/usr/bin/env python
from couchdb.client import Server
import os, subprocess

server_name = 'http://localhost:5984/'
server = Server(server_name)
metadb_name = 'flydraweb_metadata'

# install meta database design doc -------------------------------

orig_dir = os.path.abspath(os.curdir)
os.chdir( 'couchapp_metadb' )

server_metadb_name = server_name + metadb_name
subprocess.check_call( ['couchapp','push','.',server_metadb_name] )

os.chdir(orig_dir)

# get databases -----------------------------------------------
def get_db_names( server, metadb_name ):
    metadb = server[metadb_name]
    db_names = [row.key for row in metadb.view('meta/databases')]
    return db_names
db_names = get_db_names( server, metadb_name )

# install in databases -------------------------------------------
orig_dir = os.path.abspath(os.curdir)
os.chdir( 'couchapp_perdb' )

for db_name in db_names:
    subprocess.check_call( ['couchapp','push','.',db_name] )

os.chdir(orig_dir)
