#!/usr/bin/env python
import subprocess

credentials = 'astraw:abc123@'
host = 'localhost:5984'
db_names = ['altshuler','pitch']

def run(cmd):
    print ' '.join(cmd)
    if 1:
        # why does this make it work?
        cmd = ' '.join(cmd)
    subprocess.check_call(cmd,shell=True)

for db_name in db_names:
    db_url = 'http://%s%s/%s'%(credentials,host,db_name)
    cmd = ['couchapp','push','.',db_url]
    run(cmd)

    print '--------- your db is at:'
    print db_url+'/_design/analysis/_list/index/datasets'
    print
