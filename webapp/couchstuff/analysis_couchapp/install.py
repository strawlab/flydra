#!/usr/bin/env python
import subprocess

credentials = 'astraw:abc123@'
db_names = ['altshuler','pitch']

def run(cmd):
    print ' '.join(cmd)
    if 1:
        # why does this make it work?
        cmd = ' '.join(cmd)
    subprocess.check_call(cmd,shell=True)

for db_name in db_names:
    cmd = ['couchapp','push','.','http://%slocalhost:3388/%s'%(credentials,db_name)]
    run(cmd)
