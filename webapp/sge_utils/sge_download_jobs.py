import sys
from couchdb.client import Server
import subprocess

def download_jobs(couch_url, db_name):
    couch_server = Server(couch_url)
    db = couch_server[db_name]
    view_results = db.view('analysis/jobs')
    for row in view_results:
        job_id = row.value['_id']
        cmd = 'qsub -b y ~/PY/bin/flydra_sge_run_job %s %s %s'%(couch_url, db_name, job_id)
        subprocess.check_call(cmd,shell=True)

def main():
    couch_url = sys.argv[1]
    db_name = sys.argv[2]
    download_jobs(couch_url, db_name)

if __name__=='__main__':
    main()
