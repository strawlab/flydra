import sys
from couchdb.client import Server
import subprocess
import states

def download_jobs(couch_url, db_name):
    couch_server = Server(couch_url)
    db = couch_server[db_name]

    # get all jobs that are created (not executing or complete)
    view_results = db.view('analysis/jobs')
    for row in view_results:
        if row.key != states.CREATED:
            continue

        #job_id = row.value['_id']
        print 'row',row
        print 'row.value',row.value
        print 'row.key',row.key
        print 'dir(row)',dir(row)
        print 'row.id',row.id
        print

        job_id = row.id

        doc = db[job_id]
        doc['state'] = states.SUBMITTED
        # XXX use virtualenv in ~/PY
        cmd = 'qsub -b y ~/PY/bin/flydra_sge_run_job %s %s %s'%(couch_url, db_name, job_id)
        subprocess.check_call(cmd,shell=True)
        db.update([doc])

def main():
    couch_url = sys.argv[1]
    db_name = sys.argv[2]
    download_jobs(couch_url, db_name)

if __name__=='__main__':
    main()
