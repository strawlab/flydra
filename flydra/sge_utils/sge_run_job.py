import sys
from couchdb.client import Server

def run_job(couch_url, db_name, doc_id):
    couch_server = Server(couch_url)
    db = couch_server[db_name]
    job_doc = db[doc_id]
    print job_doc

def main():
    couch_url = sys.argv[1]
    db_name = sys.argv[2]
    doc_id = sys.argv[3]
    run_job(couch_url, db_name, doc_id)

if __name__=='__main__':
    main()
