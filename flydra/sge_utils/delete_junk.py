from couchdb.client import Server
import sys, os, shutil

def delete_junk(couch_url, db_name, datasink_dir):
    couch_server = Server(couch_url)
    db = couch_server[db_name]
    view_results = db.view('analysis/empty')
    for row in view_results:
        print row.id
        print row.value
        print

        jobdir = os.path.join(datasink_dir, row.id)
        if os.path.exists( jobdir ):
            shutil.rmtree( jobdir )

        db.delete( row.value )

def main():
    couch_url = sys.argv[1]
    db_name = sys.argv[2]
    datasink_dir = sys.argv[3]
    delete_junk(couch_url, db_name, datasink_dir)

if __name__=='__main__':
    main()

