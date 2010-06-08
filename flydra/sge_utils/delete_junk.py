from couchdb.client import Server
import sys, os, shutil

def delete_junk(couch_url, db_name, datasink_dir):
    couch_server = Server(couch_url)
    db = couch_server[db_name]
    view_results = db.view('analysis/jobs')
    DELETE = False
    for row in view_results:
        print row.id
        doc = db[row.id]
        print doc
        print

        jobdir = os.path.join(datasink_dir, row.id)
        if DELETE:
            if os.path.exists( jobdir ):
                shutil.rmtree( jobdir )

            db.delete( doc )

    node_types = ['3d position','plots']
    for node_type in node_types:
        startkey=["dataset:humdra_200809", node_type]
        endkey=["dataset:humdra_200809", node_type, {}]
        view_results = db.view('analysis/datanodes-by-dataset-and-property',
                               startkey=startkey,
                               endkey=endkey,
                               reduce=False,
                               )
        for row in view_results:
            print row.id
            if row.id is None:
                continue
            doc = db[row.id]
            print doc
            print

            jobdir = os.path.join(datasink_dir, row.id)
            if DELETE:
                if os.path.exists( jobdir ):
                    shutil.rmtree( jobdir )

                db.delete( doc )

def main():
    couch_url = sys.argv[1]
    db_name = sys.argv[2]
    datasink_dir = sys.argv[3]
    delete_junk(couch_url, db_name, datasink_dir)

if __name__=='__main__':
    main()

