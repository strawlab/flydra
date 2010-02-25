#!/usr/bin/env python
import couchdb.client
import pprint
import dateutil.parser
import datetime
from uuid import uuid4
import base

def main():

    # Search database to find valid parents and create all possible
    # children.

    uri = 'http://localhost:3388'
    server = couchdb.client.Server(uri)
    db = server['altshuler']
    dataset = 'dataset:humdra_200809'

    design_name = 'ekf_3d_pos'
    try:
        del db['_design/'+design_name]
    except couchdb.client.ResourceNotFound:
        pass

    view_name = 'ekf_3d_pos'
    db['_design/'+design_name] = {
        'language': 'javascript',
        'views': { #view_name: { 'map': open('map.js').read() },
                   'calibs': {'map': "function(doc) { if (doc.type=='calibration') { emit(null,null); } }"},
                   },
        }
    try:


# TODO: query DataNode view for calibration rather than hard-code
# ufmf-specific information.

        result = list(db.view(design_name+'/'+'calibs'))
        if not len(result)==1:
            raise NotImplementedError('need to provide means to select calibration')
        calibration_doc = db[result[0].id]

        result = db.view('analysis/DataNode',
                         startkey=[dataset],
                         endkey=[dataset,{}],
                         reduce=False)
        source_ids = []
        for row in result:
            assert row.key[0]==dataset, "wrong dataset" # query already did this
            if "2d position" in row.key[1]:
                source_ids.append( row.id )
        print 'source_ids',source_ids[:5]
        1/0
        result = list(db.view(design_name+'/'+view_name))
        collections = []
        for row in result:
            doc = row.value
            try:
                base.decode_time_dict(doc)
            except:
                raise

            success = False
            for coll in collections:
                try:
                    coll.append( doc )
                    success = True
                    break
                except base.NoTimeOverlapError:
                    pass
            if not success:
                # start a new ufmf collection
                collections.append( UfmfSet(doc) )

    finally:
        del db['_design/'+design_name]

    db.update( [ ufmf_set.to_data_node() for ufmf_set in collections ] )

    for ufmf_set in collections[:5]:
        #print 'collection'
        #pprint.pprint(ufmf_set.member_docs)
        pprint.pprint(ufmf_set.to_data_node())
        #print

if __name__=='__main__':
    main()

