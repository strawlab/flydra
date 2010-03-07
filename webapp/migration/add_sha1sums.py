#!/usr/bin/env python

# Used upon initial use of localglobal system to add sha1sum in
# addition to filename to documents.

import couchdb.client
import pprint
import dateutil.parser
import datetime
from uuid import uuid4

def main():

    # Search database to find valid parents and create all possible
    # children.

    uri = 'http://localhost:7840'
    server = couchdb.client.Server(uri)
    db = server['altshuler']

    lgdb = server['localglobal']
    design_name = 's3_key'
    try:
        del lgdb['_design/'+design_name]
    except couchdb.client.ResourceNotFound:
        pass
    lgdb['_design/' + design_name] = {
        'language': 'javascript',
        'views': {
                   's3_key': {'map': "function(doc) { if (doc.type=='s3') { emit(doc.key,doc.sha1sum); } }"},
                   },
        }

    dataset = 'dataset:humdra_200809'

    result = db.view('analysis/doc_types',reduce=False)
    count = 0
    for row in result:
        #print row.id, row.key
        if row.key[0]  in ['datanode','calibration','dataset']:
            continue
        count+=1

        if count%10==0:
            print 'count',count
        doc = db[row.id]

        # get uncompressed filename
        u_fname = doc['filename']

        # skip if sha1sum already present
        if 'sha1sum' in doc.keys():
            print 'uncompressed fname %s: uncompressed sha1sum %s, skipping'%( u_fname, doc['sha1sum'] )
            continue

        # get compressed filename
        c_fname = u_fname + '.lzma'

        # find sha1sum of compressed filename
        lgresult = list(lgdb.view(design_name+'/s3_key',
                             startkey = c_fname,
                             endkey = c_fname + 'ZZZZ',
                                  ))
        lgcount = 0
        for lgrow in lgresult:
            assert lgrow.key == c_fname
            c_sha1sum = lgrow.value
            lgcount += 1
            print 'compressed fname %s: compressed sha1sum %s, doc id %s'%( c_fname, c_sha1sum, lgrow.id )

        assert lgcount == 1

        # find sha1sum of uncompressed filename
        lgresult = list(lgdb.view('localglobal/sha1sum',
                                  startkey = c_sha1sum,
                                  endkey = c_sha1sum + 'ZZZZ',
                                  reduce=False,
                                  ))
        lgcount = 0
        for lgrow in lgresult:
            assert lgrow.key == c_sha1sum
            lgdoc = lgdb[ lgrow.id ]
            if lgdoc['type'] != 'compressed':
                continue
            u_sha1sum = lgdoc['uncompressed_sha1sum']
            lgcount += 1
        assert lgcount == 1
        print '  uncompressed sha1sum %s'%u_sha1sum

        doc['sha1sum'] = u_sha1sum
        db[row.id] = doc # update the document with sha1sum

    print 'count',count

if __name__=='__main__':
    main()

