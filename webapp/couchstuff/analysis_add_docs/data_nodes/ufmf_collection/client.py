#!/usr/bin/env python
import couchdb.client
import pprint
import dateutil.parser
import datetime
from uuid import uuid4
import base

# TODO: query DataNode view for ufmf rather than hard-code
# ufmf-specific information.

class UfmfSet:
    """create a 'datanode' document for a collection of ufmf files"""
    def __init__(self,first_doc):
        self.member_docs = [first_doc]
        self.dataset = first_doc['dataset']
        latest_start_time = first_doc['start_time']
        earliest_stop_time = first_doc['stop_time']
        self.overlap_times = base.TimeRange( latest_start_time,
                                             earliest_stop_time )
    def append(self,doc):
        if self.dataset != doc['dataset']:
            raise ValueError('doc not from my dataset')
        time_range_intersection = self.overlap_times.intersect(
            base.TimeRange( doc['start_time'], doc['stop_time'] ))
        if not time_range_intersection.duration > datetime.timedelta(0):
            raise base.NoTimeOverlapError(
                'cannot append doc to UfmfSet unless there is overlap')
        self.overlap_times = time_range_intersection
        self.member_docs.append( doc )
    def to_data_node(self):
        ids = sorted([ d['_id'] for d in self.member_docs ])
        parts = [ i.split(':',2) for i in ids ]
        fnames = []
        for ufmf,dataset,fname in parts:
            full_dataset = 'dataset:'+dataset # normalize to dataset _id
            try:
                assert ufmf=='ufmf'
                assert full_dataset==self.dataset
                fnames.append(fname)
            except:
                print 'i',i
                print parts
                print full_dataset
                print self.dataset
                raise

        doc = {
            'type'       : 'datanode',
            'properties' : ['ufmf collection'],
            'start_time' : base.encode_time(self.overlap_times.start),
            'stop_time'  : base.encode_time(self.overlap_times.stop),
            'dataset'    : full_dataset,
            'sources'    : ids,
            'status_tags': ['built','collection','virtual'],
            'filenames'  : sorted([ d['filename'] for d in self.member_docs ]),
            }
        common_name = base.string_start_intersection( fnames ).rstrip('_')
        common_name = common_name.rstrip('_cam')
        if len(common_name):
            doc['_id'] = 'datanode:%s:%s'%(dataset,common_name)
        base.validate_data_node(doc)
        return doc

def main():

    # Search database to find valid parents and create all possible
    # children.

    uri = 'http://localhost:3388'
    server = couchdb.client.Server(uri)
    db = server['pitch']

    design_name = 'ufmf_collection'
    try:
        del db['_design/'+design_name]
    except couchdb.client.ResourceNotFound:
        pass

    view_name = 'coll1'
    db['_design/'+design_name] = {
        'language': 'javascript',
        'views': { view_name: { 'map': open('map.js').read() },
                   },
        }
    try:

        result = list(db.view(design_name+'/'+view_name))
        collections = []
        for row in result:
            #print type(row)
            #print row.keys()
            #pprint.pprint(row.id)
            #pprint.pprint(row.value)
            doc = row.value
            try:
                base.decode_time_dict(doc)
            except:
                #print 'error decoding'
                #pprint.pprint(doc)
                raise
            #pprint.pprint(doc)

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

