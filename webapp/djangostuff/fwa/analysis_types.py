import collections
import time, datetime
import datanodes

# --- pure descriptions that could be refactored for other purposes (e.g. dependency diagrams)

class AnalysisType(object):
    def __init__(self,db=None):
        assert db is not None
        self.db = db
        self.choices = {}
#     def convert_sources_to_cmdline_args(self,sources):
#         raise NotImplementedError('Abstract base class')
    def _get_docs_shortened_sources(self,node_type,sources):
        docs = []
        unused_sources = []

        for source in sources:
            doc = self.db[source]

            accept = False
            if node_type=='2d position':
                if doc['type']=='h5' and doc['has_2d_position']:
                    accept=True
            elif node_type=='calibration':
                if doc['type']=='h5' and doc['has_calibration']:
                    accept=True
                elif doc['type']=='calibration':
                    accept=True
            else:
                raise NotImplementedError('unknown node_type %s'%node_type)
        
            if accept:
                docs.append( (node_type,doc) )
            else:
                unused_sources.append(source)

        return docs, unused_sources

class EKF_based_3D_position( AnalysisType ):
    name = 'EKF-based 3D position'
    short_description = 'convert 2D data and calibration into 3D position data'
    source_node_types = ['2d position', 'calibration']
    base_cmd = 'flydra_kalmanize'

    def __init__(self,*args,**kwargs):
        super( EKF_based_3D_position, self).__init__(*args,**kwargs)
        self.choices['--dynamic-model'] = [None,
                                           'EKF flydra, units: mm',
                                           'EKF humdra, units: mm',
                                           ]

    def get_datanode_doc_properties( self, sge_job_doc ):
        source_list = sge_job_doc['sources']
        props = {
            'type':'h5',
            'has_2d_orientation':False,
            'has_2d_position':False,
            'has_3d_orientation':False,
            'has_3d_position':True,
            'has_calibration':True,
            'source':'computed from sources %s'%source_list,
            'comments':'',
            }
        return props

#     def convert_sources_to_cmdline_args(self,sources):
#         short_sources = sources
#         docs = []
#         for snt in self.source_node_types:
#             ndocs,short_sources = self._get_docs_shortened_sources(snt,short_sources)
#             docs.extend(ndocs)
#         cmdline_args = []
#         for (node_type,doc) in docs:
#             if node_type == '2d position':
#                 cmdline_args.append( doc['filename'] )
#             elif node_type == 'calibration':
#                 cmdline_args.append('--reconstructor=xxx')
#             else:
#                 raise ValueError('unknown node_type as source: %s'%node_type)
#         #return ['unfinished commandline args']
#         return cmdline_args

def analysis_type_factory( db, class_name ):
    klass = globals()[class_name]
    atype = klass(db)
    return atype

#def get_datanode_doc_properties( sge_job_doc ):

    

# --- various django and CouchDB specific stuff ----------------
from django import forms

class InvalidRequest(ValueError):
    def __init__(self,err):
        super( InvalidRequest, self ).__init__(err)
        self.human_description = err

def add_fields_to_form( form, analysis_type ):
    assert isinstance(analysis_type,AnalysisType)
    for name, valid_values in analysis_type.choices.iteritems():
        form_choices = []
        for elem in valid_values:
            if elem is None:
                form_elem =  ('','<default value>')
            else:
                form_elem = (elem,elem)
            form_choices.append( form_elem )
        form.fields[name] = forms.ChoiceField(choices = form_choices)

class Verifier(object):
    """helper class to verify arguments from POST request"""
    def __init__(self, db, dataset, analysis_type):
        self.db = db
        self.dataset = 'dataset:'+dataset
        assert isinstance(analysis_type,AnalysisType)
        self.analysis_type = analysis_type

    def validate_new_batch_jobs_request( self, orig_query ):
        query = collections.defaultdict(list)
        for key in orig_query:
            query[key] = orig_query.getlist(key)

        n_values = dict([(snt,len(query[snt])) for snt in self.analysis_type.source_node_types])
        n_new_docs = max( n_values.itervalues() )
        for snt, n in n_values.iteritems():
            # check there are all N or 1 documents for each source node type
            if not ((n == n_new_docs) or (n == 1)):
                raise InvalidRequest('For source datanode type "%s", invalid '
                                     'number of datanodes specified.'%snt)

        # XXX TODO auto-sort corresponding documents??

        new_batch_jobs = []
        for i in range(n_new_docs):
            sources = []
            for snt in self.analysis_type.source_node_types:
                if n_values[snt] == 1:
                    sources.append( query[snt][0] ) # only one, always use it
                else:
                    sources.append( query[snt][i] )

            doc = { 'sources':sources,
                    'junk' : True, # XXX delete these docs when ready for production
                    }
            new_batch_jobs.append( doc )

        # finished with sources
        for snt in self.analysis_type.source_node_types:
            del query[snt]

        # now handle batch params that apply to all documents
        choices = []
        for choice_name in self.analysis_type.choices:
            posted_value = query.pop(choice_name)
            assert len(posted_value)==1
            posted_value = posted_value[0]

            valid_values = self.analysis_type.choices[choice_name]
            if (posted_value == '') and (None in valid_values):
                # use default
                continue
            assert posted_value in valid_values
            choices.append( (choice_name, posted_value) )

        # make sure no unhandled request data
        if len(query.keys()):
            raise InvalidRequest('Invalid request made with keys %s'%
                                 (query.keys(),))

        for doc in new_batch_jobs:
            doc['class_name'] = self.analysis_type.__class__.__name__
            doc['choices'] = choices
            doc['dataset'] = self.dataset

        return new_batch_jobs

class_names = ['EKF_based_3D_position']

def make_datanode_doc_for_sge_job( db, sge_job_doc ):
    assert sge_job_doc['type'] == 'job'
    
    start = None
    stop = None
    for source in sge_job_doc['sources']:
        sstart, sstop = datanodes.get_start_stop_times( db, source )
        # find intersection of all times
        if start is None or sstart > start:
            start = sstart
        if stop is None or sstop < stop:
            stop = sstop

    datanode_doc = {
        #'type' : 'datanode',  # let analysis_type fill this in
        'junk' : True, # XXX delete these docs when ready for production
        # 'sources' : sge_job_doc['sources'], # let analysis_type fill this in
        #'status_tags': ["unbuilt"], # let analysis_type fill this in
        'dataset' : sge_job_doc['dataset'],
        }

    if start is not None:
        datanode_doc['start_time'] = start.isoformat()
    if stop is not None:
        datanode_doc['stop_time'] = stop.isoformat()

    if 1:
        # set filenames, properties
        atype = analysis_type_factory( db, sge_job_doc['class_name'] )
        specific_properties = atype.get_datanode_doc_properties(sge_job_doc)
        datanode_doc.update(specific_properties)
    return datanode_doc

def submit_jobs( db, new_batch_jobs, user=None ):
    #now = datetime.datetime.fromtimestamp( time.time() )
    now = datetime.datetime.utcnow()
    nowstr = now.isoformat()
    sge_job_docs = []
    
    for sge_job_doc in new_batch_jobs:
        sge_job_doc['type'] = 'job'
        sge_job_doc['submit_time'] = nowstr
        sge_job_doc['state'] = 'created'
        sge_job_docs.append(sge_job_doc)

    datanode_docs = [ make_datanode_doc_for_sge_job( db, doc ) for doc in sge_job_docs ]
    
    # upload datanode documents
    datanode_results = db.update(datanode_docs)

    try:
        assert len(datanode_results) == len(sge_job_docs)
        for ((upload_ok, upload_id, upload_rev), seg_job_doc) in zip(datanode_results,sge_job_docs):
            assert upload_ok
            sge_job_doc['datanode_id'] = upload_id
        
        # upload SGE job documents
        results = db.update( sge_job_docs )
    except:
        
        # on error, erase datanode documents
        for (upload_ok, upload_id, upload_rev) in datanode_results:
            print 'XXX should delete id %s'%upload_id

        raise

    hack_error_message = '\nsubmitted %s\n'%repr(sge_job_docs)
    return hack_error_message
