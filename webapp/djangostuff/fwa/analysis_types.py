import collections
import time, datetime

# --- pure descriptions that could be refactored for other purposes (e.g. dependency diagrams)

class AnalysisType(object):
    def __init__(self,db=None):
        assert db is not None
        self.db = db
        self.choices = {}
    def convert_parents_to_cmdline_args(self,parents):
        raise NotImplementedError('Abstract base class')
    def _get_docs_shortened_parents(self,node_type,parents):
        docs = []
        unused_parents = []

        for parent in parents:
            doc = self.db[parent]

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
                unused_parents.append(parent)

        return docs, unused_parents

class EKF_based_3D_position( AnalysisType ):
    name = 'EKF-based 3D position'
    short_description = 'convert 2D data and calibration into 3D position data'
    parent_node_types = ['2d position', 'calibration']
    base_cmd = 'flydra_kalmanize'

    def __init__(self,*args,**kwargs):
        super( EKF_based_3D_position, self).__init__(*args,**kwargs)
        self.choices['--dynamic-model'] = [None,
                                           'EKF flydra, units: mm',
                                           'EKF humdra, units: mm',
                                           ]

    def convert_parents_to_cmdline_args(self,parents):
        short_parents = parents
        docs = []
        for pnt in self.parent_node_types:
            ndocs,short_parents = self._get_docs_shortened_parents(pnt,short_parents)
            docs.extend(ndocs)
        cmdline_args = []
        for (node_type,doc) in docs:
            if node_type == '2d position':
                cmdline_args.append( doc['filename'] )
            elif node_type == 'calibration':
                cmdline_args.append('--reconstructor=xxx')
            else:
                raise ValueError('unknown node_type as parent: %s'%node_type)
        #return ['unfinished commandline args']
        return cmdline_args

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
        self.dataset = dataset
        assert isinstance(analysis_type,AnalysisType)
        self.analysis_type = analysis_type

    def validate_new_batch_jobs_request( self, orig_query ):
        query = collections.defaultdict(list)
        for key in orig_query:
            query[key] = orig_query.getlist(key)

        n_values = dict([(pnt,len(query[pnt])) for pnt in self.analysis_type.parent_node_types])
        n_new_docs = max( n_values.itervalues() )
        for pnt, n in n_values.iteritems():
            # check there are all N or 1 documents for each parent node type
            if not ((n == n_new_docs) or (n == 1)):
                raise InvalidRequest('For parent datanode type "%s", invalid '
                                     'number of datanodes specified.'%pnt)

        # XXX TODO auto-sort corresponding documents??

        new_batch_jobs = []
        for i in range(n_new_docs):
            parents = []
            for pnt in self.analysis_type.parent_node_types:
                if n_values[pnt] == 1:
                    parents.append( query[pnt][0] ) # only one, always use it
                else:
                    parents.append( query[pnt][i] )

            doc = { 'parents':parents }
            new_batch_jobs.append( doc )

        # finished with parents
        for pnt in self.analysis_type.parent_node_types:
            del query[pnt]

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
            parents = doc['parents']
            doc['class_name'] = self.analysis_type.__class__.__name__
            doc['choices'] = choices

        return new_batch_jobs

class_names = ['EKF_based_3D_position']


def submit_jobs( db, new_batch_jobs, user=None ):
    #now = datetime.datetime.fromtimestamp( time.time() )
    now = datetime.datetime.utcnow()
    nowstr = now.isoformat()
    docs = []
    
    for doc in new_batch_jobs:
        doc['type'] = 'job'
        doc['submit_time'] = nowstr
        doc['state'] = 'created'
        docs.append(doc)
    
    # bulk upload docs
    db.update(docs)

    hack_error_message = '\nsubmitted %s\n'%repr(docs)
    return hack_error_message
