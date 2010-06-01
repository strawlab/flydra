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

    


class_names = ['EKF_based_3D_position']
