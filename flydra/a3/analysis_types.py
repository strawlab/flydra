import collections
import time, datetime, os, glob, shutil, stat
import subprocess
import hashlib
import datanodes
import warnings
import flydra.sge_utils.config as config

def do_sha1sum(fname):
    fd = open(fname,mode='r')
    m = hashlib.sha1()
    while 1:
        buf = fd.read(1024*1024) # read 1 MB
        m.update(buf)
        if len(buf)==0:
            break
    return m.hexdigest()

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

    def prepare_sources( self, sge_job_doc, target_dir ):
        result = {}
        source_ids = sge_job_doc['sources']
        for source_id in source_ids:
            source_doc = self.db[source_id]
            print 'source_doc'
            print source_doc
            print
            if 'filename' in source_doc:
                filename = source_doc['filename']
                outpath = os.path.join( target_dir, filename )
                fullpath1 = os.path.join( config.src_dir, filename )
                if os.path.exists(fullpath1):
                    shutil.copy( fullpath1, target_dir )
                else:
                    fullpath2 = fullpath1 + '.lzma'
                    if os.path.exists(fullpath2):
                        outfd = open(outpath,mode='wb')
                        cmd = ['unlzma','--stdout',fullpath2]
                        subprocess.check_call(cmd,stdout=outfd)
                        outfd.close()
                    else:
                        raise RuntimeError('could not file source file %s'%filename)
            else:
                if '_attachments' in source_doc:
                    filenames = source_doc['_attachments'].keys()
                    assert len( filenames )==1
                    filename = filenames[0]
                    outpath = os.path.join( target_dir, filename )
                    contents = self.db.get_attachment( source_id, filename )
                    #print 'contents'
                    #print contents.read()
                    outfd = open(outpath,mode='wb')
                    outfd.write(contents.read())
                    outfd.close()
            result[source_id] = filename
        return result

    def copy_outputs( self, job_doc, tmp_dirname ):
        '''copy known outputs from dirname'''
        copied_files = []
        warnings.warn('copying outputs not implemented')
        return copied_files

class EKF_based_3D_position( AnalysisType ):
    name = 'EKF-based 3D position'
    short_description = 'convert 2D data and calibration into 3D position data'
    source_node_types = ['2d position', 'calibration']
    base_cmd = 'flydra_kalmanize'

    def __init__(self,*args,**kwargs):
        super( EKF_based_3D_position, self).__init__(*args,**kwargs)
        self.choices['--dynamic-model'] = [None,
                                           "'EKF flydra, units: mm'",
                                           "'EKF humdra, units: mm'",
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

    def convert_sources_to_cmdline_args(self, sge_job_doc, source_info ):
        sources = sge_job_doc['sources']
        short_sources = sources
        docs = []
        for snt in self.source_node_types:
            ndocs,short_sources = self._get_docs_shortened_sources(snt,short_sources)
            docs.extend(ndocs)
        cmdline_args = []
        for (node_type,doc) in docs:
            if node_type == '2d position':
                cmdline_args.append( source_info[doc['_id']] )
            elif node_type == 'calibration':
                cmdline_args.append('--reconstructor='+source_info[doc['_id']] )
            else:
                raise ValueError('unknown node_type as source: %s'%node_type)
        return cmdline_args

    def copy_outputs( self, sge_job_doc, tmp_dirname, save_dir_base ):
        '''copy known outputs from dirname'''
        copy_files = glob.glob(os.path.join(tmp_dirname,'*.kalmanized.h5'))
        assert len(copy_files)==1
        fname = copy_files[0]

        copy_files = [f.replace(tmp_dirname+'/','') for f in copy_files]

        outdir = os.path.join( save_dir_base, sge_job_doc['datanode_id'] )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        shutil.copy2( fname, outdir )

        fname_only = os.path.split( fname )[-1]
        filesize = os.stat(fname)[stat.ST_SIZE]
        sha1sum = do_sha1sum(fname)

        datanode_doc_custom = {'filename':fname_only,
                               'filesize':filesize,
                               'sha1sum':sha1sum,
                               }
        return copy_files, datanode_doc_custom

def analysis_type_factory( db, class_name ):
    klass = globals()[class_name]
    atype = klass(db)
    return atype

class_names = ['EKF_based_3D_position']
