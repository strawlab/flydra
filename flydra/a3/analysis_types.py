import collections
import time, datetime, os, glob, shutil, stat, sys, re
import subprocess
import hashlib
import datanodes
import warnings
import flydra.sge_utils.config as config
import Image
import re, glob, os, sys, tempfile, subprocess, Image, shutil

class VarType(object):
    pass

class RangeVarType(VarType):
    def __init__(self,min=None,max=None):
        self.min=min
        self.max=max
        super(RangeVarType,self).__init__()

class EnumVarType(VarType):
    def __init__(self,options=None):
        self.options = options
        super(EnumVarType,self).__init__()
    def get_options(self):
        return self.options

class FloatVarType(RangeVarType):
    pass

class IntVarType(RangeVarType):
    pass

class BoolVarType(VarType):
    def __init__(self):
        super(BoolVarType,self).__init__()

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

def safe_copy(fullpath,target):
    '''don't leave unfinished copy lying around'''
    if os.path.isdir(target):
        fname = os.path.split(fullpath)[-1]
        outpath = os.path.join(target)
    else:
        outpath = target

    try:
        shutil.copy( fullpath, target )
    except Exception, err1:
        try:
            os.unlink(outpath)
        finally:
            raise err1


class AnalysisType(object):
    def __init__(self,db=None):
        assert db is not None
        self.db = db
        self.choices = {}
#     def convert_sources_to_cmdline_args(self,sources):
#         raise NotImplementedError('Abstract base class')

    def this_choices( self, sge_job_doc ):
        result = {}
        choices = sge_job_doc['choices']
        for choice in choices:
            assert len(choice)==2
            key = choice[0]
            if choice[1]==None:
                continue
            if choice[1]=='on':
                result[ key ] = True
            else:
                result[ key ] = choice[1]
            # XXX TODO: better parsing of types
        return result

    def _get_docs_shortened_sources(self,node_type,sources):
        docs = []
        unused_sources = []

        for source in sources:
            doc = self.db[source]

            accept = False
            if node_type=='2d position':
                if doc['type']=='h5' and doc['has_2d_position']:
                    accept=True
            elif node_type=='3d position':
                if doc['type']=='h5' and doc['has_3d_position']:
                    accept=True
            elif node_type=='calibration':
                if doc['type']=='h5' and doc['has_calibration']:
                    accept=True
                elif doc['type']=='calibration':
                    accept=True
            elif node_type=='ufmf collection':
                if doc['type']=='datanode' and 'ufmf collection' in doc['properties']:
                    accept=True
            else:
                raise NotImplementedError('unknown node_type %s'%node_type)
        
            if accept:
                docs.append( (node_type,doc) )
            else:
                unused_sources.append(source)

        return docs, unused_sources

    def _copy_source_file( self, filename, target_dir, built_dir, verbose=0 ):
        outpath = os.path.join( target_dir, filename )
        if os.path.exists(outpath):
            if verbose>=1:
                print 'outpath %s already exists, skipping copy'%outpath
                sys.stdout.flush()
            return

        fullpath1 = os.path.join( config.src_dir, filename )
        fullpath2 = fullpath1 + '.lzma'
        fullpath3 = os.path.join( built_dir, filename )

        tstart=time.time()
        if os.path.exists(fullpath1):
            if verbose>=1:
                print 'copying %s to %s'%(fullpath1,target_dir)
                sys.stdout.flush()
            safe_copy(fullpath1, target_dir)

        elif os.path.exists(fullpath2):
            if verbose>=1:
                print 'uncompressing %s to %s'%(fullpath2,target_dir)
                sys.stdout.flush()
            try:
                outfd = open(outpath,mode='wb')
                cmd = ['unlzma','--stdout',fullpath2]
                subprocess.check_call(cmd,stdout=outfd)
                outfd.close()
            except:
                os.unlink(outpath)
                raise
        elif os.path.exists(fullpath3):
            if verbose>=1:
                print 'copying (built) %s to %s'%(fullpath1,target_dir)
                sys.stdout.flush()
            safe_copy( fullpath3, target_dir )
        else:
            raise RuntimeError('could not find source file %s (tried %s)'%(filename,
                                                                           [fullpath1,
                                                                            fullpath2,
                                                                            fullpath3]))
        tstop=time.time()
        if verbose>=1:
            filesize = os.stat(outpath)[stat.ST_SIZE]
            dur = tstop-tstart
            print 'wrote %d bytes in %.1f sec ( %.1f MB/sec )'%(filesize,dur, filesize/dur/1024.0/1024.0)
            sys.stdout.flush()

    def prepare_sources( self, sge_job_doc, target_dir, verbose=0 ):
        result = {}
        source_ids = sge_job_doc['sources']
        for source_id in source_ids:
            source_doc = self.db[source_id]
            built_dir = os.path.join( config.sink_dir, source_doc['_id'] )
            if 'filename' in source_doc:
                filename = source_doc['filename']
                self._copy_source_file( filename, target_dir, built_dir, verbose=verbose )
                filename_str_list = filename # only one element
            elif 'filenames' in source_doc:
                filenames = source_doc['filenames']
                for filename in filenames:
                    self._copy_source_file( filename, target_dir, built_dir, verbose=verbose )
                filename_str_list = os.path.pathsep.join(filenames)
            elif '_attachments' in source_doc:
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
                filename_str_list = filename # only one element
            else:
                raise ValueError('no filename for source %s'%source_id)
            result[source_id] = filename_str_list
        return result

    def get_output_cmdline_args(self, sge_job_doc, source_info ):
        return []

    def get_cmdline_args_from_choices(self, sge_job_doc, source_info ):
        # XXX TODO: convert to use self.this_choices(sge_job_doc)
        choices = sge_job_doc['choices']
        cmdline_args = []
        for choice in choices:
            assert len(choice)==2
            if choice[1]==None:
                continue
            if choice[1]=='on':
                cmdline_args.append( choice[0] ) # BoolVarType
            else:
                cmdline_args.append( choice[0] + '=' + choice[1] )
        return cmdline_args

class PlotsAnalysisType( AnalysisType ):
    def get_datanode_doc_properties( self, sge_job_doc ):
        source_list = sge_job_doc['sources']
        props = {
            'type' : 'datanode',
            'sources' : source_list,
            'properties': ['plots'],
            'status_tags': ['unbuilt'],
            }
        return props

    def copy_outputs( self, sge_job_doc, tmp_dirname, save_dir_base ):
        '''copy known outputs from dirname'''
        copy_files = glob.glob(os.path.join(tmp_dirname,'*.png'))
        copy_files_short_fnames = [f.replace(tmp_dirname+'/','') for f in copy_files]

        datanode_doc_custom = {'imsize':{}}

        attachment_tuples = []
        for fname in copy_files:
            im = Image.open(fname)
            width,height = im.size
            buf = open(fname,mode='r').read()
            fname_only = os.path.split( fname )[-1]
            content_type = 'image/png'
            attachment_tuples.append( (buf,fname_only,content_type) )
            datanode_doc_custom['imsize'][fname_only] = (width,height)

        outputs = {'copied_files':copy_files_short_fnames,
                   'datanode_doc_custom':datanode_doc_custom,
                   'attachments':attachment_tuples,
                   }
        return outputs

class PlotSummary3D( PlotsAnalysisType ):
    name = 'Plot: summary position'
    short_description = 'plot of 3D position'
    source_node_types = ['3d position']
    base_cmd = 'flydra_analysis_plot_summary'

    def convert_sources_to_cmdline_args(self, sge_job_doc, source_info ):
        sources = sge_job_doc['sources']
        short_sources = sources
        docs = []
        for snt in self.source_node_types:
            ndocs,short_sources = self._get_docs_shortened_sources(snt,short_sources)
            docs.extend(ndocs)
        cmdline_args = []
        for (node_type,doc) in docs:
            if node_type == '3d position':
                cmdline_args.extend( ['-k', source_info[doc['_id']]] )
            else:
                raise ValueError('unknown node_type as source: %s'%node_type)
        assert len(cmdline_args)==2
        return cmdline_args

class EKF_based_3D_position( AnalysisType ):
    name = 'EKF-based 3D position'
    short_description = 'convert 2D data and calibration into 3D position data'
    source_node_types = ['2d position', 'calibration']
    base_cmd = 'flydra_kalmanize'

    def __init__(self,*args,**kwargs):
        super( EKF_based_3D_position, self).__init__(*args,**kwargs)
        self.choices['--dynamic-model'] = EnumVarType([None,
                                           "'EKF flydra, units: mm'",
                                           "'EKF hbird, units: mm'",
                                           "'EKF hydra, units: mm'",
                                           ])

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
        outputs = {'copied_files':copy_files,
                   'datanode_doc_custom':datanode_doc_custom}
        return outputs



class ImageBased2DOrientation( AnalysisType ):
    name = 'Image-based 2D orientation'
    short_description = 'analyze .ufmf images to extract 2D orientation'
    source_node_types = ['ufmf collection','3d position','2d position']
    base_cmd = 'flydra_analysis_image_based_orientation'

    def __init__(self,*args,**kwargs):
        super( ImageBased2DOrientation, self).__init__(*args,**kwargs)
        self.choices['--final-thresh'] = FloatVarType(min=0,max=1)
        self.choices['--stack-N-images'] = IntVarType()
        self.choices['--stack-N-images-min'] = IntVarType()
        self.choices['--save-images'] = BoolVarType()
        self.choices['--old-sync-timestamp-source'] = BoolVarType()
        self.choices['--no-rts-smoothing'] = BoolVarType()
        self.choices['--intermediate-thresh-frac'] =  FloatVarType(min=0,max=1)
        self.choices['--erode'] = IntVarType(min=0)
        self.choices['--start'] = IntVarType(min=0)
        self.choices['--stop'] = IntVarType(min=0)

    def _get_output_h5_fname( self, sge_job_doc ):
        idstr = sge_job_doc['_id']
        outfname = idstr+'.h5'
        return outfname

    def get_output_cmdline_args(self, sge_job_doc, source_info ):
        return ['--output-h5='+self._get_output_h5_fname(sge_job_doc) ]

    def get_datanode_doc_properties( self, sge_job_doc ):
        source_list = sge_job_doc['sources']
        props = {
            'type':'h5',
            'has_2d_orientation':True,
            'has_2d_position':True,
            'has_3d_orientation':False,
            'has_3d_position':False,
            'has_calibration':False,
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
            if node_type == '3d position':
                cmdline_args.extend( ['--kalman', source_info[doc['_id']]] )
            elif node_type == '2d position':
                cmdline_args.extend( ['--h5', source_info[doc['_id']]] )
            elif node_type == 'ufmf collection':
                cmdline_args.append('--ufmfs='+source_info[doc['_id']] )
            else:
                raise ValueError('unknown node_type as source: %s'%node_type)
        return cmdline_args

    def copy_outputs( self, sge_job_doc, tmp_dirname, save_dir_base ):
        '''copy known outputs from dirname'''



        # copy H5 file
        fname_short = self._get_output_h5_fname(sge_job_doc)
        fname = os.path.join( tmp_dirname, fname_short )
        copy_files = [fname_short]

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
        outputs = {'copied_files':copy_files,
                   'datanode_doc_custom':datanode_doc_custom,
                   }

        # compress saved images into movie, upload them, create link
        if self.this_choices( sge_job_doc ).get('--save-images',False):
            # the "--save-images" option was given
            save_image_fnames = glob.glob(os.path.join(tmp_dirname,'*.png'))            
            save_image_fnames = [f.replace(tmp_dirname+'/','') for f in save_image_fnames]
            saved_images = {}
            by_cam_id = self._parse_image_fnames( save_image_fnames )
            for cam_id in by_cam_id:
                for obj_id in by_cam_id[cam_id]:
                    out_fname = 'movie_' + sge_job_doc['datanode_id'] + '_' + cam_id + '_obj'+str(obj_id)+'.ogv'
                    full_out_fname = os.path.join( tmp_dirname, out_fname )
                    width,height = make_ogv( out_fname = full_out_fname, 
                                             files = by_cam_id[cam_id][obj_id],
                                             dirname=tmp_dirname )

                    # upload image to ADS's dreamhost account
                    hostname='69.163.194.242' # 'static.flydra.astraw.com'
                    dest_path=os.path.join('static.flydra.astraw.com',out_fname)
                    cmd = 'scp -p %s %s:%s'%( full_out_fname,
                                              hostname,
                                              dest_path )
                    subprocess.check_call(cmd, shell=True)

                    url = 'http://static.flydra.astraw.com/' + out_fname
                    saved_images[ url ] = (width,height)
            outputs['saved_images'] = saved_images
        return outputs

    def _parse_image_fnames(self, fnames ):
        by_cam_id = {}
        fname_re = re.compile(r'^av_obj(?P<obj_id>[0-9]+)_(?P<cam_id>.*)_frame(?P<frame>[0-9]+)\.png$')
        
        for fname in fnames:
            matchobj = fname_re.search(fname)
            cam_id = matchobj.group('cam_id')
            obj_id = int(matchobj.group('obj_id'))
            if cam_id not in by_cam_id:
                by_cam_id[cam_id] = {}
            if obj_id not in by_cam_id[cam_id]:
                by_cam_id[cam_id][obj_id] = []
            by_cam_id[cam_id][obj_id].append( fname )
        return by_cam_id

def make_ogv( out_fname = None, files = None, dirname=None ):
    files = files[:]
    files.sort()
    rename_dir = os.path.join( dirname, 'tmp_fnames')
    os.mkdir(rename_dir)
    frame_fmt = 'frame%07d.png'
    for i,fname in enumerate(files):
        os.symlink( os.path.join(dirname,fname),
                    os.path.join(rename_dir,frame_fmt%(i+1,)) )

    im = Image.open( os.path.join(dirname,fname) )
    (width,height) = im.size

    cmd = 'ffmpeg2theora -v 8 %s --inputfps 3 -o %s'%( frame_fmt, os.path.join(dirname,out_fname) )
    subprocess.check_call( cmd, shell=True, cwd=rename_dir)

    shutil.rmtree( rename_dir )
    return (width,height)

def analysis_type_factory( db, class_name ):
    klass = globals()[class_name]
    atype = klass(db)
    return atype

class_names = ['EKF_based_3D_position','PlotSummary3D','ImageBased2DOrientation']
