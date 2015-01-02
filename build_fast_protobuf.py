import os
import subprocess
import tempfile
import shutil
import hashlib

have_fastpb = False
try:
    import fastpb.generator # from https://github.com/Cue/fast-python-pb ,
    # required for subprocess's call to protoc to
    # be successful
    have_fastpb = True
except ImportError:
    pass

from distutils.core import Extension # actually monkey-patched by setuptools

def ensure_directory( dirname ):
    # equivalent of "mkdir -p dirname"
    try:
        os.mkdir( dirname )
    except OSError as err:
        if err.errno == 17:
            # directory already exists
            pass

def checksum( filename ):
    return hashlib.sha1( open(filename).read() ).hexdigest()

def move_if_different( src_file, dest_file ):
    do_move = True
    if os.path.exists(dest_file):
        if checksum(src_file) == checksum(dest_file):
            do_move = False
        else:
            os.remove(dest_file)
    if do_move:
        os.rename( src_file, dest_file )

def _emit_sources( intermediate_source_directory, proto_path, proto_filename ):

    tmpdir = tempfile.mkdtemp()

    try:
        # wrapper to compile .proto file to .cc files.
        args = ['/usr/bin/protoc',
                '--fastpython_out='+tmpdir,
                '--cpp_out='+tmpdir,
                '--proto_path='+proto_path,
                proto_filename]
        subprocess.check_call( args )
        generated_cpp_files = [ x for x in os.listdir( tmpdir ) if x.endswith('.cc') or x.endswith('.h') ]
        for x in generated_cpp_files:
            move_if_different( os.path.join( tmpdir, x ), os.path.join( intermediate_source_directory, x ) )

    finally:
        shutil.rmtree( tmpdir )

def make_fast_protobuf_extension(name,sources,intermediate_source_directory=None,**kwargs):
    # entry point
    if intermediate_source_directory is None:
        intermediate_source_directory = 'protobuf_sources'
    if len(sources) != 1:
        raise NotImplementedError('only a single source file is currently supported')

    ensure_directory( intermediate_source_directory )

    protofile = sources[0]
    assert protofile.endswith('.proto')
    proto_path, proto_filename = os.path.split( protofile )
    proto_basename = os.path.splitext( proto_filename )[0]

    sources = [ os.path.join( intermediate_source_directory, x ) for x in
                (proto_basename+'.cc', proto_basename+'.pb.cc') ]

    if have_fastpb:
        _emit_sources( intermediate_source_directory, proto_path, protofile )
    else:
        for source_filename in sources:
            if not os.path.exists(source_filename):
                raise RuntimeError('You do not have fastpb installed '
                                   'or the pre-built source files. '
                                   'Please install fastpb from '
                                   'https://github.com/Cue/fast-python-pb')

    result = Extension(name="camera_feature_point_proto",
                       sources=sources,
                       **kwargs)
    return result
