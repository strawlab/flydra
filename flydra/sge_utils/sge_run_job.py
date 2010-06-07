import sys, os, shutil, subprocess
from couchdb.client import Server
from flydra.a3.analysis_types import analysis_type_factory
import config
from optparse import OptionParser
import flydra.sge_utils.states

def run_job(couch_url, db_name, doc_id, keep=False):
    '''called by SGE process for specific work job'''

    # connect to CouchDB server
    couch_server = Server(couch_url)
    db = couch_server[db_name]

    # get job info
    job_doc = db[doc_id]
    #if job_doc['state'] != flydra.sge_utils.states.SUBMITTED:
    #    raise RuntimeError('will not run SGE job unless state is SUBMITTED')

    atype = analysis_type_factory( db, job_doc['class_name'] )

    datanode_doc = db[job_doc['datanode_id']]

    print job_doc
    print datanode_doc
    print

    # create job-specific directory in local instance store
    tmp_dirname = os.path.join(config.instance_store_dir, doc_id)
    if os.path.exists(tmp_dirname):
        raise RuntimeError('temp dir already exists: %s'%tmp_dirname)

    os.mkdir(tmp_dirname)
    try:
        job_doc['state'] = flydra.sge_utils.states.EXECUTING
        db.update( [job_doc] ) # upload new state

        # copy source files from EBS
        source_info = atype.prepare_sources( job_doc, tmp_dirname )
        print 'copied files into',tmp_dirname
        print
        orig_files = os.listdir(tmp_dirname)

        # run job in local instance store
        cmd = ['~/PY/bin/'+atype.base_cmd] # XXX use virtualenv in ~/PY
        cmd.extend( atype.convert_sources_to_cmdline_args(job_doc,source_info) )
        cmd.extend( atype.get_cmdline_args_from_choices(job_doc,source_info) )
        print ' '.join(cmd)
        print
        sys.stdout.flush()

        cmd = ' '.join(cmd) # XXX why is this needed?
        subprocess.check_call(cmd,cwd=tmp_dirname,shell=True)
        final_files = os.listdir(tmp_dirname)

        new_files = list(set(final_files) - set(orig_files))
        print 'new_files',new_files
        print
        # copy known result files to EBS
        copied_files, datanode_doc_custom = atype.copy_outputs( job_doc, tmp_dirname, config.sink_dir )
        print 'copied_files',copied_files
        print

        lost_files = list((set(final_files) - set(orig_files)) - set(copied_files))
        print 'lost_files',lost_files
        print

        # update datanode CouchDB document
        # update job CouchDB document
        datanode_doc.update( datanode_doc_custom )
        datanode_doc['comments'] = 'command: %s'%repr(cmd)
        status_tags = datanode_doc.get('status_tags',None)
        if status_tags is not None:
            status_tags.remove('unbuilt')
            status_tags.append('built')
            datanode_doc['status_tags'] = status_tags
        job_doc['state'] = flydra.sge_utils.states.COMPLETE
        db.update( [ datanode_doc, job_doc ] )

    except Exception, err:
        errors = job_doc.get('errors',[])
        if errors is None:
            errors = []
        errors.append('Failed with error: %s'%err)
        # it's no longer in the SGE queue, so set to created state
        job_doc['state'] = flydra.sge_utils.states.CREATED
        job_doc['errors'] = errors
        db.update( [job_doc] ) # upload new state

    finally:
        if not keep:
            shutil.rmtree(tmp_dirname)

def main():
    usage = '%prog COUCH_URI DB_NAME DOC_ID [options]'
    parser = OptionParser(usage)

    parser.add_option("--keep", action='store_true',
                      help="keep the intermediate files",
                      default=False)

    (options, args) = parser.parse_args()
    if len(args)!=3:
        print >> sys.stderr, 'error: invalid number of required arguments'
        sys.exit(1)

    couch_url = args[0]
    db_name = args[1]
    doc_id = args[2]
    run_job(couch_url, db_name, doc_id, keep=options.keep)

if __name__=='__main__':
    main()
