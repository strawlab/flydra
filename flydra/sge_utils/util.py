COUCHDB_STATUS_DOC_ID = 'sge_status'

def get_SGE_job_name_from_couch_job_id(job_id):
    return 'job%s'%job_id
