from django.template import RequestContext, loader, defaultfilters
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.urlresolvers import reverse
from django.utils import simplejson
from django.contrib.humanize.templatetags.humanize import intcomma
from django.core.paginator import EmptyPage, PageNotAnInteger
from django import forms

import os

import datetime
import dateutil.parser

import couchdb.http
from couchdb.client import Server

from paginatoe import SimpleCouchPaginator, CouchPaginator
import flydra.a3.analysis_types
import analysis_types_fwa
from analysis_types_fwa import fwa_id_escape
import cluster

import pystache
import pprint
import urllib2

# Use simplejson or Python 2.6 json, prefer simplejson.
try:
    import simplejson as json
except ImportError:
    import json

def _connect_couch():
    couchbase = settings.FWA_COUCH_BASE_URI
    return Server(couchbase)
couch_server = _connect_couch()
metadb = couch_server['flydraweb_metadata']

REDIRECT_OVER_SINGLE_OPTIONS=True

# helper for below
def is_access_valid(db_name,user):
    # couch permissions:
    user_id = 'user:%s'%user

    user_doc = metadb[user_id]

    db_id = 'database:%s'%db_name
    db_doc = metadb[db_id]

    valid_groups = list(set(user_doc['groups']).intersection( set( db_doc['groups'])))
    if len(valid_groups):
        return True
    else:
        return False

@login_required
def select_db(request):
    # use membership in group "couchdb_DBNAME" to allow access to CouchDB database DBNAME
    valid_db_names = []
    for group in request.user.groups.all():
        db_name = group.name
        if db_name.startswith('couchdb_'):
            valid_db_names.append( db_name[8:] )

    if REDIRECT_OVER_SINGLE_OPTIONS and len(valid_db_names)==1:
        # no need to choose manually, automatically fast-forward
        next = get_next_url(db_name=valid_db_names[0])
        return HttpResponseRedirect(next)

    selections = []
    for db_name in valid_db_names:
        selections.append( {'path':get_next_url(db_name=db_name),
                           'name':db_name,
                           })
    source, origin = loader.find_template_source('pystache_select.html') # abuse django.template to find pystache template
    contents = pystache.render( source, {'what':'database','selections':selections} )

    t = loader.get_template('pystache_wrapper.html')
    c = RequestContext(request, {"pystache_contents":contents} )
    return HttpResponse(t.render(c))

@login_required
def select_dataset(request,db_name=None):
    """view a db, which means select a dataset"""
    assert db_name is not None
    assert is_access_valid(db_name,request.user)

    db = couch_server[db_name]

    view_results = db.view('analysis/datasets')
    datasets = []
    fsf = defaultfilters.filesizeformat
    for row in view_results:
        doc = row.value['dataset_doc']
        dataset_id = doc['_id']
        assert dataset_id.startswith('dataset:')
        dataset = dataset_id[8:]
        datasets.append( dict( path=get_next_url(db_name=db_name,dataset_name=dataset),
                               dataset_id=dataset_id,
                               dataset_name=doc['name'],
                               ufmf_bytes_human=fsf(row.value['ufmf_bytes']),
                               h5_bytes_human=fsf(row.value['h5_bytes']),
                               ufmf_files=intcomma(row.value['ufmf_files']),
                               h5_files=intcomma(row.value['h5_files']),
                               ))

    if REDIRECT_OVER_SINGLE_OPTIONS and len(datasets)==1:
        next = get_next_url(db_name=db_name,dataset_name=datasets[0]['dataset_name'])
        return HttpResponseRedirect(datasets[0]['path'])

    source, origin = loader.find_template_source('pystache_datasets.html') # abuse django.template to find pystache template
    contents = pystache.render( source, {'db_name':db_name,'datasets':datasets} )

    t = loader.get_template('pystache_wrapper.html')
    c = RequestContext(request, {"pystache_contents":contents} )
    return HttpResponse(t.render(c))

@login_required
def dataset(request,db_name=None,dataset=None):
    assert db_name is not None
    assert dataset is not None
    assert is_access_valid(db_name,request.user)

    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]
    dataset_doc = db[dataset_id]
    dataset_view = db.view('analysis/datasets')
    dataset_reduction = [row for row in dataset_view][0].value # Hack: get first (and only) row

    summary_view = db.view('analysis/DataNode',
                           startkey=[dataset_id],
                           endkey=[dataset_id,{}],
                           )
    all_datanodes_value = [row for row in summary_view][0].value # Hack: get first (and only) row
    datanodes_count = all_datanodes_value['n_built']+all_datanodes_value['n_unbuilt']
    datanodes_view = db.view('analysis/DataNode',
                             startkey=[dataset_id],
                             endkey=[dataset_id,{}],
                             group_level=2,
                             )
    datanodes=[]
    for row in datanodes_view:
        properties = []
        for i,property_name in enumerate(row.key[1]):
            properties.append( {'path':get_next_url(db_name=db_name,
                                                    dataset_name=dataset,
                                                    datanode_property=property_name),
                                'name':property_name,
                                'hackspace':' ', # Hack: don't trim our whitespace
                                } )
        node_dict =  {'count':intcomma(row.value['n_built']+row.value['n_unbuilt']),
                      'properties':properties,
                      }
        node_dict.update(row.value)
        datanodes.append(node_dict)

    atypes = [ {'name':getattr(flydra.a3.analysis_types,class_name).name,
                'short_description':getattr(flydra.a3.analysis_types,class_name).short_description,
                'path':get_next_url(db_name=db_name,
                                    dataset_name=dataset,
                                    analysis_type=class_name),
                } for class_name in flydra.a3.analysis_types.class_names ]

    t = loader.get_template('dataset.html')
    c = RequestContext(request, {#'dataset':dataset_doc['name'],
                                 'num_data_nodes':intcomma(datanodes_count),
                                 'datanodes':datanodes,
                                 'dataset':dataset_reduction,

                                 'num_analysis_types' : len(atypes),
                                 'analysis_types' : atypes,

                                 } )
    return HttpResponse(t.render(c))


def _get_items_paginated(db,viewname,page_number=1,count=50,**options):
    # modified from http://www.djangosnippets.org/snippets/1209/
    myitems = db.view(viewname,
                      reduce=False,**options)

    pages_view = db.view(viewname,
                         reduce=True,**options)

    paginate = CouchPaginator(myitems, count, pages_view=pages_view)
    page = paginate.page(page_number)

    return page

@login_required
def datanodes_by_property(request,db_name=None,dataset=None,property_name=None,count=50):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    if property_name is None:
        startkey=[dataset_id]
        endkey=[dataset_id,{}]
        show_property_name='all'
    else:
        startkey=[dataset_id,property_name]
        endkey=[dataset_id,property_name,{}]
        show_property_name=property_name
    options = dict(startkey=startkey,endkey=endkey)

    page_number = request.GET.get('page', 1)
    page = _get_items_paginated(db,'analysis/datanodes-by-dataset-and-property',
                                page_number=page_number,
                                count=count,**options)

    context = {
        'items': page,
        'property_name':show_property_name,
        'doc_url': get_next_url(db_name=db_name,doc_base=True),
    }

    t = loader.get_template('datanodes.html')
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))


class JsonHttpResponse(HttpResponse):
    def __init__(self, data):
        content = simplejson.dumps(data,
                                   indent=2,
                                   ensure_ascii=False)
        super(JsonHttpResponse, self).__init__(content=content,
                                           mimetype='application/json; charset=utf8')

@login_required
def submit_SGE_jobs(request,db_name=None,dataset=None):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    analysis_types_fwa.insert_jobs_into_sge(db,
                                            os.path.join(settings.FWA_STARCLUSTER_CONFIG_DIR,db_name) )
    url = get_next_url(db_name=db_name,dataset_name=dataset)
    return HttpResponseRedirect(url) # Redirect after POST

@login_required
def view_SGE_jobs(request,db_name=None,dataset=None):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    cluster_obj = cluster.StarCluster(os.path.join(settings.FWA_STARCLUSTER_CONFIG_DIR,db_name))

    cluster_obj.update_couch_view_of_sge_status(settings.FWA_COUCH_BASE_URI, db.name)

    sge_status = db['sge_status']

    page_number = request.GET.get('page', 1)
    try:
        items = _get_items_paginated(db,'analysis/list_pending_jobs',
                                     page_number=page_number,
                                     )
    except EmptyPage:
        items = None

    context = {
        'items': items,
        'doc_url': get_next_url(db_name=db_name,doc_base=True),
        'sge_status':pprint.pformat(dict(sge_status)),
        }

    t = loader.get_template('view_SGE_jobs.html')
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))

class ClusterStartForm(forms.Form):
    n_nodes = forms.IntegerField(initial=1,min_value=1)

@login_required
def cluster_admin(request,db_name=None,dataset=None):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    cluster_obj = cluster.StarCluster(os.path.join(settings.FWA_STARCLUSTER_CONFIG_DIR,db_name))
    is_running = cluster_obj.is_running()

    if request.method == 'POST': # If the form has been submitted...
        form = ClusterStartForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            # start cluster...
            if not is_running:
                cluster_obj.start_n_nodes( form.cleaned_data['n_nodes'] )
            else:
                cluster_obj.modify_num_nodes( form.cleaned_data['n_nodes'] )
            return HttpResponseRedirect('..') # Redirect after POST
    else:
        form = ClusterStartForm() # An unbound form

    context = {'name':cluster_obj.name,
               'is_running':is_running,
               }

    context['is_ec2'] = isinstance(cluster_obj,cluster.StarCluster)
    context['form'] = form

    if is_running:
        context['num_nodes'] = cluster_obj.get_num_nodes()
        context['stop_cluster_url'] = '../cluster_stop'

    t = loader.get_template('cluster_admin.html')
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))

@login_required
def cluster_stop(request,db_name=None,dataset=None):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    cluster_obj = cluster.StarCluster(os.path.join(settings.FWA_STARCLUSTER_CONFIG_DIR,db_name))

    if cluster_obj.is_running():
        cluster_obj.shutdown()
    return HttpResponseRedirect('../cluster_admin')

@login_required
def apply_analysis_type(request,db_name=None,dataset=None,class_name=None):
    dataset_id = 'dataset:'+dataset
    db = couch_server[db_name]

    klass = getattr(flydra.a3.analysis_types,class_name)
    analysis_type = klass(db=db)

    error_message = None
    success_message = None
    if request.method == 'POST':
        # since we dynamically generated the form, we must manually ensure the POST is valid

        verifier = analysis_types_fwa.Verifier( db, dataset, analysis_type )
        try:
            new_batch_jobs = verifier.validate_new_batch_jobs_request( request.POST )

            success_message = analysis_types_fwa.upload_job_docs_to_couchdb(db, new_batch_jobs,
                                                                            os.path.join(settings.FWA_STARCLUSTER_CONFIG_DIR,db_name) )

            ## # new_batch_jobs are valid, insert them and thank user
            ## #db.append( new_datanode_documents )
            ## 1/0
            #return HttpResponseRedirect('/thanks/') # Redirect after POST

        except analysis_types_fwa.InvalidRequest, err:
            error_message = err.human_description

    # dynamically generate a form to display
    form = forms.Form()
    js_client_info = {}

    def myappend(list1,elemn):
        list1.append(elemn)
        return list1

    for source_node_type in analysis_type.source_node_types:
        datanodes_view = db.view('analysis/datanodes-by-dataset-and-property',
                                 startkey=[dataset_id,source_node_type],
                                 endkey=[dataset_id,source_node_type,{}],
                                 reduce=False)
        n_docs = len(datanodes_view)
        form.fields[ fwa_id_escape(source_node_type) ] = forms.ChoiceField(choices=[(row.id,row.id) for row in datanodes_view ],
                                                                       widget=forms.SelectMultiple())
        rows = [ dict( myappend( row['value'].items(), ('id',row['id']))) for row in datanodes_view ]
        js_client_info[ source_node_type ] = {'rows':rows,
                                              'select_id':('id_'+fwa_id_escape(source_node_type)), # django seems to do this.
                                              }
        analysis_types_fwa.add_fields_to_form( form, analysis_type )

    final_js_info = {'sources':js_client_info}
    if hasattr(analysis_type,'dominant_source_node_type'):
        final_js_info['dominant_source_node_type'] = analysis_type.dominant_source_node_type

    context = {
        'name':analysis_type.name,
        'js_client_info':json.dumps(final_js_info),
        'form':form,
        'error_message':error_message,
        'success_message':success_message,
        }

    t = loader.get_template('apply_analysis_type.html')
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))

class NotDataNode(ValueError):
    pass

def get_job_id_for_datanode( db, doc_id ):
    job_id_view = db.view('analysis/jobs-by-datanode-id',
                           startkey=doc_id,
                           endkey=doc_id,
                           )
    job_id_rows = [row for row in job_id_view] # Hack: get first (and only) row
    if len(job_id_rows):
        assert len(job_id_rows)==1
        job_id = job_id_rows[0].id
    else:
        job_id = None
    return job_id

@login_required
def datanode(request,db_name=None,doc_id=None,warn_no_specific_view=False):
    db = couch_server[db_name]
    doc = db[doc_id]
    # get datanode view of doc
    myitems = db.view('analysis/datanodes-by-docid',
                      startkey=doc_id,
                      endkey=doc_id,
                      )
    if len(myitems)==0:
        raise NotDataNode('doc_id %s is not a datanode'%doc_id)
    if len(myitems)!=1:
        raise ValueError( 'for doc_id=%s, len(myitems)==%d'%(doc_id,len(myitems)))
    for row in myitems:
        pass # XXX hack to get item

    doc = db[row.id]
    attachments = doc.get('_attachments',{})
    images = []
    for fname, attachment in attachments.iteritems():
        if attachment['content_type']=='image/png':
            width, height = doc['imsize'][fname]
            images.append( {'url':get_attachment_url( db_name=db_name, doc_id=row.id, fname=fname ),
                            'width':width,
                            'height':height,
                            'fname':fname,
                            } )

    start_time = dateutil.parser.parse(doc['start_time'])
    stop_time = dateutil.parser.parse(doc['stop_time'])
    time_range = '%s - %s, %s'%( start_time.strftime('%H:%M:%S'),
                                 stop_time.strftime('%H:%M:%S'),
                                 start_time.strftime('%A, %d %B %Y (UTC %z)' ) )

    saved_images = doc.get('saved_images',{})
    si = [ {'url':k,'width':v[0],'height':v[1]} for (k,v) in saved_images.iteritems() ]


    # get job doc
    job_id = get_job_id_for_datanode( db, doc_id )

    # get children
    children_ids = get_datanode_direct_children_ids(db, doc)

    t = loader.get_template('datanode.html')
    context = {'row':row,
               'doc_url': get_next_url(db_name=db_name,doc_base=True),
               'warn_no_specific_view':warn_no_specific_view,
               'raw_value':pprint.pformat(row.value),
               'images':images,
               'saved_images': si,
               'time_range':time_range,
               'job_id':job_id,
               'delete_doc_url': get_next_url(db_name=db_name,delete_doc_base=True),
               'id':doc_id,
               'children_ids':children_ids,
               }
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))

@login_required
def raw_doc(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    doc = db[doc_id]
    t = loader.get_template('raw_doc.html')
    c = RequestContext(request,{'id':doc['_id'],'raw':pprint.pformat(dict(doc)),
                                })
    return HttpResponse(t.render(c))

@login_required
def job_doc(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    doc = db[doc_id]
    t = loader.get_template('job_doc.html')
    c = RequestContext(request,{'id':doc['_id'],
                                'raw':pprint.pformat(dict(doc)),
                                'datanode_id':doc['datanode_id'],
                                'doc_url': get_next_url(db_name=db_name,doc_base=True),
                                'delete_doc_url': get_next_url(db_name=db_name,delete_doc_base=True),
                                })
    return HttpResponse(t.render(c))

def delete_job( db, doc, also_delete_target=True ):
    assert doc['type']=='job'
    #assert doc['state'] != 'submitted' # already in SGE, can't stop it this way...
    #assert doc['state'] != 'executing' # already in SGE, can't stop it this way...
    if also_delete_target:
        # get the target id
        target_id = doc['datanode_id']
        target_doc = db[target_id]
        delete_datanode(db, target_doc, also_delete_job=False )
    db.delete(doc)

def get_datanode_direct_children_ids(db, doc):
    doc_id = doc['_id']
    children_view = db.view('analysis/datanode-children',
                            startkey=doc_id,
                            endkey=doc_id,
                            )
    children_ids = []
    for row in children_view:
        children_ids.append( row.id )
    return children_ids

def delete_datanode( db, doc, also_delete_job=True ):

    if doc['type']=='h5':
        if 'filename' in doc:
            filenames = [ doc['filename'] ]
        else:
            filenames = []
    else:
        assert doc['type']=='datanode'
        filenames = doc.get('filenames',[])

    child_list = get_datanode_direct_children_ids(db, doc)
    if len(child_list):
        raise RuntimeError('Attempting to delete datanode which is a source for other datanodes')

    if len(filenames):
        delete_filenames(filenames)

    if also_delete_job:
        # get the job id (if it exists)
        job_id = get_job_id_for_datanode( db, doc['_id'] )
        if job_id is not None:
            job_doc = db[job_id]
            delete_job(db, job_doc, also_delete_target=False )
    db.delete(doc)

def delete_filenames(filenames):
    raise NotImplementedError('file deletion not yet implemented')

@login_required
def delete_document_multiplexer(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound, err:
        raise Http404("In database '%s', there is no document '%s'." % (db_name,doc_id))
    if doc['type']=='datanode':
        delete_datanode( db, doc )
    elif doc['type']=='h5':
        delete_datanode( db, doc )
    elif doc['type']=='job':
        delete_job( db, doc )
    else:
        raise NotImplementedError('unknown document to delete')
    t = loader.get_template('deleted_doc.html')
    c = RequestContext(request,{'id':doc['_id']})
    return HttpResponse(t.render(c))

@login_required
def document_multiplexer(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound, err:
        raise Http404("In database '%s', there is no document '%s'." % (db_name,doc_id))
    if doc['type']=='datanode':
        return datanode(request,db_name=db_name,doc_id=doc_id)
    elif doc['type']=='h5':
        return datanode(request,db_name=db_name,doc_id=doc_id)
    elif doc['type']=='job':
        return job_doc(request,db_name=db_name,doc_id=doc_id)
    else:
        try:
            return datanode(request,db_name=db_name,doc_id=doc_id,warn_no_specific_view=True)
        except NotDataNode,err:
            return raw_doc(request,db_name=db_name,doc_id=doc_id)

@login_required
def serve_attachment(request,db_name=None,doc_id=None,fname=None):
    db = couch_server[db_name]
    doc = db[doc_id]
    attachment = db.get_attachment(doc,fname)
    buf = attachment.read()
    content_type = doc['_attachments'][fname]['content_type']

    result = HttpResponse(content=buf,
                          content_type=content_type,
                          )
    return result

approot = reverse(select_db)
def get_next_url(db_name=None,
                 dataset_name=None,
                 datanode_property=None,
                 analysis_type=None,
                 doc_base=False,
                 delete_doc_base=False,
                 ):

    # doc_base and delete_doc_base are mutually exclusive
    if doc_base:
        assert not delete_doc_base
    if delete_doc_base:
        assert not doc_base

    if db_name is None:
        assert dataset_name is None
        return approot
    else:
        if dataset_name is None:
            if doc_base:
                return approot + db_name + '/doc/'
            else:
                if delete_doc_base:
                    return approot + db_name + '/delete_doc/'
                else:
                    return approot + db_name + '/'
        else:
            assert not doc_base
            assert not delete_doc_base
            if datanode_property is None:
                if analysis_type is not None:
                    return ( approot + db_name + '/' + dataset_name +
                             '/' + 'apply_analysis_type/' +
                             defaultfilters.iriencode(analysis_type) + '/' )
                else:
                    return approot + db_name + '/' + dataset_name + '/'
            else:
                assert analysis_type is None, "only one can be non-None"
                return approot + db_name + '/' + dataset_name + '/' + 'DataNodes/' + defaultfilters.iriencode(datanode_property) + '/'

def get_attachment_url(db_name=None,
                  doc_id=None,
                  fname=None):
    return approot + db_name + '/attachment/' + doc_id +'/'+fname