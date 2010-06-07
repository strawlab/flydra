from django.template import RequestContext, loader, defaultfilters
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.urlresolvers import reverse
from django.utils import simplejson
from django.contrib.humanize.templatetags.humanize import intcomma
from django.core.paginator import EmptyPage, PageNotAnInteger
from django import forms

from couchdb.client import Server

import localglobal.client

from paginatoe import SimpleCouchPaginator, CouchPaginator
import flydra.a3.analysis_types
import analysis_types_fwa

import pystache
import pprint
import urllib2

def _connect_couch():
    couchbase = settings.FWA_COUCH_BASE_URI
    return Server(couchbase)
couch_server = _connect_couch()
metadb = couch_server['flydraweb_metadata']

def _connect_localglobal():
    lghost = settings.FWA_LOCALGLOBAL_HOST
    lgport = settings.FWA_LOCALGLOBAL_PORT
    return localglobal.client.Server(host=lghost, port=lgport)
localglobal_server = _connect_localglobal()

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

    # modified from http://www.djangosnippets.org/snippets/1209/
    myitems = db.view('analysis/datanodes-by-dataset-and-property',
                      reduce=False,**options)

    pages_view = db.view('analysis/datanodes-by-dataset-and-property',
                         reduce=True,**options)

    try:
        page_number = request.GET.get('page', 1)
        paginate = CouchPaginator(myitems, count, pages_view=pages_view)
        page = paginate.page(page_number)
        items = paginate.object_list
    except EmptyPage:
        raise Http404("Page %s empty" % page_number)
    except PageNotAnInteger:
        raise Http404("No page '%s'" % page_number)

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
                                            settings.FWA_STARCLUSTER_CONFIG_FNAME )
    url = get_next_url(db_name=db_name,dataset_name=dataset)
    return HttpResponseRedirect(url) # Redirect after POST

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
                                                                            settings.FWA_STARCLUSTER_CONFIG_FNAME )

            ## # new_batch_jobs are valid, insert them and thank user
            ## #db.append( new_datanode_documents )
            ## 1/0
            #return HttpResponseRedirect('/thanks/') # Redirect after POST

        except analysis_types_fwa.InvalidRequest, err:
            error_message = err.human_description

    # dynamically generate a form to display
    form = forms.Form()
    for source_node_type in analysis_type.source_node_types:
        datanodes_view = db.view('analysis/datanodes-by-dataset-and-property',
                                 startkey=[dataset_id,source_node_type],
                                 endkey=[dataset_id,source_node_type,{}],
                                 reduce=False)
        n_docs = len(datanodes_view)
        form.fields[ source_node_type ] = forms.ChoiceField(choices=[(row.id,row.id) for row in datanodes_view ],
                                                            widget=forms.SelectMultiple())
        analysis_types_fwa.add_fields_to_form( form, analysis_type )

    context = {
        'name':analysis_type.name,
        'form':form,
        'error_message':error_message,
        'success_message':success_message,
        }

    t = loader.get_template('apply_analysis_type.html')
    c = RequestContext(request,context)
    return HttpResponse(t.render(c))

class NotDataNode(ValueError):
    pass

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

    t = loader.get_template('datanode.html')
    c = RequestContext(request,{'row':row,
                                'doc_url': get_next_url(db_name=db_name,doc_base=True),
                                'warn_no_specific_view':warn_no_specific_view,
                                'raw_value':pprint.pformat(row.value),
                                'images':images,
                                })
    return HttpResponse(t.render(c))

@login_required
def raw_doc(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    doc = db[doc_id]
    t = loader.get_template('raw_doc.html')
    c = RequestContext(request,{'id':doc['_id'],'raw':pprint.pformat(dict(doc)),
                                'localglobal':get_localglobal_context( doc['sha1sum'] ),
                                })
    return HttpResponse(t.render(c))

def get_localglobal_docs(sha1sum):
    orig_lgdocs = localglobal_server.get_sha1sum_docs( sha1sum )
    extra_lgdocs = []
    for lgdoc in orig_lgdocs:
        if lgdoc['type']!='compressed':
            continue
        extra = localglobal_server.get_sha1sum_docs( lgdoc['compressed_sha1sum'], return_compressed=False )
        extra_lgdocs.extend( extra )
    lgdocs = orig_lgdocs + extra_lgdocs
    return lgdocs

def get_localglobal_context(sha1sum):
    """prepare for localglobal_doc.html template"""
    lgdocs = get_localglobal_docs( sha1sum )
    return {'docs':lgdocs,'raw':pprint.pformat(lgdocs)}

@login_required
def h5_doc(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    doc = db[doc_id]

    sha1sum=doc.get('sha1sum',None)
    if sha1sum is not None:
        lgc = get_localglobal_context( sha1sum )
    else:
        lgc = None

    t = loader.get_template('h5_doc.html')
    c = RequestContext(request,{'id':doc['_id'],'raw':pprint.pformat(dict(doc)),
                                'localglobal': lgc,
                                })
    return HttpResponse(t.render(c))

@login_required
def document_multiplexer(request,db_name=None,doc_id=None):
    db = couch_server[db_name]
    doc = db[doc_id]
    if doc['type']=='datanode':
        return datanode(request,db_name=db_name,doc_id=doc_id)
    elif doc['type']=='h5':
        return h5_doc(request,db_name=db_name,doc_id=doc_id)
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
                 doc_base=False):

    if db_name is None:
        assert dataset_name is None
        return approot
    else:
        if dataset_name is None:
            if doc_base:
                return approot + db_name + '/doc/'
            else:
                return approot + db_name + '/'
        else:
            assert not doc_base
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
