from django.template import RequestContext, loader, defaultfilters
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django import forms
from django.core.urlresolvers import reverse
from django.contrib.humanize.templatetags.humanize import intcomma

from couchdb.client import Server
import pystache

couchbase = settings.FWA_COUCH_BASE_URI
couch_server = Server(couchbase)
metadb = couch_server['flydraweb_metadata']

class DatabaseForm(forms.Form):
    database = forms.CharField()

class DatasetForm(forms.Form):
    dataset = forms.CharField()

REDIRECT_OVER_SINGLE_OPTIONS=False

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

    if REDIRECT_OVER_SINGLE_OPTIONS and len(dataset_names)==1:
        next = get_next_url(db_name=db_name,dataset_name=dataset_names[0])
        return HttpResponseRedirect(datasets[0]['path'])

    source, origin = loader.find_template_source('pystache_datasets.html') # abuse django.template to find pystache template
    contents = pystache.render( source, {'datasets':datasets} )

    t = loader.get_template('pystache_wrapper.html')
    c = RequestContext(request, {"pystache_contents":contents} )
    return HttpResponse(t.render(c))

@login_required
def dataset(request,db_name=None,dataset=None):
    assert db_name is not None
    assert dataset is not None
    assert is_access_valid(db_name,request.user)

    t = loader.get_template('dataset.html')
    c = RequestContext(request,{'dataset':dataset})
    return HttpResponse(t.render(c))

approot = reverse(select_db)
def get_next_url(db_name=None,dataset_name=None):
    if db_name is None:
        assert dataset_name is None
        return approot
    else:
        if dataset_name is None:
            return approot + db_name + '/'
        else:
            return approot + db_name + '/' + dataset_name + '/'
