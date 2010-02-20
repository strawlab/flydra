from django.template import RequestContext, loader
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.views.generic.simple import redirect_to

from couchdb.client import Server
import couchdb.http

couchbase = settings.FWA_COUCH_BASE_URI
couch_server = Server(couchbase)
metadb = couch_server['flydraweb_metadata']

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
def db_index(request):
    db_names = [row.key for row in metadb.view('meta/databases')]

    try:
        valid_db_names = [db_name for db_name in db_names if is_access_valid(db_name,request.user)]
    except couchdb.http.ResourceNotFound,err:
        t = loader.get_template('error.html')
        c = RequestContext(request,
                           {'description':'The django user %s was not found in the '
                            'flydraweb_metadata database.'%request.user})
        return HttpResponse(t.render(c))

    if len(valid_db_names)==1:
        # no need to choose manually, automatically fast-forward
        return redirect_to(request,valid_db_names[0],permanent=False)

    t = loader.get_template('db_index.html')
    c = RequestContext(request, {"valid_db_names":valid_db_names})
    return HttpResponse(t.render(c))

@login_required
def db(request,db_name=None):
    """view a db, which means select a dataset"""
    assert db_name is not None
    assert is_access_valid(db_name,request.user)

    couch_server = Server(couchbase)
    db = couch_server[db_name]

    view_results = db.view('fw/datasets')
    #raise str( type(view_result) ) + ' ' + str(view_result)

    if len(view_results)==1:
        # only one dataset, choose it
        row = view_results.rows[0]

        dataset_id = row.id
        #raise dataset_id
        return redirect_to(request,dataset_id,permanent=False)
        #return redirect_to(request,'noexist',permanent=False)
    1/0

    t = loader.get_template('db.html')
    c = RequestContext(request)
    return HttpResponse(t.render(c))

@login_required
def dataset(request,db_name=None,dataset=None):
    assert db_name is not None
    assert dataset is not None
    assert is_access_valid(db_name,request.user)

    t = loader.get_template('dataset.html')
    c = RequestContext(request)
    return HttpResponse(t.render(c))

