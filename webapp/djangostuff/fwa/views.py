from django.template import Context, loader
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings

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
        c = Context({'description':'The django user %s was not found in the '
                     'flydraweb_metadata database.'%request.user})
        return HttpResponse(t.render(c))

    t = loader.get_template('db.html')
    c = Context({"user":request.user,"valid_db_names":valid_db_names})
    return HttpResponse(t.render(c))
