from django.template import RequestContext, loader
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django import forms
from django.core.urlresolvers import reverse

from couchdb.client import Server

couchbase = settings.FWA_COUCH_BASE_URI
couch_server = Server(couchbase)
metadb = couch_server['flydraweb_metadata']

class DatabaseForm(forms.Form):
    database = forms.CharField()

class DatasetForm(forms.Form):
    dataset = forms.CharField()

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

    if len(valid_db_names)==1:
        # no need to choose manually, automatically fast-forward
        next = get_next_url(db_name=valid_db_names[0])
        return HttpResponseRedirect(next)

    if request.method == 'POST': # If the form has been submitted...
        form = DatabaseForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            next = get_next_url(db_name=form.cleaned_data['database'])
            return HttpResponseRedirect(next)
    else:
        # GET
        choices = [ (db_name,db_name) for db_name in valid_db_names ]
        form = DatabaseForm()
        form.fields['database'].widget = forms.Select(choices=choices)

    t = loader.get_template('select.html')
    c = RequestContext(request, {"form":form,"what":"database"})
    return HttpResponse(t.render(c))

@login_required
def select_dataset(request,db_name=None):
    """view a db, which means select a dataset"""
    assert db_name is not None
    assert is_access_valid(db_name,request.user)

    db = couch_server[db_name]

    view_results = db.view('analysis/datasets')
    dataset_names = []
    for row in view_results:
        dataset_id = row.value['dataset_doc']['_id']
        assert dataset_id.startswith('dataset:')
        dataset = dataset_id[8:]
        dataset_names.append(dataset)

    if len(dataset_names)==1:
        next = get_next_url(db_name=db_name,dataset_name=dataset_names[0])
        return HttpResponseRedirect(next)

    if request.method == 'POST': # If the form has been submitted...
        form = DatasetForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            next = get_next_url(db_name=db_name,
                                dataset_name=form.cleaned_data['dataset'])
            return HttpResponseRedirect(next)
    else:
        # GET
        choices = [ (ds_name,ds_name) for ds_name in dataset_names ]
        form = DatasetForm()
        form.fields['dataset'].widget = forms.Select(choices=choices)

    t = loader.get_template('select.html')
    c = RequestContext(request, {"form":form,"what":"dataset"})
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
