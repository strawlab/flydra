from django.conf.urls.defaults import *

urlpatterns = patterns('fwa.views',
    (r'^$', 'select_db'), # show list of databases, choose the database
    (r'^(?P<db_name>\w+)/$', 'select_dataset'), # show the datasets in database, choose dataset
    (r'^(?P<db_name>\w+)/(?P<dataset>\w+)/$', 'dataset'), # show the dataset
    (r'^(?P<db_name>\w+)/(?P<dataset>\w+)/DataNodes/(?P<property_name>[\w ]+)/$', 'datanode_property'), # show the dataset
)
