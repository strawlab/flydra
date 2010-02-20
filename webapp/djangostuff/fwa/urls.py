from django.conf.urls.defaults import *

urlpatterns = patterns('fwa.views',
    (r'db/$', 'select_db'), # show list of databases, choose the database
    (r'db/(?P<db_name>\w+)/$', 'select_dataset'), # show the datasets in database, choose dataset
    (r'db/(?P<db_name>\w+)/(?P<dataset>\w+)/$', 'dataset'), # show the dataset
)
