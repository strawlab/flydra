from django.conf.urls.defaults import *

urlpatterns = patterns('fwa.views',
    (r'db/$', 'db_index'), # show list of databases, choose the database
    (r'db/(?P<db_name>\w+)/$', 'db'), # show the datasets in database, choose dataset
    (r'db/(?P<db_name>\w+)/(?P<dataset>\w+)/$', 'dataset'), # show the dataset
)
