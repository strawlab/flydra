from django.conf.urls.defaults import *

urlpatterns = patterns('fwa.views',
    (r'db/', 'db_index'),
)
