fwa - Flydra Web Analysis
*************************

This is a `django <http://djangoproject.com>`_ app for doing data
analysis on Flydra data.

Requirements
============

couchdb
-------

Install couchdb. There is one special database, called
`flydraweb_metadata` and other databases are listed in that special
database.

django.contrib.auth
-------------------

Install django.contrib.auth. In your project urls.py include::

  (r'^accounts/login/$', 'django.contrib.auth.views.login')

And in your settings.py::

  MIDDLEWARE_CLASSES = (
      'django.contrib.sessions.middleware.SessionMiddleware',
      'django.contrib.auth.middleware.AuthenticationMiddleware',
  )

and::

  INSTALLED_APPS = (
      'django.contrib.auth',
      'django.contrib.contenttypes',
      'django.contrib.sessions',
      )

Installation
============

fwa is designed to be installed as any other Django app. Just place it
in the `INSTALLED_APPS` setion of your `settings.py` file.

Configuration
=============

In your `settings.py` you need something like the following::

  # The couchdb location
  FWA_COUCH_BASE_URI = 'http://127.0.0.1:5984/'

The django users authorized through django.contrib.auth must be
present in the special metadata CouchDB database with ids of the form
`"user:<username>"`.