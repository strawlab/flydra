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

couchdb-python
--------------

Developed with 0.6.1.

pystache
--------

Developed with git clone from defunkt from Feb 27, 2010.

Pinax
-----

Developed with 0.7.1

Installation
============

fwa is designed to be installed as any other Django app. Just place it
in the `INSTALLED_APPS` setion of your `settings.py` file.

Configuration
=============

In your `settings.py` you need something like the following::

  # The couchdb location
  FWA_COUCH_BASE_URI = 'http://127.0.0.1:5984/'

Create a django group (in django admin) called 'couchdb_DBNAME' for
each CouchDB used. Make each Django user a group member to allow
access to DBNAME.
