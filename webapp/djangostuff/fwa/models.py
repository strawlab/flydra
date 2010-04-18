from django.db import models

class CouchDB_Database(models.Model):
    name     = models.CharField(max_length=200)
    user     = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
