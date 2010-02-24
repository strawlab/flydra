#!/usr/bin/env python
import couchdb.client
import pprint
import dateutil.parser
import datetime
from uuid import uuid4

class NoTimeOverlapError(ValueError):
    pass

class TimeRange:
    def __init__(self,start,stop):
        self.start = start
        self.stop = stop
    def intersect(self,other):
        assert isinstance(other,TimeRange)
        start = max( self.start, other.start )
        stop = min( self.stop, other.stop )
        return TimeRange(start,stop)
    def get_duration(self):
        return self.stop-self.start
    duration = property( get_duration )

def decode_time(time_string):
    return dateutil.parser.parse(time_string)

def encode_time(timeval):
    return timeval.isoformat()

def decode_time_dict(doc):
    for k,v in doc.iteritems():
        if k.endswith('_time'):
            if v is not None:
                newv = decode_time(v)
                doc[k] = newv
        elif isinstance(v,dict):
            decode_time_dict(v)

def string_start_intersection( names ):
    a = [ x for x in names[0] ]
    for name in names[1:]:
        for i in range(len(a)):
            if name[i] != a[i]:
                a=a[:i]
                break
    result = ''.join(a)
    return result

def test_string_start_intersection():
    a = 'abc123'
    b = 'abcdef'
    c = 'abcdadfa'
    d = 'abcdefadfs'
    assert string_start_intersection( [a,b,c,d] ) == 'abc'
