from flydra.geom import LineSegment, ThreeTuple
import result_browser
import numpy
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
time_fmt = '%Y-%m-%d %H:%M:%S %Z%z'

class Distance(PT.IsDescription):
    frame = PT.Int32Col(pos=0,indexed=True)
    distance = PT.FloatCol(pos=1)

def save_post_distances(results,post,
                        start=None,
                        stop=None):
    
    postseg = LineSegment(ThreeTuple(post[0]),
                          ThreeTuple(post[1]))
    print 'loading data...'
    (f,X,L,err,ts)=result_browser.get_f_xyz_L_err(results,include_timestamps=True)
    print 'done'
    
    if start is not None or stop is not None:
        print 'converting timestamps...'
        pacific = pytz.timezone('US/Pacific')
        att = numpy.array( [datetime.datetime.fromtimestamp(x,pacific)
                            for x in ts] )
        print 'done'
        if start is not None:
            accept_idx = (att >= start)
        else:
            accept_idx = numpy.ones( ts.shape, dtype=numpy.bool_ )

        if stop is not None:
            accept_idx = accept_idx & (att <= stop)
    else:
        accept_idx = numpy.ones( ts.shape, dtype=numpy.bool_ )

    if hasattr( results.root,'post_distance'):
        print "removing old 'post_distance' table"
        results.removeNode( results.root.post_distance, recursive=True)
    print "creating 'post_distance' table"
    table = results.createTable(results.root,'post_distance',Distance,'distance to post')
    r = table.row
    
    for i,(f_i,x_i) in enumerate(zip(f[accept_idx],X[accept_idx])):
        if i%1000==0:
            print 'computing post distances (%d of %d)...'%(i,len(X[accept_idx]))
        pt = ThreeTuple(x_i)
        dist = postseg.get_distance_from_point(pt)
        r['frame']=f_i
        r['distance']=dist
        r.append()
    table.flush()
    
if 1:
    if 1:
        # 20060515
        filename = 'DATA20060515_190905.h5'
        post = [( 471.5, 191.2, 22.7),
                ( 479.7, 205.1, 225.2),
                
                ]
        if 1:
            fstart = 369430
            fend = 377515
        pacific = pytz.timezone('US/Pacific')
        
        good_start = datetime.datetime(2006, 5, 15, 12, 0, 0, tzinfo=pacific)
        good_stop  = datetime.datetime(2006, 5, 16,  8, 0, 0, tzinfo=pacific)
        
    results = result_browser.get_results(filename)
    save_post_distances(results,post,start=good_start,stop=good_stop)
    results.close()
