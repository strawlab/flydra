import dateutil.parser

def get_start_stop_times( db, source ):
    doc = db[source]
    start_time_str = doc['start_time']
    stop_time_str = doc['stop_time']
    
    start = dateutil.parser.parse( start_time_str )
    stop = dateutil.parser.parse( stop_time_str )
    return start,stop
