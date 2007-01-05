import sys, sets
import tables
import pylab
import numpy

def drift_estimates(results):
    table = results.root.host_clock_info
    remote_hostnames = table.read(field='remote_hostname',flavor='numpy')
    hostnames = [str(x).strip() for x in list(sets.Set(remote_hostnames))]
    hostnames.sort()
    
    del remote_hostnames

    result = {}
    
    for hostname in hostnames:
        row_idx = table.getWhereList(table.cols.remote_hostname == hostname,flavor='numpy')
        start_timestamp = table.readCoordinates(row_idx,field='start_timestamp',flavor='numpy')
        stop_timestamp = table.readCoordinates(row_idx,field='stop_timestamp',flavor='numpy')
        remote_timestamp = table.readCoordinates(row_idx,field='remote_timestamp',flavor='numpy')

        measurement_error = stop_timestamp-start_timestamp
        clock_diff = stop_timestamp-remote_timestamp

        # local time when we think remote timestamp was gathered, given symmetric transmission delays
        local_timestamp = start_timestamp + measurement_error*0.5
        
        result.setdefault('hostnames',[]).append(hostname)
        result.setdefault('local_timestamp',{})[hostname] = local_timestamp
        result.setdefault('remote_timestamp',{})[hostname] = remote_timestamp
        result.setdefault('measurement_error',{})[hostname] = measurement_error
                          
    return result

def model_remote_to_local(remote_timestamps, local_timestamps):
    a1=remote_timestamps[:,numpy.newaxis]
    a2=numpy.ones( (len(remote_timestamps),1))
    A = numpy.hstack(( a1,a2))
    #A = numpy.hstack(( remote_timestamps[:,numpy.newaxis], numpy.ones( (len(remote_timestamps),1))))
    b = local_timestamps[:,numpy.newaxis]
    x,resids,rank,s = numpy.linalg.lstsq(A,b)
    gain = x[0,0]
    offset = x[1,0]
    return gain,offset
        
def main():
    fname = sys.argv[1]
    results = tables.openFile(fname,mode='r')
    d = drift_estimates(results)
    hostnames = d['hostnames']
    gain = {}; offset = {};
    for i,hostname in enumerate(hostnames):
        tgain, toffset = model_remote_to_local(d['remote_timestamp'][hostname][::10],
                                               d['local_timestamp'][hostname][::10])
        gain[hostname]=tgain
        offset[hostname]=toffset
        print repr(hostname),tgain,toffset
        
    print
    
    table = results.root.data2d_distorted

    if 1:
        import result_utils
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
        camn2hostname = {}
        for camn, cam_id in camn2cam_id.iteritems():
            hostname = '_'.join(cam_id.split('_')[:-1])
            camn2hostname[camn]=hostname

    cur_frame = None
    cur_ts = []
##    start_timestamp = None
    for row in table:
        camn = row['camn']
        hostname = camn2hostname[camn]
        remote_timestamp = row['timestamp']
        local_timestamp = remote_timestamp*gain[hostname]+offset[hostname]
##        if start_timestamp is None:
##            start_timestamp=local_timestamp
        frame = row['frame']

##        print frame,local_timestamp-start_timestamp
##        print
        
        if frame==cur_frame:
            cur_ts.append(local_timestamp)
        else:
            if len(cur_ts)>2:
                # print last frame
                cur_ts = numpy.array(cur_ts)
                mn = cur_ts.min()
                mx = cur_ts.max()
                spread = mx-mn
                spread_msec = spread*1e3
                print '% 9d % 6.2f'%(cur_frame,spread_msec)

            # reset for current frame
            cur_ts = [local_timestamp]
            cur_frame = frame
            
    
def main_old():
    fname = sys.argv[1]
    results = tables.openFile(fname,mode='r')
    d = drift_estimates(results)
    hostnames = d['hostnames']
    for i,hostname in enumerate(hostnames):
        gain, offset = model_remote_to_local(d['remote_timestamp'][hostname][::10],
                                             d['local_timestamp'][hostname][::10])
    
    if 1:
        ax=None
        for i,hostname in enumerate(hostnames):
            ax=pylab.subplot(len(hostnames),1,i+1,sharex=ax)

            clock_diff = d['local_timestamp'][hostname]-d['remote_timestamp'][hostname]

            x = d['local_timestamp'][hostname][::10]
            x=x-x[0]
            y = clock_diff[::10]
            yerr = d['measurement_error'][hostname][::10]

            sys.stdout.flush()

            #pylab.errorbar(x,y, yerr=yerr)
            pylab.plot(x,y,'k-')
            pylab.plot(x,y+yerr,'b-')
            pylab.plot(x,y-yerr,'b-')
            pylab.text(0.05, 0.05, str(hostname), transform = ax.transAxes)

        pylab.show()
    
if __name__ == '__main__':
    main()
