import tables
import numpy
import sys

# derived from check_clock_sync.py

def get_worst_sync_dict(filename_or_h5_file):
    """return dict with hostnames for keys and times (in seconds) for values"""
    if isinstance(filename_or_h5_file,tables.File):
        kresults = filename_or_h5_file
        opened = False
    else:
        kresults = tables.openFile(filename_or_h5_file,mode="r")
        opened = True

    hci = kresults.root.host_clock_info
    tbl = hci[:]
    cam_info = kresults.root.cam_info[:]

    if opened:
        kresults.close()

    hostnames = tbl['remote_hostname']
    uhostnames = numpy.unique(hostnames)
    result = {}

    for hostname in uhostnames:
        cond = hostnames == hostname
        start = tbl['start_timestamp'][cond]
        stop = tbl['stop_timestamp'][cond]
        remote = tbl['remote_timestamp'][cond]
        max_measurement_error = stop-start
        max_clock_diff = remote-start

        sortedidxs = max_clock_diff.argsort()
        earliest = max_clock_diff[sortedidxs[0]]
        latest = max_clock_diff[sortedidxs[-1]]

        if earliest < -latest:
            worst = earliest
            idx = sortedidxs[0]
        else:
            worst = latest
            idx = sortedidxs[-1]

        meas_err_at_worst = max_measurement_error[idx]
        result[hostname]=worst

    # special case: a single host called localhost in one place and real hostname in another
    if len(uhostnames)==1 and uhostnames[0]=='localhost':
        hostnames2 = numpy.unique(cam_info['hostname'])
        if len(hostnames2)==1:
            result[hostnames2[0]]=result['localhost']
    return result

def main():
    filename = sys.argv[1]
    result_dict = get_worst_sync_dict(filename)
    hostnames = result_dict.keys()
    hostnames.sort()

    for hostname in hostnames:
        print 'worst clock sync for %s: %.1f msec'%(hostname,
                                                    result_dict[hostname]*1000.0)

if __name__=='__main__':
    main()
