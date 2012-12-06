import tables
import numpy as np
import sys
import get_clock_sync
import flydra.analysis.result_utils as result_utils

def main():
    debug=False
    #debug=True

    filename = sys.argv[1]
    results = tables.openFile(filename,mode='r')
    time_model=result_utils.get_time_model_from_data(results,debug=debug)
    worst_sync_dict = get_clock_sync.get_worst_sync_dict(results)
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)

    hostnames = worst_sync_dict.keys()
    hostnames.sort()

    cam_ids = cam_id2camns.keys()
    cam_ids.sort()

    camns = camn2cam_id.keys()

    # read all data
    d2d = results.root.data2d_distorted[:]
    cam_info = results.root.cam_info[:]
    results.close()

    dt = time_model.framestamp2timestamp(1)-time_model.framestamp2timestamp(0)
    fps = 1.0/dt
    print 'fps',fps
    print [repr(i) for i in time_model.framestamp2timestamp(np.array([0,1,2]))]

    if 1:
        for camn in camns:
            cam_id = camn2cam_id[camn]

            cond1 = cam_info['cam_id']==cam_id
            assert np.sum(cond1)==1
            hostname = str(cam_info[ cond1 ]['hostname'][0])

            cond = d2d['camn']==camn
            mydata = d2d[cond]

            frame = mydata['frame']

            trigger_timestamp = time_model.framestamp2timestamp(frame)

            # on camera computer:
            cam_received_timestamp = mydata['cam_received_timestamp']

            latency_sec = cam_received_timestamp-trigger_timestamp
            mean_latency_sec = latency_sec.mean()

            ## for i in range(len(frame)):
            ##     print frame[i], repr(trigger_timestamp[i]), repr(cam_received_timestamp[i])
            ##     if i>=10:
            ##         break

            print '%s: mean latency %.1f +/- %.1f msec'%(
                cam_id,
                mean_latency_sec*1000.0,
                worst_sync_dict[hostname]*1000.0)

if __name__=='__main__':
    main()
