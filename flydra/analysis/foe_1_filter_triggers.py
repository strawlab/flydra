from __future__ import division
import numpy
import FOE_utils
import glob, time, os, sys

MAX_LATENCY = 0.035 # 35 msec

# wind
h5files = glob.glob('*.h5')
logfiles = glob.glob('escape_wall*.log')
logfiles.sort()
h5files.sort()

(all_results, all_results_times, trigger_fnos,
 projector_trig_times, tf_hzs, orig_times) = FOE_utils.get_results_and_times(logfiles,h5files,
                                                                             get_orig_times=True)

N_triggers = len(trigger_fnos)
print '%d FOE triggers'%N_triggers

#fname = 'trigger_roundtrip_data.txt' # received on other computer
fname = 'trigger_flip_data.txt' # swap buffers command sent from other computer
fd = open(fname,'r')
A = numpy.asarray([map(float,line.strip().split()) for line in fd.readlines()])
return_times = A[:,0]
log_fnos = list(A[:,1].astype(numpy.int64))

roundtrip_durs = []
all_data_avail = []
for idx in range(N_triggers):
    fno = trigger_fnos[idx]

    try:
        idx2 = log_fnos.index(fno)
    except ValueError:
        return_time = numpy.nan
    else:
        return_time = return_times[idx2] # different indexing system than the rest...
    
    orig_time = orig_times[idx]
    roundtrip = return_time-orig_time
    if 0:
        print fno
        print orig_time, time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(orig_time))
        print return_time, time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(return_time))

        print roundtrip*1000.0
        print
    roundtrip_durs.append(roundtrip)
    if not numpy.isnan(roundtrip):
        all_data_avail.append(idx)
    
roundtrip_durs=numpy.array(roundtrip_durs,dtype=numpy.float64)
# this index is into log_fnos array
accept = roundtrip_durs<MAX_LATENCY # 35 msec

n_orig = len(all_data_avail)
n_accepted = len(numpy.nonzero(accept))
print '%d original triggers with all data, %d accepted (%.2f)'%(
    n_orig, n_accepted, n_accepted/n_orig)

#print roundtrip_durs*1000.0
mean_roundtrip = numpy.mean(roundtrip_durs[all_data_avail])
std_roundtrip = numpy.std(roundtrip_durs[all_data_avail])
print 'all data: mean %f +/- %f (msec)'%(mean_roundtrip*1000.0,
                                         std_roundtrip*1000.0)

mean_roundtrip = numpy.mean(roundtrip_durs[accept])
std_roundtrip = numpy.std(roundtrip_durs[accept])
print 'accepted: mean %f +/- %f (msec)'%(mean_roundtrip*1000.0,
                                         std_roundtrip*1000.0)

trigger_fnos = trigger_fnos[accept]
projector_trig_times = projector_trig_times[accept]
tf_hzs = tf_hzs[accept]
orig_times = orig_times[accept]

fd = open('accepted_triggers.txt','w')
print >> fd, '# filtered from %s'%(' '.join(logfiles),)
for i in range(len(trigger_fnos)):
    print >>fd, trigger_fnos[i], projector_trig_times[i], tf_hzs[i]
fd.close()

if 1:
    import pylab
    pylab.subplot(2,1,1)
    pylab.hist(roundtrip_durs[all_data_avail]*1000.0,bins=100)
    pylab.title('all data available')
    pylab.subplot(2,1,2)
    pylab.hist(roundtrip_durs[accept]*1000.0,bins=100)
    pylab.title('accepted')
    pylab.xlabel('latency (msec)')
    #pylab.show()
    pylab.savefig('roundtrip_latencies')
     
    
