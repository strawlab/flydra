import numpy
import FOE_utils
import glob, time, os, sys

# wind
h5files = glob.glob('*.h5')
logfiles = glob.glob('escape_wall*.log')

(all_results, all_results_times, trigger_fnos,
 projector_trig_times, tf_hzs, orig_times) = FOE_utils.get_results_and_times(logfiles,h5files,
                                                                             get_orig_times=True)

if 1:
    #  filter out bad data (frame numbers off, can't trust data)
    good_cond = ~(  (1139987065<orig_times) & (orig_times<1139990048)  )
    trigger_fnos = trigger_fnos[good_cond]
    orig_times = orig_times[good_cond]
    projector_trig_times = projector_trig_times[good_cond]
    tf_hzs = tf_hzs[good_cond]
    del good_cond

N_triggers = len(trigger_fnos)
print '%d FOE triggers'%N_triggers

fname = 'trigger_roundtrip_data.txt'
fd = open(fname,'r')
A = numpy.asarray([map(float,line.strip().split()) for line in fd.readlines()])
return_times = A[:,0]
log_fnos = list(A[:,1].astype(numpy.int64))

roundtrip_durs = []
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
    print fno
    print orig_time, time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(orig_time))
    print return_time, time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(return_time))
    
    print roundtrip*1000.0
    print
    roundtrip_durs.append(roundtrip)
    
roundtrip_durs=numpy.array(roundtrip_durs,dtype=numpy.float64)
# this index is into log_fnos array
accept = roundtrip_durs<0.015 # 15 msec

#print roundtrip_durs*1000.0
mean_roundtrip = numpy.mean(roundtrip_durs)
std_roundtrip = numpy.std(roundtrip_durs)
print 'mean',mean_roundtrip*1000.0
print 'std',std_roundtrip*1000.0

mean_roundtrip = numpy.mean(roundtrip_durs[accept])
std_roundtrip = numpy.std(roundtrip_durs[accept])
print 'accepted mean',mean_roundtrip*1000.0
print 'accepted std',std_roundtrip*1000.0

trigger_fnos = trigger_fnos[accept]
projector_trig_times = projector_trig_times[accept]
tf_hzs = tf_hzs[accept]
orig_times = orig_times[accept]
roundtrip_durs = roundtrip_durs[accept]
print 'roundtrip_durs.shape',roundtrip_durs.shape
print 'orig_times.shape',orig_times.shape

fd = open('accepted_triggers.txt','w')
print >> fd, '# filtered from %s'%(' '.join(logfiles),)
for i in range(len(trigger_fnos)):
    print >>fd, trigger_fnos[i], projector_trig_times[i], tf_hzs[i]
    print trigger_fnos[i], repr(orig_times[i]), roundtrip_durs[i]
fd.close()

if 0:
    import pylab
    pylab.hist(roundtrip_durs[accept]*1000.0,bins=100)
    pylab.show()
     
    
