import datetime
import pytz # from http://pytz.sourceforge.net/
import glob, time

time_fmt = '%Y-%m-%d %H:%M:%S %Z%z'
pacific = pytz.timezone('US/Pacific')
fnames = glob.glob('*.log')
for fname in fnames:
    fd = open(fname,'r')
    for line in fd.readlines():
        if not line.startswith('#'):
            continue
        ls = line.strip().split()
        if ls[1]=='film' and ls[2]=='trigger':
            frame = int(ls[3])
            ts_float = float(ls[4])
            dt_ts = datetime.datetime.fromtimestamp(ts_float,pacific)
            tts = time.strftime(time_fmt, time.localtime(ts_float))
            print '% 8d %s'%(frame, tts)
    fd.close()


