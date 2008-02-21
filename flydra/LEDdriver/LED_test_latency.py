import flydra.LEDdriver.LEDdriver as LEDdriver
import time, socket, threading, Queue
import flydra.common_variables as common_variables
import flydra.kalman.flydra_tracker as flydra_tracker
import numpy

#hostname = socket.gethostbyname('mainbrain')
hostname = '127.0.0.1'
sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

port = common_variables.realtime_kalman_port

while 1:
    try:
        sockobj.bind(( hostname, port))
    except socket.error, err:
        if err.args[0]==98:
            port += 1
            continue
    break
print 'listening on',hostname,port

class Listener(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.quit_now = threading.Event()
        self.q = Queue.Queue()
    def run(self):
        tick = 0
        print 'listening...'
        while not self.quit_now.isSet():
            databuf, addr = sockobj.recvfrom(1024)
            corrected_framenumber, timestamp, state_vecs,meanP=flydra_tracker.decode_data_packet(databuf)
            recv_ts = time.time()
            self.q.put( (timestamp, state_vecs, recv_ts) )
            if 0:
                #print corrected_framenumber, timestamp, state_vecs
                line3d=None
                if len(state_vecs):
                    x,y,z=state_vecs[0][:3]
                    print x,y,z

    def quit(self):
        self.quit_now.set()

    def get_listened(self):
        result = []
        while 1:
            try:
                result.append( self.q.get_nowait() )
            except Queue.Empty:
                break
        return result

def main():
    dev = LEDdriver.Device()

    listener = Listener()
    listener.setDaemon(True) # don't let this thread keep app alive
    listener.start()

    # some bug in my firmware seems to require this...
    dev.set_carrier_frequency( 100.0 )
    fps = dev.get_carrier_frequency()
    trigger_timer_max = dev.get_timer_max()

    interval = 3.0

    while 1:
        pre = time.time()
        dev.ext_trig3()
        post = time.time()
        dur = post-pre
        av = (pre+post)/2.0
        print 'approximate time of start: %s, error estimate: max %.1f msec'%( repr(av), dur*1e3,)

        time.sleep(interval)
        data =listener.get_listened()
        IFIs = []
        IFIrs = []
        i=0
        for (timestamp, statevecs, recv_ts) in data:
            if timestamp < pre:
                continue
            if timestamp > (post+interval):
                continue
            # timestamp is the 3D reconstruction timestamp
            if i==0:
##                 print 'initial packet (%s) latency: %.1f msec (reconstruction), %.1f (receive)'%(
##                     repr(timestamp),
##                     (timestamp-av)*1e3,
##                     (recv_ts-av)*1e3,
##                     )
                print 'initial packet latency: %.1f msec (reconstruction), %.1f (receive)'%(
                    (timestamp-av)*1e3,
                    (recv_ts-av)*1e3,
                    )
            else:
                diff = (timestamp-prev_ts)*1e3
                diffr = (recv_ts-prev_rts)*1e3
                IFIs.append( diff )
                IFIrs.append( diffr )
            prev_ts = timestamp
            prev_rts = recv_ts
            i+=1
        IFIs = numpy.array(IFIs)
        IFIrs = numpy.array(IFIrs)
##         if len(IFIs):
##             print '%.1f +/- %.1f STD msec (min %.1f, max %.1f, n=%d)'%(
##                 IFIs.mean(), IFIs.std(), IFIs.min(), IFIs.max(), i )
        print

if __name__=='__main__':
    main()
