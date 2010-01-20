import threading
import Queue
import socket, time
import flydra.kalman.flydra_tracker as flydra_tracker

class Listener(threading.Thread):
    def __init__(self,sockobj):
        threading.Thread.__init__(self)
        self.sockobj=sockobj
        self.quit_now = threading.Event()
        self.q = Queue.Queue()
    def run(self):
        tick = 0
        print 'listening...'
        while not self.quit_now.isSet():
            buf, addr = self.sockobj.recvfrom(1024)
            data_packets = flydra_tracker.decode_super_packet( buf )
            for data_packet in data_packets:

                (corrected_framenumber, acquire_timestamp,
                 reconstruction_timestamp, state_vecs, meanP) = \
                 flydra_tracker.decode_data_packet(data_packet)

                recv_ts = time.time()
                self.q.put( (reconstruction_timestamp, state_vecs, recv_ts) )
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

