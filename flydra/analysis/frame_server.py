#!/usr/bin/python
import Pyro.core
import os
import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
import socket

Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!

Pyro.config.PYRO_TRACELEVEL = 1
Pyro.config.PYRO_USER_TRACELEVEL = 1
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

def DEBUG():
    print 'line',sys._getframe().f_back.f_lineno,', thread', threading.currentThread()

class FrameServer(Pyro.core.ObjBase):
    def __init__(self,*args,**kw):
        Pyro.core.ObjBase.__init__(self,*args,**kw)
        self._filename = None
        self._ts_dict = None

    def get_timestamp2frame(self):
        return self._ts_dict.copy()

    # not overriding Pyro's __init__ funciton...
    def load(self, movie_filename='/tmp/raw_video.fmf'):
        if self._filename == movie_filename:
            return
        self.fly_movie = FlyMovieFormat.FlyMovie( movie_filename )
        self._filename = movie_filename
        self._ts_dict = None
        self.load_timestamp_dict()

    def get_filename(self):
        return self._filename

    def load_timestamp_dict(self):
        result = {}
        frame0,timestamp0 = self.fly_movie.get_frame(0)
        result[timestamp0] = 0
        i = 0
        try:
            while 1:
                i += 1
                ts = self.fly_movie.get_next_timestamp()
                result[ts] = i
        except FlyMovieFormat.NoMoreFramesException:
            pass
        self._ts_dict = result

    def get_frame_by_timestamp(self, timestamp):
        try:
            return self.fly_movie.get_frame(self._ts_dict[timestamp])
        except KeyError:
            print >> sys.stderr, repr(timestamp)
            for key in self._ts_dict:
                print >> sys.stderr, ' '
                if abs(key-timestamp) < 0.02:
                    print >> sys.stderr, repr(key)
            raise

    def get_frame_prior_to_timestamp(self, target_timestamp):
        timestamps = self._ts_dict.keys()
        timestamps.sort()
        real_timestamp = None
        for ts in timestamps:
            if ts <= target_timestamp:
                real_timestamp = ts
        if real_timestamp is None:
            print >> sys.stderr, 'could not find %s in:',repr(target_timestamp)
            for ts in timestamps:
                print >> sys.stderr, ' ts',type(ts),ts
        return self.fly_movie.get_frame(self._ts_dict[real_timestamp])

    def get_frame(self, frame_number):
        return self.fly_movie.get_frame(frame_number)

    def get_timestamp(self, frame_number):
        frame, timestamp = self.fly_movie.get_frame(frame_number)
        return timestamp

    def noop(self):
        return

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 9888

    Pyro.core.initServer(banner=0,storageCheck=0)

    # start Pyro server
    hostname = socket.gethostbyname(socket.gethostname())

    daemon = Pyro.core.Daemon(host=hostname,port=port)
    frame_server = FrameServer()
    URI=daemon.connect(frame_server,'frame_server')
    print >> sys.stderr, 'listening on',URI
    while 1:
        daemon.handleRequests(60.0) # block on select
