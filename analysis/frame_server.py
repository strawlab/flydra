#!/usr/bin/python
import Pyro.core
import os
import FlyMovieFormat
import socket

Pyro.config.PYRO_MULTITHREADED = 0 # No multithreading!

Pyro.config.PYRO_TRACELEVEL = 0
Pyro.config.PYRO_USER_TRACELEVEL = 0
Pyro.config.PYRO_DETAILED_TRACEBACK = 0
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

class FrameServer(Pyro.core.ObjBase):
    def __init__(self,*args,**kw):
        Pyro.core.ObjBase.__init__(self,*args,**kw)
        self._filename = None
        self._ts_dict = None
        
    # not overriding Pyro's __init__ funciton...
    def load(self, movie_filename='/tmp/raw_video.fmf'):
        self.fly_movie = FlyMovieFormat.FlyMovie( movie_filename )
        self._filename = movie_filename
        self._ts_dict = None
        self.load_timestamp_dict()

    def get_filename(self):
        return self._filename

    def load_timestamp_dict(self):
        result = {}
        timestamp0 = self.fly_movie.get_frame(0)
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
        return self.fly_movie.get_frame(self._ts_dict[timestamp])
        
    def get_frame(self, frame_number):
        return self.fly_movie.get_frame(frame_number)

    def get_timestamp(self, frame_number):
        frame, timestamp = self.fly_movie.get_frame(frame_number)
        return timestamp

    def noop(self):
        return

if __name__ == '__main__':
    Pyro.core.initServer(banner=0,storageCheck=0)
    
    # start Pyro server
    hostname = socket.gethostbyname(socket.gethostname())
    port = 9888
    
    daemon = Pyro.core.Daemon(host=hostname,port=port)
    frame_server = FrameServer()
    URI=daemon.connect(frame_server,'frame_server')
    print 'listening on',URI
    while 1:
        daemon.handleRequests(60.0) # block on select
