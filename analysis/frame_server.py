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
    def load(self, movie_filename='/tmp/raw_video.fmf'):
        self.fly_movie = FlyMovieFormat.FlyMovie( movie_filename )

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
    while 1:
        print 'listening on',URI
        daemon.handleRequests(60.0) # block on select
