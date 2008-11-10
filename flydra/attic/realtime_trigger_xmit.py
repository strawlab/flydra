import socket
import common_variables
import flydra.kalman.flydra_tracker as flydra_tracker
import numpy
import time, math, sys


class Listener:
    def run(self):
        global downstream_kalman_hosts
        tick = 0
        gustlength = .05 # seconds
        print 'listening...'

        while 1:
            superdatabuf, addr = sockobj.recvfrom(1024)
            sys.stdout.write('.')
            sys.stdout.flush()

            if 1:
                # relay the data
                for host in downstream_kalman_hosts:
                    sender.sendto(superdatabuf,host)

##             data_packets = flydra_tracker.decode_super_packet(superdatabuf)
##            print len(data_packets)

if __name__=='__main__':
    global downstream_kalman_hosts
    hostname = '127.0.0.1'
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    port = common_variables.realtime_kalman_port

    sockobj.bind(( hostname, port))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    downstream_kalman_hosts = []
    if 1:
        downstream_kalman_hosts.append( ('astraw-office.kicks-ass.net',28931))

    for host in downstream_kalman_hosts:
        print 'forwarding to',host

    listener = Listener()
    listener.run()
