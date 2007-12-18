import socket
import common_variables
import flydra.kalman.flydra_tracker as flydra_tracker
import numpy
import time, math, sys

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

class Listener:
    def run(self):
        tick = 0
        print 'listening...'
        
        #sender.sendto('x',('127.0.0.1',common_variables.trigger_network_socket_port))
        post =  0.758, 0.128, 0.244
        post = numpy.array(post)
        #post = post + (0,0,.05)
        last_trig = 0.0
        while 1:
            databuf, addr = sockobj.recvfrom(1024)

            # relay the data
            for host in downstream_kalman_hosts:
                sender.sendto(databuf,host)
            
            corrected_framenumber, timestamp, state_vecs, meanP=flydra_tracker.decode_data_packet(databuf)
##            if len(state_vecs):
##                print
##                print 'frame %d'%corrected_framenumber
##            for i,state_vec in enumerate(state_vecs):
##                x,y,z=state_vec[:3]
##                print '  obj %d: (%f,%f,%f)'%(i, x,y,z)
                
            now = time.time()
            if abs(now-timestamp) > 0.05:
                sys.stdout.write('W')
                sys.stdout.flush()
                #print 'WARNING: %.1f msec out of sync'%((now-timestamp)*1e3,)
                #print 'WARNING: data more than 50 msec out-of-sync'
            for i,state_vec in enumerate(state_vecs):
                state_vec = numpy.array(state_vec)
                dist_m = math.sqrt(numpy.sum((post-state_vec[:3])**2))
                # distance and time since last
                if dist_m < 0.020 and (now-last_trig) > 5.0:
                    vel = state_vec[3:6]
                    mean_vel = abs( math.sqrt(numpy.sum( vel**2) ) )
                    print 'mean_vel %.2f (m/s), meanP'%(mean_vel,),meanP*1e3
                    
                    #if mean_vel > 0.14:
                    if 1:
                        sender.sendto('x',('127.0.0.1',common_variables.trigger_network_socket_port))
                        last_trig = time.time()
                        print 'trig!', last_trig, state_vec[:3]
            
listener = Listener()
listener.run()
