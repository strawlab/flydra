import socket
import select
import threading
import time
import sys
import struct

if len(sys.argv) > 1:
    RMT_HOSTNAME = sys.argv[1]
else:
    RMT_HOSTNAME = '192.168.1.102'

print 'RMT_HOSTNAME',repr(RMT_HOSTNAME)

RMT_PORT = 31422
SERVER_PORT = 31423

if sys.platform.startswith('win'):
    time_func = time.clock
else:
    time_func = time.time

def server_func():
    hostname = ''

    # open UDP server port
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj.bind((hostname, SERVER_PORT))
    sockobj.setblocking(0)
    listen_sockets = [sockobj]
    emptylist = []
    select_select = select.select
    
    timeout = 5.0

    last_i = 0
    while 1:
        try:
            in_ready, out_ready, exc_ready = select_select( listen_sockets,
                                                            emptylist,
                                                            emptylist,
                                                            timeout )
        except select.error, exc:
            print 'select.error on server socket, ignoring...'
            continue
        except socket.error, exc:
            print 'socket.error on server socket, ignoring...'
            continue
        except Exception, exc:
            raise
        except:
            print 'ERROR: received an exception not derived from Exception'
            print '-='*10,'I should really quit now!','-='*10
            continue
        for sockobj in in_ready:
            newdata, addr = sockobj.recvfrom(4096)
            recvtime = time_func()
            timetime = time.time() # not the same as recvtime on windows
            i,sendtime,rmttimetime = struct.unpack('idd',newdata)
            #print sendtime, recvtime
            tdelta = (recvtime-sendtime)*1000.0
            rmt_tdelta = (timetime-rmttimetime)*1000.0
            if tdelta < 0.5:
            #if 1:
                print '%d: %g msec (this-remote tdiff: %g msec)'%(i,tdelta,rmt_tdelta)
            #print addr,':',newdata
            if i-last_i != 1:
                print '*******SKIP:',i
            if tdelta > 5:
                print '            tdelta > 5 msec'
            last_i = i

server_thread = threading.Thread(target=server_func)
server_thread.setDaemon(True)
server_thread.start()

i=0
rmt_host = (RMT_HOSTNAME,RMT_PORT)
outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    while 1:
        i+=1
        outstr = struct.pack('id',i,time_func())
        outgoing_UDP_socket.sendto(outstr, rmt_host )
        #print 'sent data to',rmt_host
        time.sleep(0.5)
finally:
    outgoing_UDP_socket.close()
