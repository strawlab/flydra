import socket
import select
import threading
import time
import sys

RMT_HOSTNAME = '192.168.1.199'
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
            istr,timestr = newdata.split()
            i = int(istr)
            sendtime = float(timestr)
            #print sendtime, recvtime
            tdelta = recvtime-sendtime
            print '%d: %g msec'%(i,tdelta*1000.0)
            #print addr,':',newdata
            if i-last_i != 1:
                print '*******SKIP:',i
            if tdelta > 0.005:
                print '            tdelta > 5 msec'
            last_i = i
            print 'last_i',last_i

server_thread = threading.Thread(target=server_func)
server_thread.setDaemon(True)
server_thread.start()

i=0
while 1:
    i+=1
    outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    outstr = '%d %d %s'%(i,SERVER_PORT, repr(time_func()) )
    outgoing_UDP_socket.sendto(outstr,(RMT_HOSTNAME,RMT_PORT))
    outgoing_UDP_socket.close()
    time.sleep(0.1)
