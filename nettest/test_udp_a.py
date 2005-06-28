import socket
import select
import threading
import time

RMT_HOSTNAME = '192.168.1.151'
RMT_PORT = 31422
SERVER_PORT = 31423

time_func = time.clock

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
            sendtime = float(timestr)
            #print sendtime, recvtime
            tdelta = recvtime-sendtime
            print '%s: %g msec'%(istr,tdelta*1000.0)
            #print addr,':',newdata

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
