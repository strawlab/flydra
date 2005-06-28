import socket
import select
import threading
import time

RMT_PORT = 31422

def server_func():
    hostname = ''

    # open UDP server port
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sockobj.bind((hostname, RMT_PORT))
    sockobj.setblocking(0)
    listen_sockets = [sockobj]
    emptylist = []
    select_select = select.select
    
    timeout = 5.0

    print 'listening for connections...'
    outgoing_UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
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
            istr,server_port,timestr = newdata.split()
            server_port=int(server_port)

            outstr = istr + ' ' + timestr
            outgoing_UDP_socket.sendto(outstr,(addr[0],server_port))
    outgoing_UDP_socket.close()

            #print addr,':',newdata            

server_func()
