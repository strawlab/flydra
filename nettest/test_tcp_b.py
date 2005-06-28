import socket
import select
import threading
import time
import struct

SERVER_PORT = 31423
RMT_PORT = 31422

fmt = '>id'
fmt_size = struct.calcsize(fmt)

def server_func():
    hostname = ''

    # open server port
    sockobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sockobj.bind((hostname, RMT_PORT))
    sockobj.listen(1)
    sockobj.setblocking(0)

    listen_sockets = [sockobj]
    empty_list = []
    select_select = select.select
    
    timeout = 5.0

    client_sockobj = None
    while client_sockobj is None:
        print 'listening for first connection...'
        try:
            in_ready, out_ready, exc_ready = select_select( listen_sockets,
                                                            empty_list, empty_list, timeout )
        except select.error, exc:
            print 'select.error on server socket, ignoring...'
            continue
        for sockobj in in_ready:
            client_sockobj, addr = sockobj.accept()
            client_sockobj.setblocking(0)
            print 'connected from',addr

    outgoing_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print 'connecting to server',addr[0]
    outgoing_socket.connect((addr[0], SERVER_PORT))

    listen_sockets = [client_sockobj]
    while 1:
        try:
            in_ready, out_ready, exc_ready = select_select( listen_sockets,
                                                            empty_list,
                                                            empty_list,
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
            newdata = sockobj.recv(fmt_size)
            #istr,timestr = newdata.split()
            #outstr = istr + ' ' + timestr
            outgoing_socket.send(newdata)
            print 'received:',repr(newdata)
    outgoing_socket.close()

server_func()
