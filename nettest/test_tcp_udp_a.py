import socket
import select
import threading
import time
import struct
import sys

RMT_HOSTNAME = '192.168.1.101'
RMT_PORT = 31422
SERVER_PORT = 31423

fmt = '>id'
fmt_size = struct.calcsize(fmt)

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
            i,sendtime = struct.unpack(fmt,newdata)
            tdelta = recvtime-sendtime
            print '%d: %g msec'%(i,tdelta*1000.0)

##def server_func():
##    hostname = ''

##    # open server port
##    sockobj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##    sockobj.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
##    sockobj.bind((hostname, SERVER_PORT))
##    sockobj.listen(1)
##    sockobj.setblocking(0)
##    listen_sockets = [sockobj]
##    empty_list = []
##    select_select = select.select
    
##    timeout = 5.0

##    client_sockobj = None
##    while client_sockobj is None:
##        print 'SERVER: listening for first connection...'
##        try:
##            in_ready, out_ready, exc_ready = select_select( listen_sockets,
##                                                            empty_list, empty_list, timeout )
##        except select.error, exc:
##            print 'select.error on server socket, ignoring...'
##            continue
##        for sockobj in in_ready:
##            client_sockobj, addr = sockobj.accept()
##            client_sockobj.setblocking(0)
##            print 'SERVER: connected from',addr

##    listen_sockets = [client_sockobj]
##    while 1:
##        print 'SERVER: listening for input on client_sockobj...'
##        try:
##            in_ready, out_ready, exc_ready = select_select( listen_sockets,
##                                                            empty_list,
##                                                            empty_list,
##                                                            timeout )
##        except select.error, exc:
##            print 'select.error on server socket, ignoring...'
##            continue
##        except socket.error, exc:
##            print 'socket.error on server socket, ignoring...'
##            continue
##        except Exception, exc:
##            raise
##        except:
##            print 'ERROR: received an exception not derived from Exception'
##            print '-='*10,'I should really quit now!','-='*10
##            continue
##        for sockobj in in_ready:
##            newdata = sockobj.recv(fmt_size)
##            print 'SERVER: got data',repr(newdata)
##            recvtime = time_func()
##            #istr,timestr = newdata.split()
##            istr,timestr = struct.unpack(fmt,newdata)
##            sendtime = float(timestr)
##            #print sendtime, recvtime
##            tdelta = recvtime-sendtime
##            print '%s: %g msec'%(istr,tdelta*1000.0)
##            #print addr,':',newdata
##    print 'server_thread done'

server_thread = threading.Thread(target=server_func)
server_thread.setDaemon(True)
server_thread.start()

i=0
outgoing_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'connecting to remote host...'
outgoing_socket.connect((RMT_HOSTNAME, RMT_PORT))
print 'connected'
try:
    while 1:
        i+=1
        outstr = struct.pack( fmt, i,time_func())
        #outstr = '%d %s'%(i,repr(time_func()) )
        outgoing_socket.send(outstr)
        print 'SENDER: sent',repr(outstr)
        time.sleep(0.1)
finally:
    outgoing_socket.close()
    
