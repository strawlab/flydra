import socket, time, select, os

escape_wall_listener = socket.socket( socket.AF_INET, socket.SOCK_DGRAM)
escape_wall_listener.setblocking(0)
escape_wall_listener.bind( ('',28932) )

listeners = [escape_wall_listener]
empty_list = []
fname = 'trigger_roundtrip_data.txt'
full_fname = os.path.join(os.path.abspath(os.curdir),fname)
print 'saving to %s'%full_fname
data_file = open(full_fname,'a')
while 1:
    in_ready, trash1, trash2 = select.select(listeners,empty_list,empty_list,5.0)
    if escape_wall_listener in in_ready:
        data = escape_wall_listener.recv(4096)
        print >> data_file, repr(time.time()),data
        data_file.flush()
    
