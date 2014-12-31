import socket
import struct

# -----------------------------------------------------------------------

max_datagram_size = 65507 # Theoretical max (see
# http://en.wikipedia.org/wiki/User_Datagram_Protocol ,
# but if it's over ~512,
# see http://stackoverflow.com/questions/19057572 ).

class DummySender:
    def send(self,msg):
        return

class Sender:
    def __init__(self,destination_addr):
        validate_addr( self._family, destination_addr )
        self._destination_addr = destination_addr
        self._sockobj = socket.socket(self._family, socket.SOCK_DGRAM)
    def send(self,data):
        assert len(data) <= max_datagram_size
        self._sockobj.sendto(data,self._destination_addr)

class UDPSender(Sender):
    _family = socket.AF_INET

class UnixDomainDatagramSender(Sender):
    _family = socket.AF_UNIX

# -----------------------------------------------------------------------

class Receiver:
    def __init__(self, addr, socket_timeout=True):
        validate_addr( self._family, addr )
        self._sockobj = socket.socket(self._family, socket.SOCK_DGRAM)
        if socket_timeout:
            timeout_sec = 0 # in seconds
            timeout_usec = 500000 # in microseconds
            timeval=struct.pack("LL", timeout_sec, timeout_usec)
            self._sockobj.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)
        self._sockobj.bind(addr)
    def getsockname(self):
        return self._sockobj.getsockname()
    def recv(self):
        data, _ = self._sockobj.recvfrom(max_datagram_size)
        return data

class UDPReceiver(Receiver):
    _family = socket.AF_INET

class UnixDomainDatagramReceiver(Receiver):
    _family = socket.AF_UNIX

# -----------------------------------------------------------------------

def validate_addr(family,addr):
    if family == socket.AF_INET:
        assert is_udp_addr(addr)
    elif family == socket.AF_UNIX:
        assert is_uds_dgram_addr(addr)
    else:
        raise ValueError('unknown family: %r'%family)

def is_udp_addr(addr):
    if type(addr)==tuple and len(addr)==2:
        hostname = addr[0]
        port = addr[1]
        assert isinstance(hostname, basestring)
        assert isinstance(port, int)
        return True
    return False

def is_uds_dgram_addr(addr):
    return isinstance(addr, basestring)

# -----------------------------------------------------------------------

def get_sender_from_address(addr):
    if is_udp_addr(addr):
        return UDPSender( addr )
    elif is_uds_dgram_addr(addr):
        return UnixDomainDatagramSender( addr )
    else:
        raise ValueError('could not parse address: %r'%addr)

def get_dummy_sender():
    return DummySender()

