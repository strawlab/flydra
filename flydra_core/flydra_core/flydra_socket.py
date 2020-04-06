import socket
import struct
import math

# -----------------------------------------------------------------------

max_datagram_size = 65507  # Theoretical max (see
# http://en.wikipedia.org/wiki/User_Datagram_Protocol ,
# but if it's over ~512,
# see http://stackoverflow.com/questions/19057572 ).


class FlydraTransportSender:
    def __init__(self, destination_addr):
        assert isinstance(destination_addr, AddrInfoBase)
        self._dest = destination_addr
        self._sockaddr = self._dest.sockaddr
        if self._dest.is_ip_socket():
            # verify we do not need DNS lookup
            address, port = self._sockaddr[:2]  # works for IPv4 and IPv6
            assert does_host_require_DNS(address, port) == False
        self._sockobj = socket.socket(self._dest.family, socket.SOCK_DGRAM)

    def send(self, data):
        assert len(data) <= max_datagram_size
        self._sockobj.sendto(data, self._sockaddr)


class DummySender:
    def send(self, data):
        return


# -----------------------------------------------------------------------


class FlydraTransportReceiver:
    def __init__(self, addr, socket_timeout_seconds=None):
        assert isinstance(addr, AddrInfoBase)
        self._addr = addr
        self._sockobj = socket.socket(self._addr.family, socket.SOCK_DGRAM)
        if socket_timeout_seconds is not None:
            timeout_sec = int(math.floor(socket_timeout_seconds))
            timeout_usec = int((socket_timeout_seconds - timeout_sec) * 1e6)
            timeval = struct.pack("LL", timeout_sec, timeout_usec)
            self._sockobj.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)
        self._sockobj.bind(self._addr.sockaddr)

    def get_listen_addrinfo(self):
        sockaddr = self._sockobj.getsockname()
        if self._addr.is_ip_socket():
            host, port = sockaddr[:2]
            return make_addrinfo(host=host, port=port)
        else:
            assert self._addr.is_unix_domain_socket()
            return make_addrinfo(filename=sockaddr)

    def recv(self, return_sender_sockaddr=False):
        data, sockaddr = self._sockobj.recvfrom(max_datagram_size)
        if return_sender_sockaddr:
            return data, sockaddr
        return data


def get_dummy_sender():
    return DummySender()


# --------------------------------------------------------

# These classes represent addresses of either a unix domain socket or
# an IP address.


class AddrInfoBase:
    def is_unix_domain_socket(self):
        return self.family == socket.AF_UNIX

    def is_ip_socket(self):
        return self.family in [socket.AF_INET, socket.AF_INET6]

    def to_dict(self):
        """return dict to copy self when calling make_address(**result)"""
        raise NotImplementedError("derived class must override this method")

    def __eq__(self, other):
        raise NotImplementedError("derived class must override this method")

    def __repr__(self):
        return "%r" % self.to_dict()


class AddrInfoUnixDomainSocket(AddrInfoBase):
    def __init__(self, filename):
        self.sockaddr = filename
        self.family = socket.AF_UNIX

    def to_dict(self):
        return {
            "filename": self.sockaddr,
        }

    def __eq__(self, other):
        if not isinstance(other, AddrInfoUnixDomainSocket):
            return False
        if self.sockaddr != other.sockaddr:
            return False
        return True


class AddrInfoIP(AddrInfoBase):
    def __init__(self, host, port):
        self.host = host
        self.port = port

        # IPv4 or IPv6 socket
        addrs = socket.getaddrinfo(host, port)
        assert len(addrs) >= 1
        for addr in addrs:
            (family, socktype, proto, canonname, sockaddr) = addr
            if family == socket.AF_INET:
                # Prefer IPv4 for now
                break

        self.family = family
        assert self.is_ip_socket()
        self.sockaddr = sockaddr

    def to_dict(self):
        return {
            "host": self.host,
            "port": self.port,
        }

    def __eq__(self, other):
        if not isinstance(other, AddrInfoIP):
            return False
        if self.host != other.host:
            return False
        if self.port != other.port:
            return False
        return True


# --------------------------------------------------------


def does_host_require_DNS(host, port=None, family=0):
    try:
        addrs = socket.getaddrinfo(host, port, family, 0, 0, socket.AI_NUMERICHOST)
    except socket.gaierror as err:
        if err.errno == -2:
            requires_DNS = True
        else:
            raise
    else:
        requires_DNS = False
    return requires_DNS


# --------------------------------------------------------


def make_addrinfo(
    host=None, port=None, filename=None,
):
    """factory function to return and instance of AddrInfoBase"""

    if filename is not None:
        # unix domain socket
        assert host is None
        assert port is None
        return AddrInfoUnixDomainSocket(filename=filename)
    else:
        return AddrInfoIP(host=host, port=port)


# -----------------------------------------------------------------------------


def get_bind_address():
    """return a string like '0.0.0.0' or '127.0.0.1'"""
    import roslib.network

    return roslib.network.get_bind_address()


# -----------------------------------------------------------------------------


def test_flydra_ip_sockets():
    for host, port, require_DNS_expected in [
        ("localhost", None, True),
        ("127.0.0.1", None, False),
        ("::1", None, False),
        ("www.google.com", 80, True,),
        ("www.google.com", None, True),
        ("8.8.8.8", None, False),
        ("ipv6.whatismyv6.com", None, True),
        ("2001:4810::110", None, False),
    ]:
        addr_info = make_addrinfo(host=host, port=port)
        assert does_host_require_DNS(host, port) == require_DNS_expected

        addr_info2 = make_addrinfo(**addr_info.to_dict())
        assert addr_info2 == addr_info


def test_flydra_unix_domain_sockets():
    filename = "/tmp/test_flydra_socket"  # never actually created
    addr_info = make_addrinfo(filename=filename)
    addr_info2 = make_addrinfo(**addr_info.to_dict())
    assert addr_info2 == addr_info


if __name__ == "__main__":
    test_flydra_unix_domain_sockets()
    test_flydra_ip_sockets()
