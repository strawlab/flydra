#emacs, this is -*-Python-*- mode
cimport serial_comm

cdef extern from "Python.h":
    int PyString_AsStringAndSize( object obj, char **buffer, int *length) except -1
    object PyString_FromStringAndSize( char *v, int len)
    
# Global Variables and Constants *******************************************

COMM_PORT = "/dev/ttyS0"
BAUD_RATE = serial_comm.SC_BAUD_RATE

cdef void CHK( long errval ) except *:
    if errval != serial_comm.SC_SUCCESS_RC:
        raise RuntimeError("serial_comm error %d"%errval)
    
def open_port( char * port_name ):
    cdef int hcomm
    CHK(serial_comm.sc_open_port(&hcomm,port_name))
    return hcomm

def close_port( int hcomm ):
    CHK(serial_comm.sc_close_port(&hcomm))

def send_cmd( int hcomm, cmd_buffer ):
    cdef int len
    cdef char* buf

    PyString_AsStringAndSize( cmd_buffer, &buf, &len)
    CHK(serial_comm.sc_send_cmd(&hcomm, buf, len))
    
def receive_msg( int hcomm ):
    cdef int len
    cdef char* buf
    
    CHK(serial_comm.sc_receive_msg(&hcomm, buf, &len))
    return PyString_FromStringAndSize( buf, len)

def quick_send( cmd, port_name=COMM_PORT ):
    p=open_port(port_name)
    send_cmd(p,cmd)
    close_port(p)
