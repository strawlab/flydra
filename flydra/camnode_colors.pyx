cimport c_python

cdef extern from "colors.h":
    int mono8_bggr_to_red_channel(unsigned char*,int,int,int) nogil
    int mono8_bggr_to_red_color(unsigned char*,int,int,int) nogil

ctypedef struct PyArrayInterface:
    int two                       # contains the integer 2 as a sanity check
    int nd                        # number of dimensions
    char typekind                 # kind in array --- character code of typestr
    int itemsize                  # size of each element
    int flags                     # flags indicating how the data should be interpreted
    c_python.Py_intptr_t *shape   # A length-nd array of shape information
    c_python.Py_intptr_t *strides # A length-nd array of stride information
    void *data                    # A pointer to the first element of the array

cdef char uchar
uchar = ord("u")

cdef int iRED_CHANNEL
cdef int iRED_COLOR

RED_CHANNEL = 0
RED_COLOR = 1

iRED_CHANNEL = 0
iRED_COLOR = 1

def replace_with_red_image( object arr, object coding, int chan):
    assert coding=='MONO8:BGGR'
    cdef int err
    cdef PyArrayInterface* inter
    cdef object cobj

    cobj = arr.__array_struct__

    if not c_python.PyCObject_Check(cobj):
        raise TypeError("expected CObject")
    inter = <PyArrayInterface*>c_python.PyCObject_AsVoidPtr(cobj)
    assert inter.two==2
    assert inter.nd==2
    assert inter.typekind == uchar
    assert inter.itemsize == 1
    assert inter.flags & 0x401 # WRITEABLE and CONTIGUOUS
    err = 2
    with nogil:
        if chan==iRED_CHANNEL:
            err = mono8_bggr_to_red_channel( <unsigned char*>inter.data, inter.strides[0], inter.shape[0], inter.shape[1] )
        elif chan==iRED_COLOR:
            err = mono8_bggr_to_red_color( <unsigned char*>inter.data, inter.strides[0], inter.shape[0], inter.shape[1] )
    if err:
        raise RuntimeError("to_red() returned error %d"%err)
