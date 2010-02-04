cimport c_python

cdef extern from "colors.h":
    int mono8_bggr_to_red(char*,int,int,int)

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

def replace_with_red_image( object arr, object coding):
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
    err = mono8_bggr_to_red( <char*>inter.data, inter.strides[0], inter.shape[0], inter.shape[1] )
    if err:
        raise RuntimeError("to_red returned error %d"%err)
