#emacs, this is -*-Python-*- mode
# $Id: c_python.pxd 1020 2006-05-01 22:32:39Z astraw $

cimport c_lib

cdef extern from "Python.h":
    #cdef object PyExc_OSError
    ctypedef int Py_intptr_t
#    ctypedef int Py_ssize_t
    void Py_BEGIN_ALLOW_THREADS()
    void Py_END_ALLOW_THREADS()
    void Py_INCREF(object)
    void Py_DECREF(object)
    
    object PyErr_SetFromErrno(object)
    
    object PyCObject_FromVoidPtr( void* cobj, void (*destr)(void *))
    object PyCObject_FromVoidPtrAndDesc( void* cobj, void* desc, void (*destr)(void *, void *))
    int PyCObject_Check(arr)
    void* PyCObject_AsVoidPtr(object)
    
    object PyLong_FromVoidPtr( void *p)
    void* PyLong_AsVoidPtr( object )
    object PyString_FromStringAndSize( char *v, int len )
    object PyString_FromString( char *v)
    int PyFile_Check(object fobj)
    c_lib.FILE* PyFile_AsFile(object fobj)
    
#    int PyObject_AsWriteBuffer( object, void **buffer, Py_ssize_t *buffer_len)
    int PyObject_AsWriteBuffer( object, void **buffer, int *buffer_len)
