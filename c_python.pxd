#emacs, this is -*-Python-*- mode

# Structs and functions from numarray
cdef extern from "Python.h":
    void Py_BEGIN_ALLOW_THREADS()
    void Py_END_ALLOW_THREADS()
