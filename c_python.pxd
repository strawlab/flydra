#emacs, this is -*-Python-*- mode
# $Id$

cdef extern from "Python.h":
    void Py_BEGIN_ALLOW_THREADS()
    void Py_END_ALLOW_THREADS()
    void Py_DECREF(object)
