#emacs, this is -*-Python-*- mode

cdef extern from "arena_control.h":

    long arena_initialize()
    void arena_finish()

    void arena_update( double x, double y, double orientation,
        double timestamp, long framenumber )
