#emacs, this is -*-Python-*- mode

cdef extern from "arena_control.h":

    long arena_initialize()
    void arena_finish()

    long rotation_calculation_init()
    void rotation_calculation_finish( double new_x_cent, double new_y_cent )
    void rotation_update()

    void arena_update( double x, double y, double orientation,
        double timestamp, long framenumber )
