# implementation of interface defined in ArenaController.pxd

# This is a wrapper around code written in C. The point of wrapping
# here is to make a Python extension type so that Pyrex C-level code
# can be called efficiently from other Pyrex modules with Python-like
# behavior.  This way, other modules (e.g. realtime_image_analysis)
# may use arena control if possible, but gracefully ignore it
# otherwise.

cdef extern from "arena_misc.h":
    cdef void start_center_calculation( int nframes )
    cdef void end_center_calculation( double *x_center, double *y_center )
    cdef void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation )

cdef extern from "arena_control.h":
    cdef long arena_initialize()
    cdef void arena_finish()

    cdef long rotation_calculation_init()
    cdef void rotation_calculation_finish( double new_x_cent, double new_y_cent )
    cdef void rotation_update()

    cdef void arena_update( double x, double y, double orientation,
                            double timestamp, long framenumber )

cdef int is_initialized # static variable since arena_control.c is global
is_initialized=0

class ArenaControlError(Exception):
    pass

ctypedef public class ArenaController [object PyArenaControllerObject, type PyArenaControllerType]:
    def __init__(self):
        if is_initialized:
            raise ArenaControlError('arena_control already initialized')
        err = arena_initialize()
        if err != 0:
            raise ArenaControlError('error %d initializing arena_control'%err)
        is_initialized = 1
    def __dealloc__(self):
        if is_initialized:
            arena_finish()
            is_initialized = 0
    cdef void arena_update(self, double x, double y, double orientation,
                           double timestamp, long framenumber):
        arena_update(x,y,orientation,timestamp,framenumber)
    cdef void start_center_calculation( self, int nframes ):
        start_center_calculation( nframes )
    cdef void end_center_calculation( self, double *x_center, double *y_center ):
        end_center_calculation( x_center, y_center )
    cdef void update_center_calculation( self, double new_x_pos, double new_y_pos, double new_orientation ):
        update_center_calculation( new_x_pos, new_y_pos, new_orientation )

    cdef long rotation_calculation_init( self ):
        return rotation_calculation_init()
    cdef void rotation_calculation_finish( self, double new_x_cent, double new_y_cent ):
        rotation_calculation_finish( new_x_cent, new_y_cent )
    cdef void rotation_update( self ):
        rotation_update()
    
