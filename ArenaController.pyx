# implementation of interface defined in ArenaController.pxd

# This is a wrapper around code written in C. The point of wrapping
# here is to make a Python extension type so that Pyrex C-level code
# can be called efficiently from other Pyrex modules with Python-like
# behavior.  This way, other modules (e.g. realtime_image_analysis)
# may use arena control if possible, but gracefully ignore it
# otherwise.

cdef extern from "arena_control.h":
    cdef long arena_initialize()
    cdef void arena_finish()

    cdef long rotation_calculation_init( int nframes )
    cdef void rotation_calculation_finish()
    cdef void rotation_update( double fly_x_pos, double fly_y_pos, double new_orientation, double timestamp )

    cdef void arena_update( double x, double y, double orientation,
                            double timestamp, long framenumber )

cdef int is_initialized # static variable since arena_control.c is global
is_initialized=0

class ArenaControlError(Exception):
    pass

ctypedef public class ArenaController [object PyArenaControllerObject, type PyArenaControllerType]:
    def __new__(self,*args,**kw):
        if is_initialized:
            raise ArenaControlError('arena_control already initialized')
        err = arena_initialize()
        if err != 0:
            raise ArenaControlError('error %d initializing arena_control'%err)
        is_initialized = 1
    def __dealloc__(self):
        'ArenaController.__dealloc__() called'
        if is_initialized:
            self.close()
    def close(self):
        'ArenaController.close() called'
        if is_initialized:
            arena_finish()
            is_initialized = 0
    cdef void arena_update(self, double x, double y, double orientation,
                           double timestamp, long framenumber):
        arena_update(x,y,orientation,timestamp,framenumber)

    cdef long rotation_calculation_init( self, int nframes ):
        return rotation_calculation_init( nframes )
    cdef void rotation_calculation_finish( self ):
        rotation_calculation_finish()
    cdef void rotation_update( self, double fly_x_pos, double fly_y_pos, double new_orientation, double timestamp ):
        rotation_update( fly_x_pos, fly_y_pos, new_orientation, timestamp )
    
