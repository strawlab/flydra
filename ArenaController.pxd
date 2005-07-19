# Pyrex C-level interface definition

# ArenaController.pyx must implement this interface.  Defining these
# at the Pyrex C-level allows other Pyrex C code to call without going
# through the Python interpreter layer, thus avoiding the need
# acquiring the Python GIL.

ctypedef public class ArenaController [object PyArenaControllerObject, type PyArenaControllerType]:
    cdef void arena_update(self, double x, double y, double orientation,
                           double timestamp, long framenumber)
    cdef long rotation_calculation_init( self, int nframes )
    cdef void rotation_calculation_finish( self )
    cdef void rotation_update( self, double fly_x_pos, double fly_y_pos, double new_orientation, double timestamp )
