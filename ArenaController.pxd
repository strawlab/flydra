# Pyrex C-level interface definition

# ArenaController.pyx must implement this interface.  Defining these
# at the Pyrex C-level allows other Pyrex C code to call without going
# through the Python interpreter layer, thus avoiding the need
# acquiring the Python GIL.

ctypedef public class ArenaController [object PyArenaControllerObject, type PyArenaControllerType]:
    cdef void arena_update(self, double x, double y, double orientation,
                           double timestamp, long framenumber)
    cdef void start_center_calculation(self, int nframes )
    cdef void end_center_calculation(self, double *x_center, double *y_center )
    cdef void update_center_calculation(self, double new_x_pos, double new_y_pos, double new_orientation )

    cdef long rotation_calculation_init( self )
    cdef void rotation_calculation_finish( self, double new_x_cent, double new_y_cent )
    cdef void rotation_update( self )
