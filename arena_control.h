#ifndef _flydra_ARENA_CONTROL_h_
#define _flydra_ARENA_CONTROL_h_

long arena_initialize( void );
void arena_finish( void );

long rotation_calculation_init( void );
void rotation_calculation_finish( double new_x_cent, double new_y_cent );

void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber );

#endif
