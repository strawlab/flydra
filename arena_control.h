#ifndef _flydra_ARENA_CONTROL_h_
#define _flydra_ARENA_CONTROL_h_

long arena_initialize( void );
void arena_finish( void );

/* save data points for nframes, then calculate center of rotation from those data points */
long rotation_calculation_init( int nframes );
void rotation_calculation_finish( void );
void rotation_update( double fly_x_pos, double fly_y_pos, double new_orientation );

void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber );

#endif
