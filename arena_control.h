#ifndef _flydra_ARENA_CONTROL_h_
#define _flydra_ARENA_CONTROL_h_

long arena_initialize( void );
void arena_finish( void );

void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber );

#endif
