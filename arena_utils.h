#ifndef _flydra_ARENA_UTILS_h_
#define _flydra_ARENA_UTILS_h_

#define PI 3.14159265358979
#define DIST( x1,y1, x2,y2 ) sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )
#define YES 1
#define NO 0

#define CLOSED_LOOP 0
#define OPEN_LOOP 1
#define ARENA_CONTROL CLOSED_LOOP
#define CALIBRATION_PATTERN 1
/* 1 poles
   2 square
   3 vert
   4 horiz
   5 diag */

#undef ARENA_PATTERN
#if ARENA_CONTROL == OPEN_LOOP
  #define ARENA_PATTERN 1
  /* pattern starting index is 1 */
  #define BIAS_AVAILABLE NO
  #if BIAS_AVAILABLE == YES
    /* gain,bias as percentages, in 2s complement (x=x; -x=256-x) */
    #define PATTERN_BIAS_X 226
    #define PATTERN_BIAS_Y 15
  #endif
#elif ARENA_CONTROL == CLOSED_LOOP
  #define ARENA_PATTERN 3
#endif
/* else won't compile! */

#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)
#if BIAS_AVAILABLE == NO
  #define PATTERN_DEPTH 8
#endif

/****************************************************************
** unwrap *******************************************************
****************************************************************/
void unwrap( double *th1, double *th2 );
double disambiguate( double x, double y, double center_x, double center_y );
void round_position( int *pos_x, double *pos_x_f, int *pos_y, double *pos_y_f );

#endif
