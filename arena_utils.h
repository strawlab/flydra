#ifndef _flydra_ARENA_UTILS_h_
#define _flydra_ARENA_UTILS_h_

#define PI 3.14159265358979
#define DIST( x1,y1, x2,y2 ) sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )
#define YES 1
#define NO 0

#define BIAS_AVAILABLE NO

/* 1 poles
   2 square
   3 vert
   4 horiz
   5 diag */
#define CALIBRATION_PATTERN 1
#define ARENA_START_PATTERN 3

/* gain,bias in 2s complement (x=x; -x=256-x) */
/* bias seems to be a percentage */
#define EXP_GAIN_X 15
#define EXP_BIAS_X 0
#define EXP_GAIN_Y 120
#define EXP_BIAS_Y 0

#define CAL_GAIN_X 0
#define CAL_BIAS_X 30
#define CAL_GAIN_Y 0
#define CAL_BIAS_Y 45


#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)
#define RAD2PIX ((double)NPIXELS/(2.0*PI))
#define DEG2PIX ((double)NPIXELS/360.0)

#define PATTERN_DEPTH 8
#define PATTERN_START_ANGLE 33.75
#define PATTERN_END_ANGLE 112.5
#define PATTERN_DELTA_ANGLE (PATTERN_END_ANGLE - PATTERN_START_ANGLE)

#define cdi_DEVICE "/dev/comedi0"
#define cdi_SUBDEV 1
#define cdi_CHAN_X 0
#define cdi_CHAN_Y 1
#define cdi_MIN 2030
#define cdi_MAX 2995
#define cdi_MAX 2995
#define cdi_RANGE (cdi_MAX-cdi_MIN)
#define cdi_AREF AREF_GROUND

void unwrap( double *th1, double *th2 );
double disambiguate( double x, double y, double center_x, double center_y );
void round_position( int *pos_x, double *pos_x_f, int *pos_y, double *pos_y_f );

void init_analog_output( void );
void finish_analog_output( void );
void set_position_analog( int pos_x, int pos_y );

#endif
