#ifndef _flydra_ARENA_UTILS_h_
#define _flydra_ARENA_UTILS_h_

#define _ARENA_CONTROL_data_prefix_ "/home/jbender/data/"

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
#define ARENA_START_PATTERN 2

/* gain,bias in 2s complement (x=x; -x=256-x) */
/* bias seems to be a percentage */
#define EXP_GAIN_X 15
#define EXP_BIAS_X 0
#define EXP_GAIN_Y 60
#define EXP_BIAS_Y 0

/*#define CAL_GAIN_X 0
#define CAL_BIAS_X 30
#define CAL_GAIN_Y 0
#define CAL_BIAS_Y 45 */
#define CAL_GAIN_X 15
#define CAL_BIAS_X 0
#define CAL_GAIN_Y 120
#define CAL_BIAS_Y 0


#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)
#define RAD2PIX ((double)NPIXELS/(2.0*PI))
#define DEG2PIX ((double)NPIXELS/360.0)
#define PIX2RAD (2.0*PI/(double)NPIXELS)
#define PIX2DEG (360.0/(double)NPIXELS)

#define CALIB_PATTERN_DEPTH 8
#if 0
  #define PATTERN_DEPTH 8
  #define PATTERN_START_ANGLE 33.75
  #define PATTERN_END_ANGLE 112.5
#else
  #define ARENA_PATTERN_DEPTH 16
  #define PATTERN_START_ANGLE (2*PIX2DEG)
  #define PATTERN_END_ANGLE (32*PIX2DEG)
#endif
#define PATTERN_DELTA_ANGLE (PATTERN_END_ANGLE - PATTERN_START_ANGLE)

#define cdi_DEVICE "/dev/comedi0"
#define cdi_SUBDEV 1
#define cdi_CHAN_X 0
#define cdi_CHAN_Y 1
#define cdi_MIN 2030
#define cdi_MAX 2990
#define cdi_RANGE (cdi_MAX-cdi_MIN)
#define cdi_AREF AREF_GROUND

void fill_time_string( char string[] );

/* save data points for nframes, then calculate center of rotation from those data points */
void start_center_calculation( int nframes );
void end_center_calculation( double *x_center, double *y_center );
void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation );

/* some utility functions */
void unwrap( double *th1, double *th2 );
double disambiguate( double x, double y, double center_x, double center_y );
void round_position( int *pos_x, double *pos_x_f, int *pos_y, double *pos_y_f, int max_x, int max_y );

/* interface to analog output */
void init_analog_output( void );
void finish_analog_output( void );
void set_position_analog( int pos_x, int max_x, int pos_y, int max_y );

#endif
