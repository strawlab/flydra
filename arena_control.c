#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_control.h"
#include "arena_feedback.h"
#include "arena_utils.h"

FILE *datafile;

int calcing = 0;
double center_x = -1, center_y = -1;

/****************************************************************
** initialize ***************************************************
****************************************************************/
long arena_initialize( void )
{
  char cmd[8], timestring[64], filename[64];
  int serial_port;
  long errval;

  /* seed random number generator with current time */
  srand( time( NULL ) );

  /* open data file */
  fill_time_string( timestring );
  sprintf( filename, "%sfly%s.dat", _ARENA_CONTROL_data_prefix_, timestring );
  datafile = fopen( filename, "w" );
  if( datafile == 0 )
  {
    printf( "error opening data file %s\n", filename );
    return 13;
  }
  printf( "--saving data to %s\n", filename );

  /* open analog output device */
  init_analog_output();

  /* open serial port */
  errval = sc_open_port( &serial_port, SC_COMM_PORT );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error opening serial port\n" );
    return errval;
  }

  /* set pattern id */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_START_PATTERN;
  errval = sc_send_cmd( &serial_port, cmd, 3 );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error setting pattern\n" );
    return errval;
  }

  /* set initial position within pattern */
/*  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = PATTERN_DEPTH-1;
  sc_send_cmd( &serial_port, cmd, 4 );  do later in analog */

  /* set gain and bias */
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = EXP_GAIN_X; cmd[3] = EXP_BIAS_X;
  cmd[4] = EXP_GAIN_Y; cmd[5] = EXP_BIAS_Y;
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* set initial position within pattern */
  set_position_analog( 0, 0 );

  /* close serial port */
  sc_close_port( &serial_port );

  return 0;
}

/****************************************************************
** finish *******************************************************
****************************************************************/
void arena_finish( void )
{
  char cmd[8];
  int serial_port;

  if( datafile == 0 ) return;

  /* close data file */
  fclose( datafile );

  /* close analog output device */
  finish_analog_output();

  /* open serial port */
  sc_open_port( &serial_port, SC_COMM_PORT );

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* reset panels */
  cmd[0] = 2; cmd[1] = 1; cmd[2] = 0;
  sc_send_cmd( &serial_port, cmd, 3 );

  /* close serial port */
  sc_close_port( &serial_port );

  printf( "--arena control finished\n" );
}

/****************************************************************
** rotation init ************************************************
****************************************************************/
long rotation_calculation_init( void )
{
  long errval;
  int serial_port;
  char cmd[8];

  if( calcing ) return 0.0;

  /* open serial port */
  errval = sc_open_port( &serial_port, SC_COMM_PORT );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error opening serial port\n" );
    return errval;
  }

  /* set pattern id to rotation of exp/cont poles */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = CALIBRATION_PATTERN;
  errval = sc_send_cmd( &serial_port, cmd, 3 );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error setting pattern\n" );
    return errval;
  }

  /* set initial position within pattern */
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = 0;
  sc_send_cmd( &serial_port, cmd, 4 );

  /* set gain and bias */
#if BIAS_AVAILABLE == YES
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = CAL_GAIN_X; cmd[3] = CAL_BIAS_X;
  cmd[4] = CAL_GAIN_Y; cmd[5] = CAL_BIAS_Y;
  sc_send_cmd( &serial_port, cmd, 6 );
#else
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = EXP_GAIN_X; cmd[3] = EXP_BIAS_X;
  cmd[4] = EXP_GAIN_Y; cmd[5] = EXP_BIAS_Y;
  sc_send_cmd( &serial_port, cmd, 6 );
#endif

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* close serial port */
  sc_close_port( &serial_port );

  calcing = 1;

  return 0;
}

/****************************************************************
** rotation finish **********************************************
****************************************************************/
void rotation_calculation_finish( double new_x_cent, double new_y_cent )
{
  long errval;
  int serial_port;
  char cmd[8];

  /* open serial port */
  errval = sc_open_port( &serial_port, SC_COMM_PORT );

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* set pattern id to expt. pattern */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_START_PATTERN;
  errval = sc_send_cmd( &serial_port, cmd, 3 );

  /* set gain and bias */
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = EXP_GAIN_X; cmd[3] = EXP_BIAS_X;
  cmd[4] = EXP_GAIN_Y; cmd[5] = EXP_BIAS_Y;
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* set initial position within pattern */
  set_position_analog( 0, PATTERN_DEPTH-1 );

  /* close serial port */
  sc_close_port( &serial_port );

  center_x = new_x_cent;
  center_y = new_y_cent;

  calcing = 0;
}

/****************************************************************
** rotation update **********************************************
****************************************************************/
void rotation_update( void )
{
#if BIAS_AVAILABLE == NO
  static double new_pos_x_f = 0.0, new_pos_y_f = 0.0;
  int new_pos_x, new_pos_y;

  new_pos_x_f -= 0.20; /* counterclockwise turn */
  new_pos_y_f += 0.35;
  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );
  set_position_analog( new_pos_x, new_pos_y );
#endif
}

/****************************************************************
** update *******************************************************
****************************************************************/
void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber )
/* x and y in pixels relative to region of interest origin */
/* orientation in radians with 90-degree ambiguity */
/* timestamp in seconds, not starting with zero */
/* framenumber since beginning of data collection (not zero first time this function is called) */
{
  int new_pos_x, new_pos_y;
  double new_pos_x_f, new_pos_y_f;
  double theta_exp;
  double out1, out2, out3;

  if( calcing ) return;

  /* disambiguate fly's orientation using position data */
  theta_exp = disambiguate( x, y, center_x, center_y );
  /* change to best match expected angle */
  while( orientation < theta_exp - PI/4 ) orientation += PI/2;
  while( orientation >= theta_exp + PI/4 ) orientation -= PI/2;

  /* find correct pattern position given current state */
  set_patt_position( orientation, timestamp, framenumber, &new_pos_x_f, 
    &new_pos_y_f, &out1, &out2, &out3 );

  /* set pattern position */
  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );
  set_position_analog( new_pos_x, new_pos_y );
  /* kind of stupid to round and then convert to analog, but it's digital
     again on the control board -- which is also stupid, since it would be
     much more efficient to simply give the control board this digital value,
     but that's not the way it was designed */

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\t%lf\t%lf\t%lf\n",
  framenumber, timestamp, x, y, orientation, new_pos_x, new_pos_y, out1, out2, out3 );
}
