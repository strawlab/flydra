#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "c_fit_params.h"
#include "arena_control.h"
#include "arena_utils.h"

int serial_port;
FILE *datafile;

double center_x = -1, center_y = -1;

/****************************************************************
** initialize ***************************************************
****************************************************************/
long arena_initialize( void )
{
  char cmd[8], timestring[64], filename[64];
  long errval;

  /* seed random number generator with current time */
  srand( time( NULL ) );

  /* open data file */
  fill_time_string( timestring );
  sprintf( filename, "%sfly%s.dat", _c_FIT_PARAMS_data_prefix_, timestring );
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
  sc_send_cmd( &serial_port, cmd, 4 );  do later in analog  */

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

  return 0;
}

/****************************************************************
** finish *******************************************************
****************************************************************/
void arena_finish( void )
{
  char cmd[8];

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
  char cmd[8];

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

  return 0;
}

/****************************************************************
** rotation finish **********************************************
****************************************************************/
void rotation_calculation_finish( double new_x_cent, double new_y_cent )
{
  long errval;
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

int get_random_set( int cur_set, int n_sets )
{
  int r = cur_set;
  while( r == cur_set )
    r = rand() % n_sets;
  return r;
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
  char cmd[8];
  static double new_pos_x_f = 0.0, new_pos_y_f = 0.0;
  static double first_time = 0.0;
  static double avg_frametime = 0.0;
  static int ncalls = 0;
  long errval;

  static int exp_flag = 0;
  double theta_exp;

  /* experimental variables */
  static int set_time = 0.0;
  const double n_seconds_per_set = 10.0;
  const int n_sets = 4;
  /* positive is clockwise in arena */
  static int cur_set = -1;
  static int expanding = 0;
  const double expansion_rate = 1000.0; /* deg/sec */

  if( first_time == 0.0 )
  {
    first_time = set_time = timestamp;
  }

  /* disambiguate fly's orientation using position data */
  theta_exp = disambiguate( x, y, center_x, center_y );
  /* change to best match expected angle */
  while( orientation < theta_exp - PI/4 ) orientation += PI/2;
  while( orientation >= theta_exp + PI/4 ) orientation -= PI/2;

  /* debug calibration during first set */
  if( cur_set < 0 )
  {
    new_pos_x_f = orientation * RAD2PIX;
    ncalls++;
  }

  /* update experimental variables */
  if( timestamp > set_time + n_seconds_per_set )
  {
    if( cur_set < 0 )
    {
      avg_frametime = (timestamp - set_time) / ncalls;
      printf( "__avg frametime %.2lf ms (%.2f Hz)\n", avg_frametime*1000.0, 1/avg_frametime );
    }
    new_pos_y_f = 0.0;
    cur_set = get_random_set( cur_set, n_sets );
    expanding = 0;
    set_time = timestamp;

    printf( "__current experiment: set %d\n", cur_set );
    if( timestamp > first_time + 15.0*60.0 && !exp_flag )
    {
      printf( "__15 minutes\n" );
      exp_flag = 1;
    }

    /* open serial port */
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval != 0 ) printf( "**failed opening serial port!\n" );

    /* set new pattern number */
    cmd[0] = 2; cmd[1] = 3; cmd[2] = cur_set + 2;
    sc_send_cmd( &serial_port, cmd, 3 );

    /* set position within pattern */
/*    cmd[0] = 3; cmd[1] = 112; cmd[2] = new_pos_x; cmd[3] = new_pos_y;
    sc_send_cmd( &serial_port, cmd, 4 ); */

    /* start pattern */
/*    cmd[0] = 1; cmd[2] = 32;
    sc_send_cmd( &serial_port, cmd, 2 ); */

    /* close serial port */
    sc_close_port( &serial_port );
  }
  else if( cur_set >= 0 && timestamp > set_time + n_seconds_per_set/2 )
      /* halfway through set, time to expand square! */
  {
    if( expanding == 0 ) expanding = 1;
    else if( expanding == 1 ) /* do y expansion */
    {
      /* assume avg_frametime seconds between this frame and the next one */
      /* deg/sec * sec * pixels/deg = pixels */
      new_pos_y_f += expansion_rate * avg_frametime * DEG2PIX;
      /* assuming one-pixel expansion per pattern y shift */
      /* patterns expand a total of 78.75 deg (in each of x and y), so at 100 Hz
         and expanding at 1 patt per frame, expansion is 984.375 deg/sec */

      if( new_pos_y_f >= (double)PATTERN_DEPTH - 0.5 ) /* stop expanding! */
      {
        expanding = 2;
      }
    }
    else if( expanding == 2 ) /* post-expansion x movement */
    {
    }
  }
  else if( cur_set >= 0 && expanding == 0 ) /* pre-expansion x movement */
  {
    /* gently try to get square in front of fly */
    if( orientation > PI/4 && orientation < 7*PI/4 )
    {
    }
    else if( orientation <= PI/4 )
    {
    }
    else if( orientation >= 7*PI/4 )
    {
    }
  }

  /* set pattern position */
  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );
  set_position_analog( new_pos_x, new_pos_y );
  /* kind of stupid to round and then convert to analog, but it's digital
     again on the control board -- which is also stupid, since it would be
     much more efficient to simply give the control board this digital value,
     but that's not the way it was designed */

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\t%d\t%d\t%d\n",
  framenumber, timestamp, x, y, orientation, new_pos_x, new_pos_y,
  cur_set, cur_set, cur_set+2 );
}
