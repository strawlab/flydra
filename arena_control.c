#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "c_fit_params.h"
#include "arena_control.h"
#include "arena_utils.h"

int serial_port;
int is_port_open = 0;
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
#if ARENA_CONTROL == OPEN_LOOP
  printf( "--open-loop mode\n" );
#else
  printf( "--closed-loop mode\n" );
#endif

  /* open serial port */
  if( !is_port_open )
  {
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval != SC_SUCCESS_RC )
    {
      printf( "error opening serial port\n" );
      return errval;
    }
    is_port_open = 1;
  }

  /* set pattern id */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_PATTERN;
  errval = sc_send_cmd( &serial_port, cmd, 3 );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error setting pattern\n" );
    return errval;
  }

  /* set initial position within pattern */
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = PATTERN_DEPTH-1;
  sc_send_cmd( &serial_port, cmd, 4 );

#if ARENA_CONTROL == OPEN_LOOP && BIAS_AVAILABLE == YES
  /* set gain and bias */
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = 0; cmd[3] = PATTERN_BIAS_X; /* x gain, bias */
  cmd[4] = 0; cmd[5] = PATTERN_BIAS_Y; /* y gain, bias */
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* close serial port */
  sc_close_port( &serial_port );
  is_port_open = 0;
#endif

  return 0;
}

/****************************************************************
** finish *******************************************************
****************************************************************/
void arena_finish( void )
{
  char cmd[8];

  if( datafile == 0 ) return;

  fclose( datafile );

#if ARENA_CONTROL == OPEN_LOOP && BIAS_AVAILABLE == YES
  /* open serial port */
  sc_open_port( &serial_port, SC_COMM_PORT );
  is_port_open = 1;

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );
#endif

  /* reset panels */
  cmd[0] = 2; cmd[1] = 1; cmd[2] = 0;
  sc_send_cmd( &serial_port, cmd, 3 );

  /* close serial port */
  sc_close_port( &serial_port );
  is_port_open = 0;

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
  if( !is_port_open )
  {
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval != SC_SUCCESS_RC )
    {
      printf( "error opening serial port\n" );
      return errval;
    }
    is_port_open = 1;
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

#if BIAS_AVAILABLE == YES
  /* set gain and bias */
  cmd[0] = 5; cmd[1] = 128;
  /* gain,bias as percentages, in 2s complement (x=x; -x=256-x) */
  cmd[2] = 0; cmd[3] = 30; /* x gain, bias */
  cmd[4] = 0; cmd[5] = 45; /* y gain, bias */
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* close serial port */
  sc_close_port( &serial_port );
  is_port_open = 0;
#endif

  return 0;
}

/****************************************************************
** rotation finish **********************************************
****************************************************************/
void rotation_calculation_finish( double new_x_cent, double new_y_cent )
{
  long errval;
  char cmd[8];

#if BIAS_AVAILABLE == YES
  /* open serial port */
  if( !is_port_open )
  {
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval == 0 ) is_port_open = 1;
  }

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );
#endif

  /* set pattern id to expt. pattern */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_PATTERN;
  errval = sc_send_cmd( &serial_port, cmd, 3 );

  /* set initial position within pattern */
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = PATTERN_DEPTH-1;
  sc_send_cmd( &serial_port, cmd, 4 );

#if ARENA_CONTROL == OPEN_LOOP && BIAS_AVAILABLE == YES
  /* set gain and bias */
  cmd[0] = 5; cmd[1] = 128;
  cmd[2] = 0; cmd[3] = PATTERN_BIAS_X;
  cmd[4] = 0; cmd[5] = PATTERN_BIAS_Y;
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* close serial port */
  sc_close_port( &serial_port );
  is_port_open = 0;
#endif

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
  long errval;
  char cmd[8];
  static int update = 1;

  new_pos_x_f -= 0.20; /* counterclockwise turn */
  new_pos_y_f += 0.35;
  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );

  /* ensure serial port is open */
  if( !is_port_open )
  {
    printf( "**found serial port closed in calculation\n" );
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval == 0 ) is_port_open = 1;
    else printf( "**failed opening serial port!\n" );
  }

  /* set pattern position */
  cmd[0] = 3; cmd[1] = 112;
  cmd[2] = new_pos_x; cmd[3] = new_pos_y;
  if( update == 1 && is_port_open ) sc_send_cmd( &serial_port, cmd, 4 );  
  update++;
  if( update > 2 ) update = 0;
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
{
  int new_pos_x, new_pos_y;
  char cmd[8];
  static double new_pos_x_f = 0.0, new_pos_y_f = 0.0;
  long errval;
  static int update = 1;
  static long firstframe = 0;
  static double last_orientation;
  static int exp_flag = 0;
  double theta_exp;

  /* experimental variables */
  static long ncalls = 0;
  const int n_calls_per_set = 101* 20; /* 101 Hz * n seconds */
  const int n_sets = 4;
  /* positive is clockwise in arena */
  static int cur_set = -1;
  static int expanding = 0;
  const double expansion_rate = 0.25; /* y pos step per frame */

  if( firstframe == 0 )
  {
    firstframe = framenumber;
    last_orientation = orientation;
  }

  /* disambiguate fly's orientation using position data */
  theta_exp = disambiguate( x, y, center_x, center_y );
  /* change to best match expected angle */
  while( orientation < theta_exp - PI/4 ) orientation += PI/2;
  while( orientation >= theta_exp + PI/4 ) orientation -= PI/2;

  /* debug calibration during first set */
  if( cur_set < 0 )
    new_pos_x_f += orientation * NPIXELS/(2*PI);

  /* update experimental variables */
  ncalls++;
  if( ncalls > n_calls_per_set )
  {
    if( cur_set < 0 ) new_pos_y_f = 0.0;
    cur_set = get_random_set( cur_set, n_sets );
    expanding = 0;
    ncalls = 0;

    printf( "__current experiment: set %d\n", cur_set );
    if( framenumber > firstframe + 101*60*15 && !exp_flag )
    {
      printf( "__15 minutes\n" );
      exp_flag = 1;
    }

    /* set new pattern number */
    if( !is_port_open )
    {
      printf( "**found serial port closed!\n" );
      errval = sc_open_port( &serial_port, SC_COMM_PORT );
      if( errval == 0 ) is_port_open = 1;
      else printf( "**failed opening serial port!\n" );
    }
    cmd[0] = 2; cmd[1] = 3; cmd[2] = cur_set + 2;
    if( is_port_open ) sc_send_cmd( &serial_port, cmd, 3 );
  }
  else if( cur_set >= 0 && ncalls > n_calls_per_set/2 )
      /* halfway through set, time to expand square! */
  {
    if( expanding == 0 ) expanding = 1;
    else if( expanding == 1 )
    {
      new_pos_y_f += expansion_rate;
      if( new_pos_y_f >= (double)PATTERN_DEPTH - 0.5 ) /* stop expanding! */
      {
        new_pos_y_f = 0.0;
        expanding = 2;
      }
    }
    else if( expanding == 2 ) /* post-expansion x movement */
    {
    }
  }

  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );

#if ARENA_CONTROL == CLOSED_LOOP || BIAS_AVAILABLE == NO
  /* ensure serial port is open */
  if( !is_port_open )
  {
    printf( "**found serial port closed!\n" );
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval == 0 ) is_port_open = 1;
    else printf( "**failed opening serial port!\n" );
  }

  /* set pattern position */
  cmd[0] = 3; cmd[1] = 112;
  cmd[2] = new_pos_x; cmd[3] = new_pos_y;
  if( update == 1 && is_port_open ) sc_send_cmd( &serial_port, cmd, 4 );

  update++;
  if( update > 2 ) update = 0; /* don't send pattern position every time */
#endif

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\t%ld\t%d\t%d\n",
  framenumber, timestamp, x, y, orientation, new_pos_x, new_pos_y,
  ncalls, cur_set, cur_set+1 );
}
