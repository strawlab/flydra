#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "serial_comm/serial_comm.h"
#include "c_fit_params.h"
#include "arena_control.h"

#define PI 3.14159
#define YES 1
#define NO 0

#define CLOSED_LOOP 0
#define OPEN_LOOP 1
#define ARENA_CONTROL CLOSED_LOOP

#undef ARENA_PATTERN
#if ARENA_CONTROL == OPEN_LOOP
  #define ARENA_PATTERN 1
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
  #define PATTERN_DEPTH 8.0
#endif

int serial_port;
int is_port_open = 0;
FILE *datafile;

double center_x = -1, center_y = -1;
int is_calculating = 0;

/****************************************************************
** initialize ***************************************************
****************************************************************/
long arena_initialize( void )
{
  char cmd[8], timestring[64], filename[64];
  long errval;

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
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = 0;
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

  /* set pattern id to 1 -- rotation of exp/cont poles */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = 1;
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

  is_calculating = 1;

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
  sc_send_cmd( &serial_port, cmd, 3 );

  /* set initial position within pattern */
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = 0;
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

  is_calculating = 0;
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

  if( !is_calculating ) return;

  new_pos_x_f += 0.20;
  if( new_pos_x_f > (double)NPIXELS ) new_pos_x_f -= (double)NPIXELS;
  else if( new_pos_x_f < 0.0 ) new_pos_x_f = 0.0;
  new_pos_x = new_pos_x_f - (int)new_pos_x_f >= 0.5? (int)new_pos_x_f+1 : (int)new_pos_x_f;

  new_pos_y_f += 0.35;
  if( new_pos_y_f > PATTERN_DEPTH ) new_pos_y_f -= PATTERN_DEPTH;
  else if( new_pos_y_f < 0.0 ) new_pos_y_f = 0.0;
  new_pos_y = new_pos_y_f - (int)new_pos_y_f >= 0.5? (int)new_pos_y_f+1 : (int)new_pos_y_f;

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
  static long ncalls = 0;
  
  /* experimental variables */
  int n_calls_per_set = 101*60; /* 1 minute */
  const int n_sets = 5;
  int pix_offset_set[5] = {4,12,20,28,36};
  static int cur_set = 0;

#if ARENA_CONTROL == CLOSED_LOOP
  #if 1
  /* disambiguate fly's orientation using position data */
  if( center_x != center_y ) /* have been changed from -1 */
  {
    if( x >= center_x && y >= center_y ) /* quadrant I */
    {
      while( orientation < 0 ) orientation += PI/2;
      while( orientation > PI/2 ) orientation -= PI/2;
    }
    else if( x < center_x && y >= center_y ) /* quadrant II */
    {
      while( orientation <= PI/2 ) orientation += PI/2;
      while( orientation > PI ) orientation -= PI/2;
    }
    else if( x < center_x && y < center_y ) /* quadrant III */
    {
      while( orientation <= PI ) orientation += PI/2;
      while( orientation >= 3*PI/2 ) orientation -= PI/2;
    }
    else /* quadrant IV */
    {
      while( orientation < 3*PI/2 ) orientation += PI/2;
      while( orientation >= 2*PI ) orientation -= PI/2;
    }

    /* set pattern position based on experimental variables */
    new_pos_x_f = NPIXELS * fabs( orientation/(2*PI) );
    new_pos_x_f += pix_offset_set[cur_set];

    new_pos_y_f = 0.0;

    ncalls++;
    if( ncalls > (cur_set + 1)*n_calls_per_set )
    {
      cur_set++;
      if( cur_set >= n_sets )
      {
        cur_set = 0;
        ncalls = 0;
      }
      printf( "__current experiment: set %d\n", cur_set );
    }
  }
  else /* no center found, wrap at 90 deg */
  {
    new_pos_x_f = (NPIXELS/4) * fabs( orientation/(PI/2) );
    new_pos_y_f = 0.0;
  }

  #else
  new_pos_x_f += 0.15;
  new_pos_y_f = 0.0;
  #endif /* 0 */

#else /* open-loop */
  #if BIAS_AVAILBLE == NO
  new_pos_x_f -= 0.25;
  new_pos_y_f += 0.15;
  #endif
#endif /* closed vs. open loop */

#if ARENA_CONTROL == CLOSED_LOOP || BIAS_AVAILABLE == NO
  if( !is_calculating )
  {
    /* ensure serial port is open */
    if( !is_port_open && !is_calculating )
    {
      printf( "**found serial port closed!\n" );
      errval = sc_open_port( &serial_port, SC_COMM_PORT );
      if( errval == 0 ) is_port_open = 1;
      else printf( "**failed opening serial port!\n" );
    }

    /* condition and round pattern position */
    while( new_pos_x_f > (double)NPIXELS ) new_pos_x_f -= (double)NPIXELS;
    while( new_pos_x_f < 0.0 ) new_pos_x_f += (double)NPIXELS;
    new_pos_x = new_pos_x_f - (int)new_pos_x_f >= 0.5? (int)new_pos_x_f+1 : (int)new_pos_x_f;
    while( new_pos_y_f > (double)PATTERN_DEPTH ) new_pos_y_f -= (double)PATTERN_DEPTH;
    while( new_pos_y_f < 0.0 ) new_pos_y_f += (double)PATTERN_DEPTH;
    new_pos_y = new_pos_y_f - (int)new_pos_y_f >= 0.5? (int)new_pos_y_f+1 : (int)new_pos_y_f;

    /* set pattern position */
    cmd[0] = 3; cmd[1] = 112;
    cmd[2] = new_pos_x; cmd[3] = new_pos_y;
    if( update == 1 && is_port_open ) sc_send_cmd( &serial_port, cmd, 4 );

    update++;
    if( update > 2 ) update = 0;
  }
#endif

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\t%ld\t%d\t%d\n",
  framenumber, timestamp, x, y, orientation, new_pos_x, new_pos_y,
  ncalls, cur_set, pix_offset_set[cur_set] );
}
