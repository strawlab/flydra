#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "serial_comm/serial_comm.h"
#include "c_fit_params.h"
#include "arena_control.h"

#define PI 3.14159

#define CLOSED_LOOP 0
#define OPEN_LOOP 1
#define ARENA_CONTROL CLOSED_LOOP

#undef ARENA_PATTERN
#if ARENA_CONTROL == OPEN_LOOP
  #define ARENA_PATTERN 1
  /* gain,bias as percentages, in 2s complement (x=x; -x=256-x) */
  #define PATTERN_BIAS_X 226
  #define PATTERN_BIAS_Y 15
#elif ARENA_CONTROL == CLOSED_LOOP
  #define ARENA_PATTERN 3
#endif
/* else won't compile! */

#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)

#define NPIXELS_OFFSET 0
/* number of pixels to add from pixel 0 to orientation 0 (= facing up?) */

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

#if ARENA_CONTROL == OPEN_LOOP
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

#if ARENA_CONTROL == OPEN_LOOP
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

  /* open serial port */
  if( !is_port_open )
  {
    errval = sc_open_port( &serial_port, SC_COMM_PORT );
    if( errval == 0 ) is_port_open = 1;
  }

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* set pattern id to expt. pattern */
  cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_PATTERN;
  sc_send_cmd( &serial_port, cmd, 3 );

  /* set initial position within pattern */
  cmd[0] = 3; cmd[1] = 112; cmd[2] = 0; cmd[3] = 0;
  sc_send_cmd( &serial_port, cmd, 4 );

#if ARENA_CONTROL == OPEN_LOOP
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
** update *******************************************************
****************************************************************/
void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber )
{
  int new_pos_x, new_pos_y;
#if ARENA_CONTROL == CLOSED_LOOP
  char cmd[8];
  double new_pos_x_f;
  long errval;
  static int ncalls = 0;
  double o_orient = orientation;

  /* determine new position for pattern */
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
    new_pos_x_f = NPIXELS * fabs( orientation/(2*PI) ) + NPIXELS_OFFSET;
    new_pos_x = new_pos_x_f - (int)new_pos_x_f >= 0.5? (int)new_pos_x_f+1 : (int)new_pos_x_f;
    new_pos_y = 0;    
  }
  else /* no center found, wrap at 90 deg */
  {
    orientation += PI/2;
    new_pos_x_f = (NPIXELS/4) * fabs( orientation/(PI/2) ) + NPIXELS_OFFSET;
    new_pos_x = new_pos_x_f - (int)new_pos_x_f >= 0.5? (int)new_pos_x_f+1 : (int)new_pos_x_f;
    new_pos_y = 0;
  }

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

    /* set pattern position */
    cmd[0] = 3; cmd[1] = 112;
    cmd[2] = new_pos_x; cmd[3] = new_pos_y;
    if( ncalls == 0 && is_port_open ) sc_send_cmd( &serial_port, cmd, 4 );
  
    ncalls++;
    if( ncalls > 4 ) ncalls = 0;
      /* 4 works, 2 doesn't, at 19200 baud */
      /* 3 doesn't work at 57600 baud */
  }
#endif

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\n",
  framenumber, timestamp, x, y, o_orient, orientation, new_pos_x, new_pos_y );
}
