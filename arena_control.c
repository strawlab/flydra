#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_control.h"

#define CLOSED_LOOP 0
#define OPEN_LOOP 1
#define ARENA_CONTROL OPEN_LOOP

#if ARENA_CONTROL == OPEN_LOOP
  #define ARENA_PATTERN 1
#else
  #define ARENA_PATTERN 1
#endif

#define DATA_PREFIX "/home/jbender/data/"

#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)

#define PI 3.14159

static int serial_port;
static FILE *datafile;

/****************************************************************
** initialize ***************************************************
****************************************************************/
long arena_initialize( void )
{
  char cmd[8], filename[64];
  time_t now;
  struct tm *timeinfo;

  /* open data file, name with current time (pad with zeros) */
  time( &now );
  timeinfo = localtime( &now );
  sprintf( filename, "%sfly%d-", DATA_PREFIX, timeinfo->tm_year + 1900 );
  if( timeinfo->tm_mon + 1 < 10 ) strcat( filename, "0" );
  sprintf( &filename[strlen( filename )], "%d-", timeinfo->tm_mon + 1 );
  if( timeinfo->tm_mday < 10 ) strcat( filename, "0" );
  sprintf( &filename[strlen( filename )], "%d_", timeinfo->tm_mday );
  if( timeinfo->tm_hour < 10 ) strcat( filename, "0" );
  sprintf( &filename[strlen( filename )], "%d-", timeinfo->tm_hour );
  if( timeinfo->tm_min < 10 ) strcat( filename, "0" );
  sprintf( &filename[strlen( filename )], "%d-", timeinfo->tm_min );
  if( timeinfo->tm_sec < 10 ) strcat( filename, "0" );
  sprintf( &filename[strlen( filename )], "%d.dat", timeinfo->tm_sec );
  datafile = fopen( filename, "w" );
  if( datafile == 0 )
  {
    printf( "error opening data file %s\n", filename );
    return 13;
  }
  printf( "--saving data to %s\n", filename );
#if ARENA_CONTROL == OPEN_LOOP
  printf( "--open-loop mode\n" );
#elif ARENA_CONTROL == CLOSED_LOOP
  printf( "--closed-loop mode\n" );
#else
  printf( "--no loop mode set!\n" );
  return 14;
#endif

  /* open serial port */
  long errval = sc_open_port( &serial_port, SC_COMM_PORT );
  if( errval != SC_SUCCESS_RC )
  {
    printf( "error opening serial port\n" );
    return errval;
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
  /* gain,bias as percentages, in 2s complement (x=x; -x=256-x) */
  cmd[2] = 0; cmd[3] = 226; /* x gain, bias */
  cmd[4] = 0; cmd[5] = 15; /* y gain, bias */
  sc_send_cmd( &serial_port, cmd, 6 );

  /* start pattern */
  cmd[0] = 1; cmd[1] = 32;
  sc_send_cmd( &serial_port, cmd, 2 );

  /* close serial port */
  sc_close_port( &serial_port );
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

  /* stop pattern */
  cmd[0] = 1; cmd[1] = 48;
  sc_send_cmd( &serial_port, cmd, 2 );
#endif

  /* reset panels */
  cmd[0] = 2; cmd[1] = 1; cmd[2] = 0;
  sc_send_cmd( &serial_port, cmd, 3 );

  /* close serial port */
  sc_close_port( &serial_port );
}

/****************************************************************
** update *******************************************************
****************************************************************/
void arena_update( double x, double y, double orientation,
    double timestamp, long framenumber )
{
  char cmd[8];
  int new_pos_x, new_pos_y;
  double new_pos_x_f;
  static int ncalls = 0;

#if ARENA_CONTROL == CLOSED_LOOP
  /* determine new position for pattern */
  orientation += PI/2;
  new_pos_x_f = (NPIXELS/2) * fabs( orientation/PI );
  new_pos_x = new_pos_x_f - (int)new_pos_x_f >= 0.5? (int)new_pos_x_f+1 : (int)new_pos_x_f;
    /* tried #defining this, but it didn't work */
  new_pos_y = 0;

  /* set pattern position */
  cmd[0] = 3; cmd[1] = 112;
  cmd[2] = new_pos_x; cmd[3] = 0;
  if( ncalls == 0 ) sc_send_cmd( &serial_port, cmd, 4 );
  
  ncalls++;
  if( ncalls > 3 ) ncalls = 0;
    /* 4 works, 2 doesn't, at 19200 baud */
    /* 3 doesn't work at 57600 baud */
#endif

  /* write data to file */
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\n", framenumber,
    timestamp, x, y, orientation, new_pos_x, new_pos_y );
}
