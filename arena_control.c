#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "serial_comm/serial_comm.h"
#include "c_fit_params.h"
#include "arena_control.h"

#define PI 3.14159265358979
#define DIST( x1,y1, x2,y2 ) sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )
#define YES 1
#define NO 0

#define CLOSED_LOOP 0
#define OPEN_LOOP 1
#define ARENA_CONTROL CLOSED_LOOP

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
  #define ARENA_PATTERN 1
#endif
/* else won't compile! */

#define NPIXELS_PER_PANEL 8
#define NPANELS_CIRCUMFERENCE 8
#define NPIXELS (NPIXELS_PER_PANEL*NPANELS_CIRCUMFERENCE)
#if BIAS_AVAILABLE == NO
  #define PATTERN_DEPTH 8
#endif

int serial_port;
int is_port_open = 0;
FILE *datafile;

double center_x = -1, center_y = -1;

/****************************************************************
** unwrap *******************************************************
****************************************************************/
void unwrap( double *th1, double *th2 )
{
  /* make both positive */
  while( *th1 < 0 ) *th1 += 2*PI;
  while( *th2 < 0 ) *th2 += 2*PI;

  /* force th1 and th2 to be no more than PI apart */
  if( *th1 - *th2 > PI ) *th2 += 2*PI;
  else if( *th2 - *th1 > PI ) *th1 += 2*PI;
}

/****************************************************************
** disambiguate *************************************************
****************************************************************/
double disambiguate( double x, double y )
/* orientation is returned with a 90-degree ambiguity from c_fit_params() */
{
  double theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8;
  double th_dist12, th_dist34, th_dist56, th_dist78;
  double theta_f;

  /* find expected angle given center of mass */
  /* x = r*cos(t); y = r*sin(t) */
  theta1 = acos( (x - center_x) / DIST( x,y, center_x,center_y ) );
  theta2 = asin( (y - center_y) / DIST( x,y, center_x,center_y ) );
  unwrap( &theta1, &theta2 );
  th_dist12 = fabs( theta1 - theta2 );

  /* try different symmetries -- acos and asin are ambiguous, too */
  theta3 = 2*PI - theta1;
  theta4 = theta2;
  unwrap( &theta3, &theta4 );
  th_dist34 = fabs( theta3 - theta4 );

  theta5 = theta1;
  theta6 = PI - theta2;
  unwrap( &theta5, &theta6 );
  th_dist56 = fabs( theta5 - theta6 );

  theta7 = 2*PI - theta1;
  theta8 = PI - theta2;
  unwrap( &theta7, &theta8 );
  th_dist78 = fabs( theta7 - theta8 );

  /* average the best two possibilities */
  if( th_dist12 <= th_dist34 && th_dist12 <= th_dist56 && th_dist12 <= th_dist78 )
    theta_f = (theta1 + theta2) / 2;
  else if( th_dist34 <= th_dist56 && th_dist34 <= th_dist78 )
    theta_f = (theta3 + theta4) / 2;
  else if( th_dist56 < th_dist78 )
    theta_f = (theta5 + theta6) / 2;
  else theta_f = (theta7 + theta8) / 2;

  while( theta_f < 0 ) theta_f += 2*PI;
  while( theta_f >= 2*PI ) theta_f -= 2*PI;

  return theta_f;
}

/****************************************************************
** round_position ***********************************************
****************************************************************/
void round_position( int *pos_x, double *pos_x_f, int *pos_y, double *pos_y_f )
{
  *pos_x = *pos_x_f - (int)*pos_x_f >= 0.5? (int)*pos_x_f+1 : (int)*pos_x_f;
  while( *pos_x > NPIXELS )
  {
    *pos_x -= NPIXELS;
    *pos_x_f -= (double)NPIXELS;
  }
  while( *pos_x < 0 )
  {
    *pos_x += NPIXELS;
    *pos_x_f += (double)NPIXELS;
  }
  *pos_x = *pos_x_f - (int)*pos_x_f >= 0.5? (int)*pos_x_f+1 : (int)*pos_x_f;

  *pos_y = *pos_y_f - (int)*pos_y_f >= 0.5? (int)*pos_y_f+1 : (int)*pos_y_f;
  while( *pos_y >= PATTERN_DEPTH )
  {
    *pos_y -= PATTERN_DEPTH;
    *pos_y_f -= (double)PATTERN_DEPTH;
  }
  while( *pos_y < 0 )
  {
    *pos_y += PATTERN_DEPTH;
    *pos_y_f += (double)PATTERN_DEPTH;
  }
  *pos_y = *pos_y_f - (int)*pos_y_f >= 0.5? (int)*pos_y_f+1 : (int)*pos_y_f;
}

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
  cmd[0] = 1; cmd[1] = 0;
  sc_send_cmd( &serial_port, cmd, 2 );

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
  static long firstframe = 0;
  static double last_orientation;
  static int exp_flag = 0;
  double theta_exp;

  /* experimental variables */
  const int n_calls_per_set = 101*30; /* 30 sec */
  const int n_sets = 2;
  /* positive is clockwise in arena */
  static int cur_set = 0;

  if( firstframe == 0 )
  {
    firstframe = framenumber;
    last_orientation = orientation;
  }

  /* disambiguate fly's orientation using position data */
  theta_exp = disambiguate( x, y );
  /* change to best match expected angle */
  while( orientation < theta_exp - PI/4 ) orientation += PI/2;
  while( orientation >= theta_exp + PI/4 ) orientation -= PI/2;

  if( cur_set == 0 ) /* vis open loop */
    new_pos_x_f += (orientation - last_orientation) * NPIXELS/(2*PI);
  /* else do nothing -- vis closed loop */
  new_pos_y_f = 0.0;
  round_position( &new_pos_x, &new_pos_x_f, &new_pos_y, &new_pos_y_f );

  /* update experimental variables */
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
    if( framenumber > firstframe + 101*60*15 && !exp_flag )
    {
      printf( "__15 minutes\n" );
      exp_flag = 1;
    }
    /* set pattern number again for robustness' sake */
    cmd[0] = 2; cmd[1] = 3; cmd[2] = ARENA_PATTERN;
    if( is_port_open ) sc_send_cmd( &serial_port, cmd, 3 );
  }

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
  fprintf( datafile, "%ld\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%d\t%d\t%ld\t%d\t%.4lf\n",
  framenumber, timestamp, x, y, orientation, new_pos_x, new_pos_y,
  ncalls, cur_set, cur_set );
}
