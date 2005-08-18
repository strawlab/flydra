#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_utils.h"

/* saccade-triggered rotations: striped background and single-stripe foreground */

#define LATENCY_FROM_TIMESTAMP( timestamp ) ((systime()-timestamp)*1000)

int get_random_set( int cur_set, int n_sets )
{
  int r = cur_set;
  while( r == cur_set )
    r = rand() % n_sets;
  return r;
}

double systime( void )
{
  struct timespec ts;
  clock_gettime( CLOCK_REALTIME, &ts );
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

/****************************************************************
** set_patt_position ********************************************
****************************************************************/
void set_patt_position( double orientation, double timestamp, long framenumber,
    double *patt_x, double *patt_y, double *out1, double *out2, double *out3 )
/* INPUT: */
/* orientation in radians */
/* timestamp in seconds, not starting with zero */
/* framenumber since beginning of data collection (not zero first time this function is called) */
/* OUTPUT: */
/* pattern x, y position */
/* three arbitrary output variables */
{
  static double new_pos_x_f = 0.0;
  static double new_pos_y_f = 0.0;
  static double first_time = 0.0;
  static clock_t first_clock;

  static int expt_flag = 0;

  /* experimental variables */
  static double trig_time = 0.0;
  const double n_seconds_wait_after_trigger = 5.0;
  const int n_sets = 9;
  /* positive is clockwise in arena */
  static int trig_direction;
  static int cur_set = 0;
  static double last_orientation;
  static int move_flag = 0;
  const double dor_thresh = 3.0; /* threshold, degrees */
  /* rotation 40 deg in 80 ms */
  const double rotation_amp = 40.0;
  const double rotation_dur = 80.0;
  static double start_pos_x, start_pos_y, end_pos_x, end_pos_y;
  double time_frac;

  if( first_time == 0.0 )
  {
    first_time = timestamp;
    first_clock = clock();
    srand( time( NULL ) );
  }

  /* update experimental variables */
  if( timestamp > trig_time + n_seconds_wait_after_trigger )
  {
    move_flag = 0;
    cur_set = get_random_set( cur_set, n_sets );
    printf( "__current experiment: set %d\n", cur_set );

    /* bring positions back into range, to keep things from getting out of hand */
    while( new_pos_x_f >= (double)NPIXELS ) new_pos_x_f -= (double)NPIXELS;
    while( new_pos_x_f < 0.0 ) new_pos_x_f += (double)NPIXELS;
    while( new_pos_y_f >= (double)NPIXELS ) new_pos_y_f -= (double)NPIXELS;
    while( new_pos_y_f < 0.0 ) new_pos_y_f += (double)NPIXELS;
  }
  else if( cur_set >= 0 )
  {
    if( !move_flag && fabs( orientation - last_orientation ) > dor_thresh*PI/180 )
    {
      move_flag = 1;
      trig_time = timestamp;
      trig_direction = (orientation > last_orientation? 1:-1);
      start_pos_x = new_pos_x_f;
      start_pos_y = new_pos_y_f;

      switch( cur_set )
      /* stripe rotates with pattern X, background with pattern Y */
      /* 0: rotate stripe with saccade
         1: rotate stripe against saccade
         2: rotate background with saccade
         3: rotate background against saccade
         4: stripe with, background with
         5: stripe with, background against
         6: stripe against, background with
         7: stripe against, background against
         8: no rotation */
      /* rotation 40 deg in 80 ms */
      {
        case( 0 ):
          end_pos_x = start_pos_x + trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y;
          break;
        case( 1 ):
          end_pos_x = start_pos_x - trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y;
          break;
        case( 2 ): 
          end_pos_x = start_pos_x;
          end_pos_y = start_pos_y + trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 3 ): 
          end_pos_x = start_pos_x;
          end_pos_y = start_pos_y - trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 4 ): 
          end_pos_x = start_pos_x + trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y + trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 5 ): 
          end_pos_x = start_pos_x + trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y - trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 6 ): 
          end_pos_x = start_pos_x - trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y + trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 7 ): 
          end_pos_x = start_pos_x - trig_direction*rotation_amp*DEG2PIX;
          end_pos_y = start_pos_y - trig_direction*rotation_amp*DEG2PIX;
          break;
        case( 8 ):
          end_pos_x = start_pos_x;
          end_pos_y = start_pos_y;
      } /* switch cur_set */
    } /* if saccade has just begun */

    if( move_flag ) /* mid-experiment */
    {
      time_frac = (timestamp - trig_time)*1000 / rotation_dur;
      if( time_frac > 1.0 ) time_frac = 1.0; /* done rotating */
      new_pos_x_f = start_pos_x + time_frac*(end_pos_x - start_pos_x);
      new_pos_y_f = start_pos_y + time_frac*(end_pos_y - start_pos_y);
    }
  } /* if */

/*
  if( exp_frames == -1 ) printf( "  latency this frame: %.3lf ms\n", LATENCY_FROM_TIMESTAMP( timestamp ) );
*/

  last_orientation = orientation;

  if( !expt_flag && timestamp > first_time + 15.0*60.0 )
  {
    printf( "__15 minutes\n" );
    expt_flag = 1;
  }

  /* set output variables */
  *patt_x = new_pos_x_f;
  *patt_y = new_pos_y_f;
  *out1 = 0.0;
  *out2 = 0.0;
  *out3 = (double)cur_set;
}
