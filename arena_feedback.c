#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_utils.h"

int get_random_set( int cur_set, int n_sets )
{
  int r = cur_set;
  while( r == cur_set )
    r = rand() % n_sets;
  return r;
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
  char cmd[8];
  static double new_pos_x_f = 0.0, new_pos_y_f = 0.0;
  static double first_time = 0.0;
  static double avg_frametime = 0.0;
  static int ncalls = 0;
  long errval;
  int serial_port;

  static int exp_flag = 0;

  /* experimental variables */
  static int set_time = 0.0;
  const double n_seconds_per_set = 10.0;
  const int n_sets = 4;
  /* positive is clockwise in arena */
  static int cur_set = -1;
  static int expanding = 0;
  const double expansion_rate = 1000.0; /* deg/sec */

  double use_orientation;
  double use_pos_x;

  if( first_time == 0.0 )
  {
    first_time = set_time = timestamp;
  }

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
  else if( cur_set >= 0 )
  {
    switch( expanding )
    {
      case 0:
        /* assume avg_frametime seconds between this frame and the next one */
        /* deg/sec * sec * pixels/deg = pixels */
        new_pos_y_f += expansion_rate * avg_frametime * DEG2PIX;
        /* assuming one-pixel expansion per pattern y shift */
        /* patterns expand a total of 78.75 deg (in each of x and y), so at 100 Hz
           and expanding at 1 patt per frame, expansion is 984.375 deg/sec */

        if( new_pos_y_f >= (double)PATTERN_DEPTH - 0.5 ) /* stop expanding */
        {
          new_pos_y_f = (double)PATTERN_DEPTH - 1;
          expanding = 1;
        }
      break;
      case 1:
        if( timestamp > set_time + n_seconds_per_set/2  ) /* contraction time */
        {
          new_pos_y_f -= expansion_rate * avg_frametime * DEG2PIX;
          if( new_pos_y_f < 0.5 ) /* stop contracting */
          {
            new_pos_y_f = 0.0;
            expanding = 2;
          }
        }
      break;
      case 2:
        /* gently try to get square in front of fly */
        use_orientation = orientation;
        use_pos_x = new_pos_x_f / RAD2PIX; /* in radians */
        if( fabs( use_orientation - use_pos_x ) > PI ) /* unwrap */
          if( use_orientation < use_pos_x ) use_orientation += 2*PI;
          else use_pos_x += 2*PI;
        if( fabs( use_orientation - use_pos_x ) > PI/2 ) /* adjust x pos */
          if( use_orientation < use_pos_x ) new_pos_x_f -= 0.1;
          else new_pos_x_f += 0.1; /* gently! */
      break;
    } /* switch */
  } /* if */

  *patt_x = new_pos_x_f;
  *patt_y = new_pos_y_f;
  *out1 = expansion_rate;
  *out2 = (double)cur_set;
  *out3 = (double)cur_set + 2; /* pattern number */
}
