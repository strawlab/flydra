#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_utils.h"

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
/*  char cmd[8];
  long errval;
  int serial_port; */
  static double new_pos_x_f = 0.0, new_pos_y_f = 0.0;
  static double first_time = 0.0;
  static double avg_frametime = 0.0;
  static int ncalls = 0;

  static int exp_flag = 0;

  /* experimental variables */
  static int set_time = 0.0;
  const double n_seconds_per_set = 10.0;
  const int n_sets = 1;
  /* positive is clockwise in arena */
  static int cur_set = -1;
  static int expanding = 0;
  const double expansion_rate = 1000.0; /* deg/sec */

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
    new_pos_x_f = orientation * RAD2PIX;
    new_pos_y_f = 0.0;
    cur_set = 0;
    expanding = 0;
    set_time = timestamp;

    printf( "__current experiment: set %d\n", cur_set );
    if( timestamp > first_time + 15.0*60.0 && !exp_flag )
    {
      printf( "__15 minutes\n" );
      exp_flag = 1;
    }
  }
  else if( cur_set == 0 )
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
          new_pos_y_f = 2.0;
          expanding = 1;
        }
      break;
      /* yeah, yeah, it's a switch statement with only one case, so what? */
    } /* switch */
  } /* if */

  /* set output variables */
  *patt_x = new_pos_x_f;
  *patt_y = new_pos_y_f;
  *out1 = expansion_rate;
  *out2 = (double)expanding;
  *out3 = (double)cur_set;
}
