#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serial_comm/serial_comm.h"
#include "arena_utils.h"

/* full object w/ varying (const.) velocity, concenctric squares with const. velocity */

#define THETA_DEG_FROM_R_A_V_T( r, a, v, t ) ((2 * atan( r / (v*t) )) * 180/PI) /* assuming a=0 */
#define POS_Y_F_FROM_THETA( theta ) (theta * ((ARENA_PATTERN_DEPTH - 1.0)/180.0))
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
  char cmd[8];
  int serial_port;
  const double new_pos_x_f = 0.0;
  static double new_pos_y_f = 0.0;
  static double first_time = 0.0;
  static clock_t first_clock;

  static int exp_flag = 0;
  static int exp_frames = -1;
  static double avg_frametime;

  /* experimental variables */
  static double set_time = 0.0;
  const double n_seconds_per_set = 10.0;
  const int n_sets = 4;
  /* positive is clockwise in arena */
  static int cur_set = 0;
  static int expanding = 0;
  /* see /home/jbender/matlab/anal/calc_expanding_obj4.m */
  const double time_to_collision = -650; /* ms, initial */
  double theta;
  static double theta_init;
  const double r = 10.0		/100.0;		/* cm -> m */
/*  const double v = -1.5		/1000.0;	/* m/s -> m/ms */
  const double v_vals[3] = {-1.0,-2.0,-1.5};	/* m/s */
  static double v;
  const double a = 0.0		/1000000.0;	/* m/s^2 -> m/ms^2 */
  static double t;
  double x_calc;

  if( first_time == 0.0 )
  {
    first_time = timestamp;
    first_clock = clock();
    set_time = 0.0;
    srand( time( NULL ) );
  }

  exp_frames++;

  /* update experimental variables */
  if( timestamp > set_time + n_seconds_per_set )
  {
    cur_set = get_random_set( cur_set, n_sets );
    /* set this pattern */
    if( sc_open_port( &serial_port, SC_COMM_PORT ) == SC_SUCCESS_RC )
    {
      /* set pattern id to expt. pattern */
      if( cur_set < 2 )	{	cmd[0] = 2; cmd[1] = 3; cmd[2] = 2; /* full square */ }
      else {			cmd[0] = 2; cmd[1] = 3; cmd[2] = 3; /* conc. squares */ }
      sc_send_cmd( &serial_port, cmd, 3 );
      /* start pattern */
      cmd[0] = 1; cmd[1] = 32;
      sc_send_cmd( &serial_port, cmd, 2 );
      /* close serial port */
      sc_close_port( &serial_port );
    }
    else printf( "**error opening serial port\n" );
    
    t = time_to_collision;
    expanding = 0;
    if( cur_set < 2 ) v = v_vals[cur_set]; /* variable vel. */
    else v = v_vals[2]; /* old const vel. */
    v /= 1000.0; /* -> m/ms */
    theta = theta_init = THETA_DEG_FROM_R_A_V_T( r, a, v, t );
    new_pos_y_f = POS_Y_F_FROM_THETA( theta );

    if( set_time == 0.0 ) avg_frametime = 0.001780; /* estimate */
    else avg_frametime = (timestamp - set_time) / exp_frames;
    set_time = timestamp;
    exp_frames = -1;
    printf( "__current experiment: set %d\n", cur_set );
    if( !exp_flag && timestamp > first_time + 15.0*60.0 )
    {
      printf( "__15 minutes\n" );
      exp_flag = 1;
    }
  }
  else if( cur_set >= 0 )
  {
    switch( expanding )
    {
      case 0: /* expand */
        if( t >= 0.0 ) /* stop expanding */
        {
          theta = 180.0;
          new_pos_y_f = POS_Y_F_FROM_THETA( theta );
          t = 0.0;
          expanding++;
          break;
        }
        /* from /home/jbender/matlab/anal/calc_expanding_obj4.m
           for an object with radius 'r', inital velocity 'v', and constant acceleration 'a',
           position 'x'=0 at time 't'=0 and beginning at t<0

          % first find initial velocity at t=max(|t|)
          v0 = a*max( abs( t ) ) + v;
          % use this initial velocity at t=0, so v=v at t=max(|t|)
          theta = 2.*atan( r./(v0.*(t) + 0.5*a.*t.^2));
          x = r./tan(theta./2);
          v = [diff( x ) 0]; % cheap derivative
          theta = theta .* (180/pi); % deg
          phi = -r.*(v0+a.*t)./( t.^2 .*( v0+0.5*a.*t).^2 + r^2 );
          phi = phi .* (180/pi); % deg/s
        */

        theta = THETA_DEG_FROM_R_A_V_T( r, a, v, t );
        x_calc = v*t;

        /* deg * pixels/deg * pattern_index/pixels = pattern index */
        new_pos_y_f = POS_Y_F_FROM_THETA( theta );
//printf( "%.1f ", theta );

        /* increment time */
        t += avg_frametime * 1000.0;
        break;
      case 1:
        if( timestamp > set_time + n_seconds_per_set/2.0 ) expanding++;
      break;
      case 2: /* contract */
        if( t >= -time_to_collision ) /* stop contracting */
        {
          t = -time_to_collision;
          theta = THETA_DEG_FROM_R_A_V_T( r, a, -v, t );
          new_pos_y_f = POS_Y_F_FROM_THETA( theta );
          expanding++;
          break;
        }

        theta = THETA_DEG_FROM_R_A_V_T( r, a, -v, t );
        x_calc = -v*t;

        /* deg * pixels/deg * pattern_index/pixels = pattern index */
        new_pos_y_f = POS_Y_F_FROM_THETA( theta );

        /* increment time */
        t += avg_frametime * 1000.0;
      break;
    } /* switch */
  } /* if */

/*
  if( exp_frames == -1 ) printf( "  latency this frame: %.3lf ms\n", LATENCY_FROM_TIMESTAMP( timestamp ) );
*/

  /* set output variables */
  *patt_x = new_pos_x_f;
  *patt_y = new_pos_y_f;
  *out1 = theta;
  *out2 = t;
  *out3 = (double)cur_set;
}
