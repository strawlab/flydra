#include <comedilib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "arena_utils.h"

comedi_t *cdi_dev = NULL;

double *x_pos_calc, *y_pos_calc;
int curr_frame = -1;
FILE *calibfile;

/****************************************************************
** fill_time string *********************************************
****************************************************************/
void fill_time_string( char string[] )
{
  time_t now;
  struct tm *timeinfo;

  /* fill string with current time (pad with zeros) */
  time( &now );
  timeinfo = localtime( &now );
  sprintf( string, "%d-", timeinfo->tm_year + 1900 );
  if( timeinfo->tm_mon + 1 < 10 ) strcat( string, "0" );
  sprintf( &string[strlen( string )], "%d-", timeinfo->tm_mon + 1 );
  if( timeinfo->tm_mday < 10 ) strcat( string, "0" );
  sprintf( &string[strlen( string )], "%d_", timeinfo->tm_mday );
  if( timeinfo->tm_hour < 10 ) strcat( string, "0" );
  sprintf( &string[strlen( string )], "%d-", timeinfo->tm_hour );
  if( timeinfo->tm_min < 10 ) strcat( string, "0" );
  sprintf( &string[strlen( string )], "%d-", timeinfo->tm_min );
  if( timeinfo->tm_sec < 10 ) strcat( string, "0" );
  sprintf( &string[strlen( string )], "%d", timeinfo->tm_sec );
}

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
double disambiguate( double x, double y, double center_x, double center_y )
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
  while( *pos_x >= NPIXELS )
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
** init_analog_output *******************************************
****************************************************************/
void init_analog_output( void )
{
  cdi_dev = comedi_open( cdi_DEVICE );
  if( cdi_dev == NULL )
  {
    printf( "**failed opening comedi device %s\n", cdi_DEVICE );
    comedi_perror( cdi_DEVICE );
  }
}

/****************************************************************
** finish_analog_output *****************************************
****************************************************************/
void finish_analog_output( void )
{
  comedi_close( cdi_dev );
  cdi_dev = NULL;
}

/****************************************************************
** set_position_analog ******************************************
****************************************************************/
void set_position_analog( int pos_x, int pos_y )
{
  lsampl_t ana_x, ana_y;

  if( cdi_dev == NULL ) return;

  ana_x = pos_x*cdi_RANGE/(NPIXELS-1) + cdi_MIN;
  comedi_data_write( cdi_dev, cdi_SUBDEV, cdi_CHAN_X, 0, cdi_AREF, ana_x );

  ana_y = pos_y*cdi_RANGE/(PATTERN_DEPTH-1) + cdi_MIN;
  comedi_data_write( cdi_dev, cdi_SUBDEV, cdi_CHAN_Y, 0, cdi_AREF, ana_y );

  /* in order for this to work with a single output range, the x and y gains must
     be set on the controlle  in a ratio consistent with their possible values here; 
     here; specifically, they must have a ratio equal to NPIXELS/PATTERN_DEPTH
     (currently gains are 120 and 15 for NPIXELS=64 and PATTERN_DEPTH=8) -- more
     accurately, instead of NPIXELS one should use the pattern width, but presumably
     that will always equal the number of possible pattern positions, NPIXELS */
}

/****************************************************************
** start_center_calculation *************************************
****************************************************************/
void start_center_calculation( int nframes )
{
  char timestring[64], filename[64];

  fill_time_string( timestring );
  sprintf( filename, "%scalib%s.dat", _ARENA_CONTROL_data_prefix_, timestring );
  calibfile = fopen( filename, "w" );

  printf( "==saving center calculation to %s\n", filename );

  x_pos_calc = (double*)malloc( nframes * sizeof( double ) );
  y_pos_calc = (double*)malloc( nframes * sizeof( double ) );
  curr_frame = 0;
}

/****************************************************************
** end_center_calculation ***************************************
****************************************************************/
void end_center_calculation( double *x_center, double *y_center )
{
  int i;
  double x_min = 9999.9, x_max = -1.0;
  double y_min = 9999.9, y_max = -1.0;

  /* mean could be skewed; instead use middle, assuming a circle */
  for( i = 0; i < curr_frame; i++ )
  {
    if( x_pos_calc[i] < x_min ) x_min = x_pos_calc[i];
    else if( x_pos_calc[i] > x_max ) x_max = x_pos_calc[i];
    if( y_pos_calc[i] < y_min ) y_min = y_pos_calc[i];
    else if( y_pos_calc[i] > y_max ) y_max = y_pos_calc[i];
  }
  *x_center = (x_max - x_min)/2 + x_min;
  *y_center = (y_max - y_min)/2 + y_min;

  free( x_pos_calc );
  free( y_pos_calc );
  curr_frame = -1;

  fclose( calibfile );
  printf( "==done calculating center %.4lf %.4lf\n", *x_center, *y_center );
}

/****************************************************************
** update_center_calculation ************************************
****************************************************************/
void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation )
{
  fprintf( calibfile, "%lf\t%lf\t%lf\n", new_x_pos, new_y_pos, new_orientation );

  x_pos_calc[curr_frame] = new_x_pos;
  y_pos_calc[curr_frame] = new_y_pos;
  curr_frame++;
}

