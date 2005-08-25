#include <comedilib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "arena_utils.h"

comedi_t *cdi_dev = NULL;

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
** unwrap_pi4 ***************************************************
****************************************************************/
void unwrap_pi4( double *th1, double *th2 )
{
  /* make both positive */
  while( *th1 < 0 ) *th1 += 2*PI;
  while( *th2 < 0 ) *th2 += 2*PI;

  /* force th1 and th2 to be no more than PI/4 apart */
  if( *th1 - *th2 > PI/4 ) *th2 += PI/2;
  else if( *th2 - *th1 > PI/4 ) *th1 += PI/2;
}

/****************************************************************
** disambiguate *************************************************
****************************************************************/
double disambiguate( double x, double y, double center_x, double center_y )
/* orientation has an inherent 180-degree ambiguity */
{
  double theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8;
  double th_dist12, th_dist34, th_dist56, th_dist78;
  double theta_f;

  if( center_x == 0.0 || center_y == 0.0 ) return 0.0;

  /* find expected angle given center of mass */
  /* x = r*cos(t); y = r*sin(t) */
  theta1 = acos( (x - center_x) / DIST( x,y, center_x,center_y ) );
  theta2 = asin( (y - center_y) / DIST( x,y, center_x,center_y ) );
  unwrap( &theta1, &theta2 );
  th_dist12 = fabs( theta1 - theta2 );

  /* try different symmetries -- acos and asin are 90-degree ambiguous */
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

  /* average the best pair of possibilities */
  if( th_dist12 <= th_dist34 && th_dist12 <= th_dist56 && th_dist12 <= th_dist78 )
    theta_f = (theta1 + theta2) / 2;
  else if( th_dist34 <= th_dist56 && th_dist34 <= th_dist78 )
    theta_f = (theta3 + theta4) / 2;
  else if( th_dist56 < th_dist78 )
    theta_f = (theta5 + theta6) / 2;
  else theta_f = (theta7 + theta8) / 2;

  /* ensure it's on [0 2*pi) */
  while( theta_f < 0 ) theta_f += 2*PI;
  while( theta_f >= 2*PI ) theta_f -= 2*PI;

  return theta_f;
}

/****************************************************************
** round_position ***********************************************
****************************************************************/
void round_position( int *pos_x, double *pos_x_f, int *pos_y, double *pos_y_f, int max_x, int max_y )
{
  /* set int_x = round( double_x ) */
  if( *pos_x_f >= 0.0 )
    *pos_x = *pos_x_f - (int)*pos_x_f >= 0.5? (int)*pos_x_f+1 : (int)*pos_x_f;
  else
    *pos_x = *pos_x_f - (int)*pos_x_f <= -0.5? (int)*pos_x_f-1 : (int)*pos_x_f;

  /* make sure int_x is on [0 max_x), wrap double_x if int_x wraps */
  while( *pos_x >= max_x )
  {
    *pos_x -= max_x;
    *pos_x_f -= (double)max_x;
  }
  while( *pos_x < 0 )
  {
    *pos_x += max_x;
    *pos_x_f += (double)max_x;
  }

  if( *pos_y_f >= 0.0 )
    *pos_y = *pos_y_f - (int)*pos_y_f >= 0.5? (int)*pos_y_f+1 : (int)*pos_y_f;
  else
    *pos_y = *pos_y_f - (int)*pos_y_f <= -0.5? (int)*pos_y_f-1 : (int)*pos_y_f;
  while( *pos_y >= max_y )
  {
    *pos_y -= max_y;
    *pos_y_f -= (double)max_y;
  }
  while( *pos_y < 0 )
  {
    *pos_y += max_y;
    *pos_y_f += (double)max_y;
  }
}

/****************************************************************
** fit_circle ***************************************************
****************************************************************/
void fit_circle( double *x_data, double *y_data, int n_data, double *x_cent, double *y_cent )
{
  double sum_x = 0.0, sum_y = 0.0;
  double sum_sq_x = 0.0, sum_sq_y = 0.0;
  double a,b,c,d,e,f;
  int i;
  int use_n_data = n_data; /* some data may be 0 */

  /* least-squares fit of a circle to data */
/* from matlab fit_circle.m
for i=1:n,
	a = a + data(1,i) * (2*n*data(1,i) - 2*sum_x);
	b = b + data(2,i) * (2*n*data(1,i) - 2*sum_x);
	c = c + data(1,i) * (2*n*data(2,i) - 2*sum_y);
	d = d + data(2,i) * (2*n*data(2,i) - 2*sum_y);
	e = e + data(1,i) * (n*data(1,i)^2 + n*data(2,i)^2 - sum_sq_x - sum_sq_y);
	f = f + data(2,i) * (n*data(1,i)^2 + n*data(2,i)^2 - sum_sq_x - sum_sq_y);
end
x0 = (d*e - c*f) / (a*d - b*c);
y0 = (a*f - b*e) / (a*d - b*c);
r = sqrt( (1/n) * sum( (data(1,:) - x0).^2 + (data(2,:) - y0).^2 ) );
*/
  for( i = 0; i < n_data; i++ )
  {
    sum_x += x_data[i];
    sum_sq_x += x_data[i] * x_data[i];
    sum_y += y_data[i];
    sum_sq_y += y_data[i] * y_data[i];
    if( x_data[i] == 0.0 && y_data[i] == 0.0 ) use_n_data--;
  }

  a=b=c=d=e=f = 0.0;
  for( i = 0; i < n_data; i++ )
  {
    a += x_data[i] * (2*use_n_data*x_data[i] - 2*sum_x);
    b += y_data[i] * (2*use_n_data*x_data[i] - 2*sum_x);
    c += x_data[i] * (2*use_n_data*y_data[i] - 2*sum_y);
    d += y_data[i] * (2*use_n_data*y_data[i] - 2*sum_y);
    e += x_data[i] * (use_n_data*x_data[i]*x_data[i] + use_n_data*y_data[i]*y_data[i] - sum_sq_x - sum_sq_y);
    f += y_data[i] * (use_n_data*x_data[i]*x_data[i] + use_n_data*y_data[i]*y_data[i] - sum_sq_x - sum_sq_y);
  }

  if( a*d == b*c ) printf( "**error fitting circle -- divide by zero\n" );
  *x_cent = (d*e - c*f) / (a*d - b*c);
  *y_cent = (a*f - b*e) / (a*d - b*c);
  if( (unsigned long)*x_cent == (unsigned long)0x00000000 )
  {
    printf( "**warning: x center = nan\n" );
    *x_cent = 0.0;
  }
  if( (unsigned long)*y_cent == (unsigned long)0x00000000 )
  {
    printf( "**warning: y center = nan\n" );
    *y_cent = 0.0;
  }
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
void set_position_analog( int pos_x, int max_x, int pos_y, int max_y )
{
  lsampl_t ana_x, ana_y;

  if( cdi_dev == NULL ) return;

  ana_x = pos_x*cdi_RANGE/(max_x-1) + cdi_MIN;
  comedi_data_write( cdi_dev, cdi_SUBDEV, cdi_CHAN_X, 0, cdi_AREF, ana_x );

  ana_y = pos_y*cdi_RANGE/(max_y-1) + cdi_MIN;
  comedi_data_write( cdi_dev, cdi_SUBDEV, cdi_CHAN_Y, 0, cdi_AREF, ana_y );

  /* in order for this to work with a single output range, the x and y gains must
     be set on the controller in a ratio consistent with their possible values here; 
     in this case, they must have a ratio equal to max_x/max_y
     (for example, gains are 120 and 15 for NPIXELS=64 and PATTERN_DEPTH=8) */
}
