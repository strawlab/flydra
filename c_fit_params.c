/* $Id$ */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ippi.h"
#include "c_fit_params.h"

#define CHK( a ) (a == ippStsNoErr? 1:0)
#define MAX( a, b ) (a > b? a:b)

double *x_pos_calc, *y_pos_calc;
int curr_frame = -1;
FILE *calibfile;

/****************************************************************
** print_img2 ***************************************************
****************************************************************/
void print_img2( unsigned char *img, int width, int height, int img_step )
{
  int i,j;
  unsigned char *ptr;

  for( i = 0, ptr = img; i < height; i++, ptr += img_step )
  {
    for( j = 0; j < width; j++ )
      printf( "%4d", ptr[j] );
    printf( "\n" );
  }
}

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
** fit_params ***************************************************
****************************************************************/
int fit_params( IppiMomentState_64f *pState, double *x0, double *y0,
		double *Mu00,
		double *Uu11, double *Uu20, double *Uu02,
		int width, int height, unsigned char *img, int img_step )
{
  double Mu10, Mu01;
  IppiSize roi_size;
  IppiPoint roi_offset;

  roi_offset.x = 0; 
  roi_offset.y = 0;
  roi_size.width = width;
  roi_size.height = height;

  /* get moments */
  if( !CHK( ippiMoments64f_8u_C1R( (Ipp8u*)img, img_step, roi_size, pState ) ) )
  {
    printf( "failed calculating moments\n" );
    return 11;
  }

  /* calculate center of gravity from spatial moments */
  if( !CHK( ippiGetSpatialMoment_64f( pState, 0, 0, 0, roi_offset, (Ipp64f*)Mu00 ) ) )
  {
    printf( "failed getting spatial moment 0 0\n" );
    return 21;
  }
  if( !CHK( ippiGetSpatialMoment_64f( pState, 1, 0, 0, roi_offset, (Ipp64f*)&Mu10 ) ) )
  {
    printf( "failed getting spatial moment 1 0\n" );
    return 22;
  }
  if( !CHK( ippiGetSpatialMoment_64f( pState, 0, 1, 0, roi_offset, (Ipp64f*)&Mu01 ) ) )
  {
    printf( "failed getting spatial moment 0 1\n" );
    return 23;
  }
  if( *Mu00 != 0.f )
  {
    *x0 = Mu10 / *Mu00;
    *y0 = Mu01 / *Mu00;
    /* relative to ROI origin */
  }
  else
  {
    *x0 = -1; /* XXX These should really set nan. */
    *y0 = -1;
  }

#if 0
  /* square image for orientation calculation */
  if( !CHK( ippiSqr_8u_C1IRSfs( (Ipp8u*)img, img_step, roi_size, 5 ) ) )
  {
    printf( "failed squaring image\n" );
    return 60;
  }

  /* get moments again */
  if( !CHK( ippiMoments64f_8u_C1R( (Ipp8u*)img, img_step, roi_size, pState ) ) )
  {
    printf( "failed calculating moments 2\n" );
    return 61;
  }
#endif

  /* calculate blob orientation from central moments */
  if( !CHK( ippiGetCentralMoment_64f( pState, 1, 1, 0, (Ipp64f*)Uu11 ) ) )
  {
    return 31;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 2, 0, 0, (Ipp64f*)Uu20 ) ) )
  {
    return 32;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 0, 2, 0, (Ipp64f*)Uu02 ) ) )
  {
    return 33;
  }

  return 0;
}

/****************************************************************
** start_center_calculation *************************************
****************************************************************/
void start_center_calculation( int nframes )
{
  char timestring[64], filename[64];

  fill_time_string( timestring );
  sprintf( filename, "%scalib%s.dat", _c_FIT_PARAMS_data_prefix_, timestring );
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

