/* $Id: c_fit_params.c 350 2005-01-23 00:30:37Z astraw $ */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "arena_misc.h"

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
** start_center_calculation *************************************
****************************************************************/
void start_center_calculation( int nframes )
{
  char timestring[64], filename[64];

  fill_time_string( timestring );
  sprintf( filename, "%scalib%s.dat", _ARENA_MISC_data_prefix_, timestring );
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

