#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ippi.h"
#include "c_fit_params.h"

#define CHK( a ) (a == ippStsNoErr? 1:0)
#define MAX( a, b ) (a > b? a:b)

IppiMomentState_64f *pState;

void print_img2( float *img, int width, int height, int img_step )
{
  int i,j;
  float *ptr;

  for( i = 0, ptr = img; i < height; i++, ptr += img_step / sizeof( float ) )
  {
    for( j = 0; j < width; j++ )
      printf( "%4.0f", ptr[j] );
    printf( "\n" );
  }
}

int init_moment_state()
{
  if( !CHK( ippiMomentInitAlloc_64f( &pState, ippAlgHintFast ) ) )
  {
    printf( "failed allocating moment state\n" );
    return 1;
  }
  return 0;
}

int free_moment_state()
{
  if( !CHK( ippiMomentFree_64f( pState ) ) )
  {
    printf( "failed freeing moment state\n" );
    return 2;
  }
  return 0;
}

int fit_params( double *x0, double *y0, double *orientation,
                       int index_x, int index_y, int centroid_search_radius,
                       int width, int height, float *img, int img_step )
{
  int left,bottom,right,top;
  double Mu00, Mu10, Mu01;
  double Uu11, Uu20, Uu02;
  IppiSize roi_size;
  IppiPoint roi_offset;

  /* figure out ROI based on inputs */
  left = index_x - centroid_search_radius;
  if( left < 0 ) left = 0;
  right = index_x + centroid_search_radius;
  if( right >= width ) right = width - 1;
  bottom = index_y - centroid_search_radius;
  if( bottom < 0 ) bottom = 0;
  top = index_y + centroid_search_radius;
  if( top >= height ) top = height - 1;

  roi_offset.x = left;
  roi_offset.y = bottom;
  roi_size.width = right - left + 1;
  roi_size.height = top - bottom + 1;

  /* get moments */
  /* TODO: currently the ROI is tied to (0,0) */
  if( !CHK( ippiMoments64f_32f_C1R( (Ipp32f*)img, img_step, roi_size, pState ) ) )
  {
    printf( "failed calculating moments\n" );
    return 11;
  }

  /* calculate center of gravity from spatial moments */
  if( !CHK( ippiGetSpatialMoment_64f( pState, 0, 0, 0, roi_offset, (Ipp64f*)&Mu00 ) ) )
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
  if( Mu00 != 0.f )
  {
    *x0 = Mu10 / Mu00;
    *y0 = Mu01 / Mu00;
  }
  else
  {
    *x0 = -1;
    *y0 = -1;
  }

  /* calculate blob orientation from central moments */
  if( !CHK( ippiGetCentralMoment_64f( pState, 1, 1, 0, (Ipp64f*)&Uu11 ) ) )
  {
    printf( "failed getting central moment 1 1\n" );
    return 31;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 2, 0, 0, (Ipp64f*)&Uu20 ) ) )
  {
    printf( "failed getting central moment 2 0\n" );
    return 32;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 0, 2, 0, (Ipp64f*)&Uu02 ) ) )
  {
    printf( "failed getting central moment 0 2\n" );
    return 33;
  }
  *orientation = 0.5 * atan2( 2*Uu11, Uu20 - Uu02 ); /* 90-degree ambiguity! */

  return 0;
}

int fit_params_once_float( double *x0, double *y0, double *orientation,
                       int width, int height, float *img )
{
  int i;
  int sts;
  int img_step;
  Ipp32f* new_img;

  new_img = ippiMalloc_32f_C1( width, height, &img_step );
  if( !new_img )
  {
    printf( "failed allocating memory\n" );
    return 101;
  }

  for( i = 0; i < height; i++ )
    memcpy( new_img + (img_step/sizeof( Ipp32f ))*i,
      img + width*i, width*sizeof( float ) );

  sts = init_moment_state();
  if( sts != 0 ) return sts;

  sts = fit_params( x0, y0, orientation, width/2, height/2,
    MAX( width, height ), width, height, new_img, img_step );
  if( sts != 0 ) return sts;

  sts = free_moment_state();
  if( sts != 0 ) return sts;

  ippiFree( new_img );

  return 0;
}

int fit_params_once_char( double *x0, double *y0, double *orientation,
                       int width, int height, unsigned char *img )
{
  int i;
  int sts;
  int new_img_step, use_img_step;
  Ipp8u* new_img;
  Ipp32f* use_img;
  IppiSize roi = {width,height};

  new_img = ippiMalloc_8u_C1( width, height, &new_img_step );
  if( !new_img )
  {
    printf( "failed allocating memory 8u\n" );
    return 101;
  }

  for( i = 0; i < height; i++ )
    memcpy( new_img + new_img_step*i,
      img + width*i, width );

  use_img = ippiMalloc_32f_C1( width, height, &use_img_step );
  if( !use_img )
  {
    printf( "failed allocating memory 32f\n" );
    return 102;
  }

  if( !CHK( ippiConvert_8u32f_C1R( new_img, new_img_step, use_img, use_img_step, roi ) ) )
  {
    printf( "failed converting\n" );
    return 113;
  }

  sts = init_moment_state();
  if( sts != 0 ) return sts;

  sts = fit_params( x0, y0, orientation, width/2, height/2,
    MAX( width, height ), width, height, use_img, use_img_step );
  if( sts != 0 ) return sts;

  sts = free_moment_state();
  if( sts != 0 ) return sts;

  ippiFree( new_img );
  ippiFree( use_img );

  return 0;
}

