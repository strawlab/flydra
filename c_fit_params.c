/* $Id$ */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ippi.h"
#include "c_fit_params.h"

#define CHK( a ) (a == ippStsNoErr? 1:0)

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
