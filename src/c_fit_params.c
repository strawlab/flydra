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
CFitParamsReturnType fit_params( IppiMomentState_64f *pState, double *x0, double *y0,
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
    return CFitParamsOtherError;
  }

  /* calculate center of gravity from spatial moments */
  if( !CHK( ippiGetSpatialMoment_64f( pState, 0, 0, 0, roi_offset, (Ipp64f*)Mu00 ) ) )
  {
    return CFitParamsOtherError;
  }
  if( !CHK( ippiGetSpatialMoment_64f( pState, 1, 0, 0, roi_offset, (Ipp64f*)&Mu10 ) ) )
  {
    return CFitParamsOtherError;
  }
  if( !CHK( ippiGetSpatialMoment_64f( pState, 0, 1, 0, roi_offset, (Ipp64f*)&Mu01 ) ) )
  {
    return CFitParamsOtherError;
  }
  if( *Mu00 != 0.f )
  {
    *x0 = Mu10 / *Mu00;
    *y0 = Mu01 / *Mu00;
    /* relative to ROI origin */
  }
  else
  {
    return CFitParamsZeroMomentError;
  }

  /* calculate blob orientation from central moments */
  if( !CHK( ippiGetCentralMoment_64f( pState, 1, 1, 0, (Ipp64f*)Uu11 ) ) )
  {
    return CFitParamsCentralMomentError;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 2, 0, 0, (Ipp64f*)Uu20 ) ) )
  {
    return CFitParamsCentralMomentError;
  }
  if( !CHK( ippiGetCentralMoment_64f( pState, 0, 2, 0, (Ipp64f*)Uu02 ) ) )
  {
    return CFitParamsCentralMomentError;
  }

  return CFitParamsNoError;
}
