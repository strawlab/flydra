/* $Id$ */
#ifndef _flydra_C_FIT_PARAMS_h_
#define _flydra_C_FIT_PARAMS_h_

#include "ippi.h"

typedef enum {
  CFitParamsNoError,
  CFitParamsZeroMomentError,
  CFitParamsOtherError,
  CFitParamsCentralMomentError
} CFitParamsReturnType;

/**********************************************************
* Function to find the center of gravity and orientation of the
* shape found in an image within a region of interest defined by:
*   (index_x +/- centroid_search_radius, index_y +/- ...)
* bounded by the rectangle with corners at (0,0) and (width,height),
* The "unsigned char" type is assumed to be the same as IPP's "8u", and
* the image rows must be aligned to a 32-byte boundary (meaning
* that img_step should be a multiple of 32).  The parameter img_step
* is the number of bytes between the beginning of adjacent rows
* in the image pointer.
**********************************************************/
CFitParamsReturnType fit_params( IppiMomentState_64f *pState, double *x0, double *y0,
				 double *Mu00,
				 double *Uu11, double *Uu20, double *Uu02,
				 int width, int height, unsigned char *img, int img_step );

#endif
