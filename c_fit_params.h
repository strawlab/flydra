/* $Id$ */
#ifndef _flydra_C_FIT_PARAMS_h_
#define _flydra_C_FIT_PARAMS_h_

#include "ippi.h"

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
int fit_params( IppiMomentState_64f *pState, double *x0, double *y0,
		double *Mu00,
		double *Uu11, double *Uu20, double *Uu02,
		int width, int height, unsigned char *img, int img_step );

/**********************************************************
* fill string with current time (pad with zeros)
**********************************************************/
void fill_time_string( char string[] );

#define _c_FIT_PARAMS_data_prefix_ "/home/jbender/data/"

/**********************************************************
* save data points for nframes, then calculate center of
* rotation from those data points
**********************************************************/
void start_center_calculation( int nframes );
void end_center_calculation( double *x_center, double *y_center );
void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation );

#endif
