#ifndef _flydra_C_FIT_PARAMS_h_
#define _flydra_C_FIT_PARAMS_h_

/**********************************************************
* Family of functions to take an arbitrary image and return
* the center of gravity and orientation of the image shape.
* If this will be called more than once, consider using
* fit_params() below.
**********************************************************/
/*================ may not be working!!!! =======================*/
int fit_params_once_float( double *x0, double *y0, double *orientation,
                       int width, int height, float *img );
int fit_params_once_char( double *x0, double *y0, double *orientation,
                       int width, int height, unsigned char *img );
/*================ may not be working!!!! =======================*/

/**********************************************************
* Initialization of the IPP "moment state" structure, which
* is used locally in fit_params().  Call this before calling
* fit_params() for the first time.
**********************************************************/
int init_moment_state( void );

/**********************************************************
* Deallocation of the IPP "moment state" structure.  Call this
* after calling fit_params() for the last time.
**********************************************************/
int free_moment_state( void );

/**********************************************************
* Function to find the center of gravity and orientation of the
* shape found in an image within a region of interest defined by:
*   (index_x +/- centroid_search_radius, index_y +/- ...)
* bounded by the rectangle with corners at (0,0) and (width,height),
* The "unsigned char" type is assumed to be the same as IPP's "8u", and
* the image rows must be aligned to a 32-byte boundary (meaning
* that img_step should be a multiple of 32).  The parameter img_step
* is the number of bytes between the beginning of adjacent rows
* in the image pointer.  init_moment_state() MUST BE called before
* calling this function.
**********************************************************/
int fit_params( double *x0, double *y0, double *orientation,
                       int index_x, int index_y, int centroid_search_radius,
                       int width, int height, unsigned char *img, int img_step );

void start_center_calculation( int nframes );
void end_center_calcuation( double *x_center, double *y_center );
void update_center_calculation( double new_x_pos, double new_y_pos );

#endif
