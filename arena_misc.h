#define _ARENA_MISC_data_prefix_ "/home/jbender/data/"

/**********************************************************
* fill string with current time (pad with zeros)
**********************************************************/
void fill_time_string( char string[] );

/**********************************************************
* save data points for nframes, then calculate center of
* rotation from those data points
**********************************************************/
void start_center_calculation( int nframes );
void end_center_calculation( double *x_center, double *y_center );
void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation );
