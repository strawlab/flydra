#emacs, this is -*-Python-*- mode

cdef extern from "c_fit_params.h":

    int init_moment_state()
    int free_moment_state()

    int fit_params( double *x0, double *y0, double *orientation,
                       int width, int height, unsigned char *img, int img_step )

    void start_center_calculation( int nframes )
    void end_center_calculation( double *x_center, double *y_center )
    void update_center_calculation( double new_x_pos, double new_y_pos, double new_orientation )

