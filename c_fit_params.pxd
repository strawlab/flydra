#emacs, this is -*-Python-*- mode

cdef extern from "c_fit_params.h":

    int init_moment_state()
    int free_moment_state()

    int fit_params( double *x0, double *y0, double *orientation,
                       int index_x, int index_y, int centroid_search_radius,
                       int width, int height, float *img, int img_step )
