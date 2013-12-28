cdef extern from "refraction.h":
    double find_fastest_path_fermat_(double n1, double n2, double z1, double h, double z2, double epsilon, double scale )

cpdef double find_fastest_path_fermat(double n1, double n2, double z1, double h, double z2, double epsilon, double scale ):
    return find_fastest_path_fermat_(n1,n2,z1,h,z2, epsilon, scale )
