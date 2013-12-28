cdef extern from "Roots3And4.h":
    int SolveQuartic(double c[5], double s[4])

cdef double real_root_less_than(double p4, double p3, double p2, double p1, double p0, double maxval):
    cdef double c[5]
    cdef double s[4]
    cdef int i, num, found

    c[4] = p4
    c[3] = p3
    c[2] = p2
    c[1] = p1
    c[0] = p0

    num = SolveQuartic(c,s)

    found = 0
    for i in range(num):
        if s[i] <= maxval:
            result = s[i]
            if found != 0:
                raise ValueError('more than one valid root found')
            found = 1
    if found==0:
        raise ValueError('valid root not found')
    return result
