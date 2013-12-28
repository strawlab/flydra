cdef extern from "Roots3And4.h":
    int SolveQuartic(double c[5], double s[4])

def roots(double p4, double p3, double p2, double p1, double p0):
    cdef double c[5]
    cdef double s[4]
    cdef int i, num

    c[4] = p4
    c[3] = p3
    c[2] = p2
    c[1] = p1
    c[0] = p0

    num = SolveQuartic(c,s)
    result = [ s[i] for i in range(num) ]
    return result
