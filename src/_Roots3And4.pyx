import warnings

cdef extern from "Roots3And4.h":
    int SolveQuartic(double c[5], double s[4])

cdef double real_nonnegative_root_less_than(double p4, double p3, double p2, double p1, double p0, double maxval) except *:
    cdef double c[5]
    cdef double s[4]
    cdef int i, num, found
    cdef double si

    c[4] = p4
    c[3] = p3
    c[2] = p2
    c[1] = p1
    c[0] = p0

    num = SolveQuartic(c,s)

    found = 0
    for i in range(num):
        si = s[i]
        if si <= maxval and si >= 0:
            result = si
            if found != 0:
                raise ValueError('more than one valid root found')
            found = 1
    if found==0:
        # hmm, sometimes numerical round-off error gets us here. try again with eps.
        eps=1e-7
        for i in range(num):
            si = s[i]
            if si <= maxval + eps and si >= -eps:
                result = si
                if found != 0:
                    raise ValueError('more than one valid root found')
                found = 1
                warnings.warn('solution to quartic roots required compensation '
                              'for round-off error.')
        if found==0:
            raise ValueError('valid root not found')
        if result < 0:
            # non-negative only to do round-off, make zero
            result = 0
    return result
