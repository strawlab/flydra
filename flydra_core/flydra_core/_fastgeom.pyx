# emacs, this is -*-Python-*- mode
#cython: language_level=2

def make_ThreeTuple(a,b,c):
    return ThreeTuple((a,b,c))

cdef class ThreeTuple:
    def __init__(self,vals):
        cdef ThreeTuple tt
        if isinstance(vals,ThreeTuple):
            tt = vals
            self.a = tt.a
            self.b = tt.b
            self.c = tt.c
            return
        self.a, self.b, self.c = vals

    def __reduce__(self):
        """this allows ThreeTuple to be pickled"""
        args = (self.a, self.b, self.c)
        return (make_ThreeTuple, args)

    def __repr__(self):
        return 'ThreeTuple( (%f,%f,%f))'%((self.a),
                                         (self.b),
                                         (self.c))
    def __sub__(ThreeTuple self not None, ThreeTuple other not None):
        cdef ThreeTuple result
        result = ThreeTuple.__new__(ThreeTuple)
        result.a = self.a-other.a
        result.b = self.b-other.b
        result.c = self.c-other.c
        return result

#     def __iter__(self):
#         return __iter__((self.a, self.b, self.c))

    def __add__(ThreeTuple self not None, ThreeTuple other not None):
        cdef ThreeTuple result
        result = ThreeTuple.__new__(ThreeTuple)
        result.a = self.a+other.a
        result.b = self.b+other.b
        result.c = self.c+other.c
        return result

    def __richcmp__(ThreeTuple x not None,
                    ThreeTuple y not None,
                    int op):
        if op==2: # test for equality
            if ((x.a==y.a) and
                (x.b==y.b) and
                (x.c==y.c)):
                return True
            else:
                return False
        else:
            raise NotImplementedError("this comparison not supported")

    def __neg__(self):
        cdef ThreeTuple result
        result = ThreeTuple.__new__(ThreeTuple)
        result.a = -self.a
        result.b = -self.b
        result.c = -self.c
        return result

    def __mul__(x,y):
        cdef ThreeTuple tt
        cdef float other

        if isinstance(x,ThreeTuple):
            tt = x
            other = y
        else:
            tt = y
            other = x
        cdef ThreeTuple result
        result = ThreeTuple.__new__(ThreeTuple)
        result.a = other*tt.a
        result.b = other*tt.b
        result.c = other*tt.c
        return result

    def cross(self, ThreeTuple other not None):

        cdef ThreeTuple result
        result = ThreeTuple.__new__(ThreeTuple)
        result.a = self.b*other.c - self.c*other.b
        result.b = self.c*other.a - self.a*other.c
        result.c = self.a*other.b - self.b*other.a
        return result

    def dot(self, ThreeTuple other not None):
        return self.a*other.a + self.b*other.b + self.c*other.c

def make_PlueckerLine(u,v):
    return PlueckerLine(u,v)

cdef class PlueckerLine:
    def __init__(self, ThreeTuple u_ not None, ThreeTuple v_ not None):
        self.u = u_
        self.v = v_
    def __reduce__(self):
        """this allows PlueckerLine to be pickled"""
        args = (self.u, self.v)
        return (make_PlueckerLine, args)
    def to_hz(self):
        return (self.v.c, -self.v.b, self.u.a, self.v.a, -self.u.b, self.u.c)
    def __repr__(self):
        return 'PlueckerLine(%s,%s)'%(repr(self.u),repr(self.v)) #

#     property u:
#         def __get__(self):
#             return self.u

#     property v:
#         def __get__(self):
#             return self.v

    def __richcmp__(PlueckerLine x not None,
                    PlueckerLine y not None,
                    int op):
        if op==2: # test for equality
            if ((x.u==y.u) and
                (x.v==y.v)):
                return True
            else:
                return False
        else:
            raise NotImplementedError("this comparison not supported")

    def closest(self):
        cdef ThreeTuple VxU
        cdef float UdotU
        VxU = self.v.cross(self.u)
        UdotU = self.u.dot(self.u)
        return ThreeTuple((VxU.a/UdotU,
                           VxU.b/UdotU,
                           VxU.c/UdotU))
    def dist2(self):
        return self.v.dot(self.v) / self.u.dot(self.u)

    def translate(self, ThreeTuple x not None):
        cdef ThreeTuple on_line
        cdef ThreeTuple on_new_line_a
        cdef ThreeTuple on_new_line_b

        on_line = self.closest()
        on_new_line_a = on_line+x
        on_new_line_b = on_new_line_a + self.u
        return line_from_points(on_new_line_a,on_new_line_b)

def line_from_points(ThreeTuple p not None, ThreeTuple q not None):
    cdef PlueckerLine result
    result = PlueckerLine.__new__(PlueckerLine)
    result.u = p-q
    result.v = p.cross(q)
    return result

def line_from_HZline(P):
    u = ThreeTuple( (P[2], -P[4], P[5] ) )
    v = ThreeTuple( (P[3], -P[1], P[0] ) )
    return PlueckerLine(u,v)
