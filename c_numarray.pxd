# emacs, this is -*-Python-*- mode

# This code shamelessly stolen from PyTables, copyright (c) Francesc
# Alted

# Structs and functions from numarray
cdef extern from "numarray/numarray.h":

  ctypedef enum NumRequirements:
    NUM_CONTIGUOUS
    NUM_NOTSWAPPED
    NUM_ALIGNED
    NUM_WRITABLE
    NUM_C_ARRAY
    NUM_UNCONVERTED

  ctypedef enum NumarrayByteOrder:
    NUM_LITTLE_ENDIAN
    NUM_BIG_ENDIAN

  cdef enum:
    UNCONVERTED
    C_ARRAY

  ctypedef enum NumarrayType:
    tAny
    tBool	
    tInt8
    tUInt8
    tInt16
    tUInt16
    tInt32
    tUInt32
    tInt64
    tUInt64
    tFloat32
    tFloat64
    tComplex32
    tComplex64
    tObject
    tDefault
    tLong
  
# Declaration for the PyArrayObject
# This does not work with pyrex 0.8 and better anymore. It's worth
# analyzing what's going on.
  
  struct PyArray_Descr:
     int type_num, elsize
     char type

  #ctypedef class numarray.numarraycore.NumArray [object PyArrayObject]:
  # This does not work because NumArray is actually a python class
  # derived from the c extension class _numarray.
  # Thanks to Simon Burton for pointing out this. 2003-01-12
  ctypedef class numarray._numarray._numarray [object PyArrayObject]:
    # Compatibility with Numeric
    cdef char *data
    cdef int nd
    cdef int *dimensions, *strides
    cdef object base
    cdef PyArray_Descr *descr
    cdef int flags
    # New attributes for numarray objects
    cdef object _data         # object must meet buffer API */
    cdef object _shadows      # ill-behaved original array. */
    cdef int    nstrides      # elements in strides array */
    cdef long   byteoffset    # offset into buffer where array data begins */
    cdef long   bytestride    # basic seperation of elements in bytes */
    cdef long   itemsize      # length of 1 element in bytes */
    cdef char   byteorder     # NUM_BIG_ENDIAN, NUM_LITTLE_ENDIAN */
    cdef char   _aligned      # test override flag */
    cdef char   _contiguous   # test override flag */

  # The numarray initialization funtion
  void import_libnumarray()

# CharArray type
CharType = records.CharType

# Conversion tables from/to classes to the numarray enum types
toenum = {numarray.Bool:tBool,   # Boolean type added
          numarray.Int8:tInt8,       numarray.UInt8:tUInt8,
          numarray.Int16:tInt16,     numarray.UInt16:tUInt16,
          numarray.Int32:tInt32,     numarray.UInt32:tUInt32,
          numarray.Int64:tInt64,     numarray.UInt64:tUInt64,
          numarray.Float32:tFloat32, numarray.Float64:tFloat64,
          CharType:97   # ascii(97) --> 'a' # Special case (to be corrected)
          }

toclass = {tBool:numarray.Bool,  # Boolean type added
           tInt8:numarray.Int8,       tUInt8:numarray.UInt8,
           tInt16:numarray.Int16,     tUInt16:numarray.UInt16,
           tInt32:numarray.Int32,     tUInt32:numarray.UInt32,
           tInt64:numarray.Int64,     tUInt64:numarray.UInt64,
           tFloat32:numarray.Float32, tFloat64:numarray.Float64,
           97:CharType   # ascii(97) --> 'a' # Special case (to be corrected)
          }

# Define the CharType code as a constant
cdef enum:
  CHARTYPE = 97

# Functions from numarray API
cdef extern from "numarray/libnumarray.h":
  object NA_InputArray (object, NumarrayType, int)
  object NA_OutputArray (object, NumarrayType, int)
  object NA_IoArray (object, NumarrayType, int)
  object NA_New( void*,NumarrayType,int,...)
  object NA_NewArray(void* buffer, NumarrayType type, int ndim, ...)
