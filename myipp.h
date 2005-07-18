typedef Ipp8u unsigned char;
typedef Ipp16u unsigned short;
typedef Ipp32u unsigned int;

typedef Ipp8s signed char;
typedef Ipp16s signed short;
typedef Ipp32s signed int;

typedef Ipp32f float;
typedef Ipp64f double;

typedef IppStatus int;
typedef IppiMomentState_64f void*;

struct _IppiSize {
  Ipp32s width;
  Ipp32s height;
};
typedef IppiSize _IppiSize;

struct _IppiPoint {
  Ipp32s x;
  Ipp32s y;
};
typedef IppiPoint _IppiPoint;

IppStatus ippiAbsDiff_8u_C1R(...);
IppStatus ippiMaxIndx_8u_C1R(...);
IppStatus ippiThreshold_Val_8u_C1IR(...);

IppStatus ippiMomentInitAlloc_64f(...);
IppStatus ippiMoments64f_8u_C1R(...);
IppStatus ippiGetSpatialMoment_64f(...);
IppStatus ippiGetCentralMoment_64f(...);
IppStatus ippiMomentFree_64f( IppiMomentState_64f* );

IppStatus ippiMalloc_8u_C1(...);
IppStatus ippiMalloc_32f_C1(...);
IppStatus ippiFree(void*);

IppStatus ippiSet_8u_C1R(...);
IppStatus ippiSet_32f_C1R(...);

IppStatus ippiAddWeighted_8u32f_C1IR(...);
IppStatus ippiConvert_32f8u_C1R(...);
IppStatus ippiCopy_8u_C1R(...);

enum IppCmpOp {
  ippCmpLess;
};

enum IppHintAlgorithm {
  ippAlgHintFast;
};
