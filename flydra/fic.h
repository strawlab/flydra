#include <stdint.h>

typedef uint8_t Fic8u;
typedef float Fic32f;
typedef double Fic64f;

typedef enum {
  ficStsNoErr                         =  0,
  ficStsNotImplemented,
  ficStsOnlyContiguousDataSupported,
  ficStsShapeMismatch
} FicStatus;

typedef struct {
  int width;
  int height;
} FiciSize;

typedef struct {
  int dummy;
} ficMomentState_64f;

FicStatus ficMomentInitAlloc_64f( ficMomentState_64f** );
FicStatus ficMomentFree_64f( ficMomentState_64f* );

FicStatus ficiMinIndx_8u_C1R(const Fic8u* pSrc, const int srcStep,
                             const FiciSize roiSize, Fic8u* val,
                             int* x, int*y);
FicStatus ficiMaxIndx_8u_C1R(const Fic8u* pSrc, const int srcStep,
                             const FiciSize roiSize, Fic8u* val,
                             int* x, int*y);
FicStatus ficiMaxIndx_32f_C1R(const Fic32f* pSrc, const int srcStep,
                              const FiciSize roiSize, Fic32f* val,
                              int* x, int*y);
FicStatus ficiDotProd_8u64f_C1R(const Fic8u* pSrc1, const int src1Step,
                                const Fic8u* pSrc2, const int src2Step,
                                const FiciSize roiSize, Fic64f* result);
FicStatus ficiDotProd_32f64f_C1R(const Fic32f* pSrc1, const int src1Step,
                                 const Fic32f* pSrc2, const int src2Step,
                                 const FiciSize roiSize, Fic64f* result);

FicStatus ficiFilterSobelHoriz_32f_C1R( Fic32f *pSrc, int srcStep, Fic32f *pDst, int dstStep, FiciSize dstRoiSize );
FicStatus ficiFilterSobelVert_32f_C1R ( Fic32f *pSrc, int srcStep, Fic32f *pDst, int dstStep, FiciSize dstRoiSize );
FicStatus ficiFilterSobelHoriz_8u_C1R( Fic8u *pSrc, int srcStep, Fic8u *pDst, int dstStep, FiciSize dstRoiSize );
FicStatus ficiFilterSobelVert_8u_C1R ( Fic8u *pSrc, int srcStep, Fic8u *pDst, int dstStep, FiciSize dstRoiSize );
