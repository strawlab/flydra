#include "ippdefs.h"

#include <stdio.h>

#define IMPOS8u(im,step,bot,left) ((im)+((bot)*(step))+(left))
#define IMPOS32f(im,step,bot,left) ((im)+((bot)*(step/4))+(left))
#define CHK( errval ) \
  if ( errval ) \
    { \
      fprintf(stderr,"IPP ERROR %d: %s in file %s (line %d), exiting because I may not have GIL\n",errval,ippCoreGetStatusString(errval),__FILE__,__LINE__); \
      exit(1); \
    }

/* When we don't have Python GIL, we can't raise exception, so let's just quit */
#define SET_ERR( errval )						\
  {  printf("SET_ERR(%d) called, %s: %d\n",errval,__FILE__,__LINE__);	\
    exit(errval);							\
  }
