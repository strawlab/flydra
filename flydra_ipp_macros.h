#include "ippdefs.h"

#include <stdio.h>

#define IMPOS8u(im,step,bot,left) ((im)+((bot)*(step))+(left))
#define IMPOS32f(im,step,bot,left) ((im)+((bot)*(step/4))+(left))
#define CHK( errval ) \
  if ( errval ) \
    { \
      fprintf(stderr,"IPP ERROR %d: %s in file %s (line %d)\n",errval,ippCoreGetStatusString(errval),__FILE__,__LINE__); \
      exit(1); \
    }
