#include "colors.h"

int mono8_bggr_to_red( char* base, int step, int height, int width ) {
  int row,col;
  char* destptr;
  char* srcptr;
  for (row=0; row< height; row++) {
    destptr = (base + row*step);
    srcptr = (base + ( ((row/2)*2)+1)*step)-1;
    for (col=0; col<width; col++) {
      if (col%2==0) {
	srcptr++;
	srcptr++;
      }
      *destptr = *srcptr;
      destptr++;
    }
  }
  return 0;
}
