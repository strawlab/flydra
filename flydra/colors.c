#include "colors.h"

int mono8_bggr_to_red_channel( unsigned char* base, int step, int height, int width ) {
  int row,col;
  unsigned char* destptr;
  unsigned char* srcptr;
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

int mono8_bggr_to_red_color( unsigned char* base, int step, int height, int width ) {
  int row,col;
  char* destptr;
  char* redsrcptr;
  char* greensrcptr;
  for (row=0; row< height; row++) {
    destptr = (base + row*step);
    redsrcptr = (base + ( ((row/2)*2)+1)*step)-1;
    greensrcptr = (base + ( ((row/2)*2)+1)*step);
    for (col=0; col<width; col++) {
      if (col%2==0) {
	redsrcptr++;
	redsrcptr++;
      }
      if (*redsrcptr > *greensrcptr) {
	*destptr = (*redsrcptr - *greensrcptr);
      } else {
	*destptr = 0;
      }
      destptr++;
      if (col%2==0) {
	greensrcptr++;
	greensrcptr++;
      }
    }
  }
  return 0;
}
