#include <stdlib.h>
#include "colors.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

int mono8_bggr_to_red_channel( unsigned char* base, int step, int height, int width ) {
  int row,col;
  unsigned char* destptr;
  unsigned char* srcptr;
  for (row=0; row< height; row++) {
    destptr = (base + row*step);
    srcptr = (base + ( (row/2)*2)*step);
    for (col=0; col<width; col++) {
      *destptr = *srcptr;
      destptr++;
      if (col%2==0) {
	srcptr++;
	srcptr++;
      }
    }
  }
  return 0;
}

int mono8_bggr_to_red_color( unsigned char* base, int step, int height, int width ) {
  int row,col;
  unsigned char* destptr;
  unsigned char* redsrcptr;
  unsigned char* greensrcptr;
  unsigned char* bluesrcptr;
  unsigned char red, green, blue;
  unsigned char* tmp;
  float bluegreen;
  float red_contrast;
  
  tmp = malloc( height*step );
  if (tmp==NULL) {
    return 1;
  }
  for (row=0; row< height; row++) {
    destptr = (tmp + row*step);
    bluesrcptr = (base + ( ((row/2)*2)+1)*step)-1;
    greensrcptr = (base + ( ((row/2)*2)+1)*step);
    redsrcptr = (base + ( (row/2)*2)*step);
    for (col=0; col<width; col++) {
      if (col%2==0) {
	bluesrcptr++;
	bluesrcptr++;
      }

      blue = *bluesrcptr;
      red = *redsrcptr;
      green = *greensrcptr;

      bluegreen = MAX(blue,green);
      if (red > bluegreen) {
	red_contrast = (red-bluegreen) / (red+bluegreen) * 255.0f;
      } else {
	red_contrast = 0.0f;
      }

      *destptr = (unsigned char)red_contrast;
      destptr++;
      if (col%2==1) {
	greensrcptr++;
	greensrcptr++;
      }
      if (col%2==0) {
	redsrcptr++;
	redsrcptr++;
      }
    }
  }
  /* copy results onto original image */
  memcpy( (void*)base, (void*)tmp, height*step );
  free(tmp);
  return 0;
}
