#include <stdlib.h>
#include <string.h>
#include "colors.h"

#define CHK(errval) if ((errval)!=ippStsNoErr) return 1;

int mono8_bggr_to_red_color( Ipp8u* im, int step, IppiSize sz) {
  int color_step, hls_step, hue_step;
  Ipp8u *color_im, *hls_im, *hue_im;
  IppiRect rect;
  Ipp32f h_channel[3];

  rect.x = 0;
  rect.y = 0;
  rect.width = sz.width;
  rect.height = sz.height;

  h_channel[0] = 1;
  h_channel[1] = 0;
  h_channel[2] = 0;

  color_im = ippiMalloc_8u_C3(sz.width, sz.height, &color_step );
  if (color_im==NULL) {
    return 1;
  }

  hue_im = ippiMalloc_8u_C3(sz.width, sz.height, &hue_step );
  if (hue_im==NULL) {
    return 1;
  }

  hls_im = ippiMalloc_8u_C3(sz.width, sz.height, &hls_step );
  if (hls_im==NULL) {
    return 1;
  }

  CHK( ippiCFAToRGB_8u_C1C3R(im, rect, sz, step, 
			     color_im, color_step,
			     ippiBayerBGGR, 0));
  CHK( ippiRGBToHSV_8u_C3R(color_im, color_step, hls_im, hls_step, sz));
  CHK(ippiColorToGray_8u_C3C1R(hls_im, hls_step,
			       hue_im, hue_step, sz, 
			       h_channel));

  CHK(ippiCopy_8u_C1R(hue_im, hue_step,
		      im, step, sz ));
  ippiFree(color_im);
  ippiFree(hue_im);
  ippiFree(hls_im);
  return 0;
}
