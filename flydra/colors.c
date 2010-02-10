#include <stdlib.h>
#include <string.h>
#include "colors.h"

#define CHK(errval) if ((errval)!=ippStsNoErr) return 1;

int mono8_bggr_to_red_color( Ipp8u* im, int step, IppiSize sz) {
  int color_step, hls_step, hue_step, lum_step;
  Ipp8u *color_im, *hls_im, *hue_im, *lum_im;
  IppiRect rect;
  Ipp32f h_channel[3];
  Ipp32f l_channel[3];

  rect.x = 0;
  rect.y = 0;
  rect.width = sz.width;
  rect.height = sz.height;

  // hue channel
  h_channel[0] = 1;
  h_channel[1] = 0;
  h_channel[2] = 0;

  // lum channel
  l_channel[0] = 0;
  l_channel[1] = 0;
  l_channel[2] = 1;

  color_im = ippiMalloc_8u_C3(sz.width, sz.height, &color_step );
  if (color_im==NULL) {
    return 1;
  }

  hue_im = ippiMalloc_8u_C3(sz.width, sz.height, &hue_step );
  if (hue_im==NULL) {
    return 1;
  }

  lum_im = ippiMalloc_8u_C3(sz.width, sz.height, &lum_step );
  if (lum_im==NULL) {
    return 1;
  }

  hls_im = ippiMalloc_8u_C3(sz.width, sz.height, &hls_step );
  if (hls_im==NULL) {
    return 1;
  }

  // debayer
  CHK( ippiCFAToRGB_8u_C1C3R(im, rect, sz, step,
			     color_im, color_step,
			     ippiBayerBGGR, 0));
  // convert to HSV
  CHK( ippiRGBToHLS_8u_C3R(color_im, color_step, hls_im, hls_step, sz));

  // extract hue channel, threshold
  CHK(ippiColorToGray_8u_C3C1R(hls_im, hls_step,
			       hue_im, hue_step, sz,
			       h_channel));



  CHK( ippiThreshold_Val_8u_C1IR(hue_im, hue_step, sz,
				 150, 0, ippCmpLess));

  CHK( ippiThreshold_Val_8u_C1IR(hue_im, hue_step, sz,
				 255, 255, ippCmpGreater));

  // extract lum channel, threshold

  CHK( ippiColorToGray_8u_C3C1R(hls_im, hls_step,
			       lum_im, lum_step, sz,
			       l_channel));

  CHK( ippiThreshold_Val_8u_C1IR(lum_im, lum_step, sz,
				 100, 0, ippCmpLess));

  //CHK( ippiThreshold_Val_8u_C1IR(lum_im, lum_step, sz,
  //				 100, 1, ippCmpGreater));


  // mult lum and hue - the 1 is the scale factor
  CHK( ippiMul_8u_C1IRSfs(lum_im, lum_step,
			  hue_im, hue_step, sz, 8));

  //CHK( ippiErode3x3_8u_C1IR( hue_im, hue_step, sz));


  CHK(ippiCopy_8u_C1R(hue_im, hue_step,
		      im, step, sz ));
  ippiFree(color_im);
  ippiFree(hue_im);
  ippiFree(lum_im);
  ippiFree(hls_im);
  return 0;
}
