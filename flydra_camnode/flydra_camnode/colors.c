#include <stdlib.h>
#include <string.h>
#include "colors.h"
#include <stdio.h>

#define CHK(errval) if ((errval)!=ippStsNoErr) {			\
    fprintf(stderr,"%s, line %d: ipp error: %d\n",__FILE__,__LINE__,(errval)); \
    return 1;								\
  }

int mono8_bggr_to_red_color( Ipp8u* im, int step, IppiSize sz, int color_range_1, int color_range_2, int color_range_3, int sat_thresh) {
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
    fprintf(stderr,"%s, line %d: error allocating image\n",__FILE__,__LINE__);
    return 1;
  }

  hue_im = ippiMalloc_8u_C3(sz.width, sz.height, &hue_step );
  if (hue_im==NULL) {
    fprintf(stderr,"%s, line %d: error allocating image\n",__FILE__,__LINE__);
    return 1;
  }

  lum_im = ippiMalloc_8u_C3(sz.width, sz.height, &lum_step );
  if (lum_im==NULL) {
    fprintf(stderr,"%s, line %d: error allocating image\n",__FILE__,__LINE__);
    return 1;
  }

  hls_im = ippiMalloc_8u_C3(sz.width, sz.height, &hls_step );
  if (hls_im==NULL) {
    fprintf(stderr,"%s, line %d: error allocating image\n",__FILE__,__LINE__);
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
				 color_range_3, 0, ippCmpGreater));


  CHK( ippiThreshold_LTValGTVal_8u_C1IR(hue_im, hue_step, sz,
					color_range_1, 255,
					color_range_2, 255));


  CHK( ippiThreshold_Val_8u_C1IR(hue_im, hue_step, sz,
				 255, 0, ippCmpLess));



  // extract lum channel, threshold

  CHK( ippiColorToGray_8u_C3C1R(hls_im, hls_step,
			       lum_im, lum_step, sz,
			       l_channel));

  CHK( ippiThreshold_Val_8u_C1IR(lum_im, lum_step, sz,
				 sat_thresh, 0, ippCmpLess));

  // mult lum and hue - the 1 is the scale factor
  CHK( ippiMul_8u_C1IRSfs(lum_im, lum_step,
		  hue_im, hue_step, sz, 8));

  CHK(ippiCopy_8u_C1R(hue_im, hue_step,
		      im, step, sz ));
  ippFree(color_im);
  ippFree(hue_im);
  ippFree(lum_im);
  ippFree(hls_im);
  return 0;
}
