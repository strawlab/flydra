#include <math.h>
#include "crosen.h"

/*

  This code calculates rays according to [Fermat's principle of least
  time](http://en.wikipedia.org/wiki/Fermat's_principle). Light
  traveling from point 1 to point 2 (or vice-versa) takes the path of
  least time.

   1--
      \---                      medium 1
          \----
               \---
                   \----
                        \---
                            \----
                                 \---
   ====================================0====== interface
                                        \
                                         \
                                          \
        medium 2                           \
                                            \
                                             2


  n1: refractive index of medium 1
  n2: refractive index of medium 2
  z1: height of point 1
  z2: depth  of point 2
  h1: horizontal distance between points 1,0
  h2: horizontal distance between points 2,0
  h:  horizontal distance between points 1,2

 */

double duration(double x[], double e[]) {
  double n1, n2, z1, h, z2, h1, h2, result;

  h1 = x[0];

  n1 = e[0];
  n2 = e[1];
  z1 = e[2];
  h  = e[3];
  z2 = e[4];

  h2 = h-h1;

  result = n1*sqrt(h1*h1 + z1*z1) + n2*sqrt(z2*z2 + h2*h2);

  return result;
}

double find_fastest_path_fermat_(double n1, double n2, double z1, double h, double z2, double epsilon, double scale ) {
  double h1[] = {h};
  double args[] = {n1,n2,z1,h,z2};
  simplex(duration,h1,1,epsilon,scale,args);
  return h1[0];
}
