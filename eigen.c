#include <math.h>

int eigen_2x2_real( double A, double B, 
		    double C, double D,
		    double *evalA, double *evecA1,
		    double *evalB, double *evecB1 )
/*

Consider the matrix M:

M = A B
    C D

The eigenvalues are:

evalA
evalB

The corresponding eigenvectors are

evecA = [ evecA1, 1 ]
evecB = [ evecB1, 1 ]

(Because the 2nd component of each eigenvectors is unity, they are not
returned by this function.)

*/

{
  double inside;

  if (C==0) { return -1; } /* will face divide by zero */
  inside = A*A + 4*B*C - 2*A*D + D*D;
  if (inside<0) { return -2; } /* complex answer */
  inside = sqrt(inside);
  *evalA = 0.5*( A + D - inside);
  *evalB = 0.5*( A + D + inside);
  
  *evecA1 = (-A+D+inside)/(-2*C);
  *evecB1 = (-A+D-inside)/(-2*C);
  
  return 0;
}
