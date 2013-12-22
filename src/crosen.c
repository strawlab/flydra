/* 
 * Program: rosen.c
 * Author : Michael F. Hutt
 * http://www.mikehutt.com
 * 11/3/97
 * $Id: crosen.c,v 1.4 2007/07/10 12:42:02 mike Exp $
 *
 * Copyright (c) 1997-2004 <Michael F. Hutt>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * An implementation of the Nelder-Mead simplex method applied to
 * Rosenbrock's function.
 */


#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#define MAX_IT      1000      /* maximum number of iterations */
#define ALPHA       1.0       /* reflection coefficient */
#define BETA        0.5       /* contraction coefficient */
#define GAMMA       2.0       /* expansion coefficient */

double rosen(double x[])
{
  return (100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1.0-x[0])*(1.0-x[0]));
}


double simplex(double (*func)(double[]), double start[],int n, double EPSILON, double scale)
{
  
  int vs;        	/* vertex with smallest value */
  int vh;        	/* vertex with next smallest value */
  int vg;        	/* vertex with largest value */
  
  int i,j,m,row;
  int k;		/* track the number of function evaluations */
  int itr;		/* track the number of iterations */
  
  double **v;           /* holds vertices of simplex */
  double pn,qn;         /* values used to create initial simplex */
  double *f;            /* value of function at each vertex */
  double fr;            /* value of function at reflection point */
  double fe;            /* value of function at expansion point */
  double fc;            /* value of function at contraction point */
  double *vr;           /* reflection - coordinates */
  double *ve;           /* expansion - coordinates */
  double *vc;           /* contraction - coordinates */
  double *vm;           /* centroid - coordinates */
  double min;
  
  double fsum,favg,s,cent;
  
  /* dynamically allocate arrays */
  
  /* allocate the rows of the arrays */
  v =  (double **) malloc ((n+1) * sizeof(double *));
  f =  (double *) malloc ((n+1) * sizeof(double));
  vr = (double *) malloc (n * sizeof(double));
  ve = (double *) malloc (n * sizeof(double));  
  vc = (double *) malloc (n * sizeof(double));  
  vm = (double *) malloc (n * sizeof(double));  
  
  /* allocate the columns of the arrays */
  for (i=0;i<=n;i++) {
    v[i] = (double *) malloc (n * sizeof(double));
  }
  
  
  /* create the initial simplex */
  /* assume one of the vertices is 0,0 */
  
  pn = scale*(sqrt(n+1)-1+n)/(n*sqrt(2));
  qn = scale*(sqrt(n+1)-1)/(n*sqrt(2));
  
  for (i=0;i<n;i++) {
    v[0][i] = start[i];
  }
  
  for (i=1;i<=n;i++) {
    for (j=0;j<n;j++) {
      if (i-1 == j) {
	v[i][j] = pn + start[j];
      }
      else {
	v[i][j] = qn + start[j];
      }
    }
  }
  
  /* find the initial function values */
  for (j=0;j<=n;j++) {
    f[j] = func(v[j]);
  }
  
  k = n+1;
  
  /* print out the initial values */
  printf("Initial Values\n");
  for (j=0;j<=n;j++) {
    printf("%f %f %f\n",v[j][0],v[j][1],f[j]);
  }
  
  
  /* begin the main loop of the minimization */
  for (itr=1;itr<=MAX_IT;itr++) {     
    /* find the index of the largest value */
    vg=0;
    for (j=0;j<=n;j++) {
      if (f[j] > f[vg]) {
	vg = j;
      }
    }
    
    /* find the index of the smallest value */
    vs=0;
    for (j=0;j<=n;j++) {
      if (f[j] < f[vs]) {
	vs = j;
      }
    }
    
    /* find the index of the second largest value */
    vh=vs;
    for (j=0;j<=n;j++) {
      if (f[j] > f[vh] && f[j] < f[vg]) {
	vh = j;
      }
    }
    
    /* calculate the centroid */
    for (j=0;j<=n-1;j++) {
      cent=0.0;
      for (m=0;m<=n;m++) {
	if (m!=vg) {
	  cent += v[m][j];
	}
      }
      vm[j] = cent/n;
    }
    
    /* reflect vg to new vertex vr */
    for (j=0;j<=n-1;j++) {
      vr[j] = (1+ALPHA)*vm[j] - ALPHA*v[vg][j];
    }
    fr = func(vr);
    k++;
    
    /* added <= */
    if (fr <= f[vh] && fr > f[vs]) {
      for (j=0;j<=n-1;j++) {
	v[vg][j] = vr[j];
      }
      f[vg] = fr;
    }
    
    /* investigate a step further in this direction */
    /* added <= */
    if ( fr <=  f[vs]) {
      for (j=0;j<=n-1;j++) {
	ve[j] = GAMMA*vr[j] + (1-GAMMA)*vm[j];
      }
      fe = func(ve);
      k++;
      
      /* by making fe < fr as opposed to fe < f[vs],
	 Rosenbrocks function takes 63 iterations as opposed 
	 to 64 when using doubles and e = 1.0e-6. */
      
      if (fe < fr) {
	for (j=0;j<=n-1;j++) {
	  v[vg][j] = ve[j];
	}
	f[vg] = fe;
      }
      else {
	for (j=0;j<=n-1;j++) {
	  v[vg][j] = vr[j];
	}
	f[vg] = fr;
      }
    }
    
    /* check to see if a contraction is necessary */
    if (fr > f[vh]) {
      for (j=0;j<=n-1;j++) {
	vc[j] = BETA*v[vg][j] + (1-BETA)*vm[j];
      }
      fc = func(vc);
      k++;
      if (fc < f[vg]) {
	for (j=0;j<=n-1;j++) {
	  v[vg][j] = vc[j];
	}
	f[vg] = fc;
      }
      /* at this point the contraction is not successful,
	 we must halve the distance from vs to all the 
	 vertices of the simplex and then continue.
	 10/31/97 - modified to account for ALL vertices. 
      */
      else {
	for (row=0;row<=n;row++) {
	  if (row != vs) {
	    for (j=0;j<=n-1;j++) {
	      v[row][j] = v[vs][j]+(v[row][j]-v[vs][j])/2.0;
	    }
	  }
	}
	f[vg] = func(v[vg]);
	k++;
	f[vh] = func(v[vh]);
	k++;
	
	
      }
    }
    
    /* print out the value at each iteration */
    printf("Iteration %d\n",itr);
    for (j=0;j<=n;j++) {
      printf("%f %f %f\n",v[j][0],v[j][1],f[j]);
    }
    
    /* test for convergence */
    fsum = 0.0;
    for (j=0;j<=n;j++) {
      fsum += f[j];
    }
    favg = fsum/(n+1);
    s = 0.0;
    for (j=0;j<=n;j++) {
      s += pow((f[j]-favg),2.0)/(n);
    }
    s = sqrt(s);
    if (s < EPSILON) break;
  }
  /* end main loop of the minimization */
  
  /* find the index of the smallest value */
  vs=0;
  for (j=0;j<=n;j++) {
    if (f[j] < f[vs]) {
      vs = j;
    }
  }
  
  printf("The minimum was found at\n"); 
  for (j=0;j<n;j++) {
    printf("%e\n",v[vs][j]);
    start[j] = v[vs][j];
  }
  min=func(v[vs]);
  k++;
  printf("%d Function Evaluations\n",k);
  printf("%d Iterations through program\n",itr);
  
  free(f);
  free(vr);
  free(ve);
  free(vc);
  free(vm);
  free(*v);
  return min;
}

int main()
{
  double start[] = {-1.2,1.0};
  double min;
  int i;
  
  min=simplex(rosen,start,2,1.0e-4,1);
  
  for (i=0;i<2;i++) {
    printf("%f\n",start[i]);
  }
  return 0;
}




