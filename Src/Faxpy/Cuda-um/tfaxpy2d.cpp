// Faxpy is a simple matrix addition test 

#include "gpu.h"
#include "util.h"
#include <iostream>

void faxpy2d(float *a, float *x, float *y, float b, int nx, int ny, int NBLOCKS, int NTHREADS);

int main(int argc, char *argv[])
{
  int      nx,ny;
  float *x;
  float *y;
  float *a;
  float *c;
  int i,j,k,l;
  int niter;
  float b;
  float t0,t;
  int nm;

  nx=256;
  ny=256;
  int NTHREADS=256;
  int NBLOCKS=2048;
  
  nm=7;
  for (l=0; l<nm; l=l+1){
    x = (float *)GpuNew(nx*ny*sizeof(float));
    y = (float *)GpuNew(nx*ny*sizeof(float));
    a = (float *)GpuNew(nx*ny*sizeof(float));

    for(i=0; i<nx; i=i+1){
      for(j=0; j<ny; j=j+1){
        x[idx2(nx,i,j)] = 1.0;
        y[idx2(nx,i,j)] = 1.0;
      }
    }

    // Perform the vector addition 1000 times
    niter = 1000;

    t0 = Clock();
    for(i=0; i<niter; i=i+1){
      b=1.0;
      faxpy2d(a,x,y,b,nx,ny,NBLOCKS, NTHREADS);
      y[0,0]=1.0;
    }

    t=Clock()-t0;
    nx=2*nx;
    ny=2*ny;
    GpuDelete(x);
    GpuDelete(y);
    GpuDelete(a);
  }  
  return(0);
}
