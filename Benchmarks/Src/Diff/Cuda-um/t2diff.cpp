// diff is a differentiator test
#include <math.h>
#include <iostream>
#include "gpu.h"
#include "util.h"
#include "diff.h"

int main(int argc, char *argv[])
{
  float *a;     // Input array
  float *out;   // Output array
  float  x  ;   // sin argument
  float  dx ;   // Grid spacing
  float  ddx;   // sin arg spacing
  int    nx ;   // Grid lenght in x-direction
  int    ny ;   // Grid lenght in y-direction
  int   i,j ;   // Loop indices
  int   l   ;   // Operator length
  float PI=3.14159; // Obvious
  float t0  ;   // Start time
  float  t  ;   // End time
  int niter ;   // No of calls to kernel functions
  struct diff *df; // Diff object

  l=6;
  nx=8192;
  ny=8192;
  int NTHREADS=256;
  int NBLOCKS=204800;
  dx=10.0;

  // New differentiator object
  df = DiffNew(l);
  a = (float *)GpuNew(nx*ny*sizeof(float));
  out = (float *)GpuNew(nx*ny*sizeof(float));

 // Input data
  dx=10.0;
  ddx = dx*(2.0*PI)/(dx*nx);
  for(j=0;j<ny; j=j+1){
    for (i=0;i<nx;i=i+1){
      x = (dx*i)*(2.0*PI)/(dx*nx);
      a[idx2(nx,i,j)] = sin(x);
    }
  }

  // Differentiate a, output in out
  niter = 1500;
  t0 = Clock();
  for(i=0; i<niter; i=i+1){
    DiffDxplus(df,a,out,dx,nx,ny,NTHREADS,NBLOCKS);
    DiffDxminus(df,a,out,dx,nx,ny,NTHREADS,NBLOCKS);
    DiffDyplus(df,a,out,dx,nx,ny,NTHREADS,NBLOCKS);
    DiffDyminus(df,a,out,dx,nx,ny,NTHREADS,NBLOCKS);
  }

  t=Clock()-t0;
  std::cout << "time: " << t << "\n";

  GpuDelete(a);
  GpuDelete(out);

  return(0);
}
