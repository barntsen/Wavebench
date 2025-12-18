// Faxpy is a simple matrix addition test 

#include "util.h"
#include "gpu.h"


 __global__ void kernel_faxpy2d(float *a, float *x, float *y, int nx, int ny,float b)
{
  int N,pno;
  int i,j;

  N=ny*nx; // No of processors

  for(pno=blockIdx.x*blockDim.x + threadIdx.x; pno<N; pno+=blockDim.x*gridDim.x)
  {
    i = pno%(nx);
    j = floorf(pno/nx);
    a[idx2(nx,i,j)] = b*x[idx2(nx,i,j)] + y[idx2(nx,i,j)];   
  }

  for(pno=blockIdx.x*blockDim.x + threadIdx.x; pno<N; pno+=blockDim.x*gridDim.x)
  {
    i = pno%(nx);
    j = floorf(pno/nx);
    a[idx2(nx,i,j)] = b*x[idx2(nx,i,j)] + y[idx2(nx,i,j)];   
  }

  for(pno=blockIdx.x*blockDim.x + threadIdx.x; pno<N; pno+=blockDim.x*gridDim.x)
  {
    i = pno%(nx);
    j = floorf(pno/nx);
    a[idx2(nx,i,j)] = b*x[idx2(nx,i,j)] + y[idx2(nx,i,j)];   
  }

}

void faxpy2d(float *a, float *x, float *y, float b, int nx, int ny, int NBLOCKS, int NTHREADS)
{
  kernel_faxpy2d<<<NBLOCKS,NTHREADS>>>(a,x,y,nx,ny,b);
  GpuError();
}

