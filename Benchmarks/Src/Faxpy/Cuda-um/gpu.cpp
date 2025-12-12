#include "gpu.h"
extern "C" {
#include <stdio.h>
}

// GpuNew -- Allocate unified memory on host and gpu
void *GpuNew(int n);
void GpuDelete(void *f);
void GpuErrorCheck(char *s);
void * GpuNew(int n){
  void *f;
  cudaError_t cerr;
  cerr = cudaMallocManaged(&f, n);
  if(cerr != cudaSuccess){
    fprintf(stderr,"GpuNew:%s\n ", cudaGetErrorString(cerr)) ;
    exit(1);
  }
  return(f);
}

//GpuDelete -- Delete unified memory on host and gpu
void GpuDelete(void *f){
  cudaError_t cerr;

  cerr=cudaFree(f);
  if(cerr != cudaSuccess){
    fprintf(stderr,"GpuDelete:%s\n ", cudaGetErrorString(cerr)) ;
    exit(1);
  }
  cerr=cudaDeviceSynchronize();
  if(cerr != cudaSuccess){
    fprintf(stderr,"GpuDelete:%s\n ", cudaGetErrorString(cerr)) ;
    exit(1);
  }
}

//GpuError -- Check for gpu errors
void GpuError(){
  cudaDeviceSynchronize();
  cudaError_t cerr;
  cerr = cudaGetLastError();
  if(cerr != cudaSuccess){
    fprintf(stderr,"%s\n",cudaGetErrorString(cerr));
    exit(1);
  }
}
