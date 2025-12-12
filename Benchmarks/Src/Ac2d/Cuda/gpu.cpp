#include "gpu.h"
extern "C" {
#include <stdio.h>
}

// GpuNew -- Allocate memory on host 
void *GpuNew(int n);
void GpuDelete(void *f);
void GpuErrorCheck(char *s);

void * GpuNew(int n){
  void * mem;
  
  mem = malloc(n*sizeof(float));
  
  return(mem);
}

//GpuDelete -- Delete memory on host 
void GpuDelete(void *f){
  free(f);
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
