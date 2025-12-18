
#ifndef GPU_H
#define GPU_H
  extern int NTHREADS;
  extern int NBLOCKS;
#endif

void *GpuNew(int n);
void GpuDelete(void *f);
void GpuError();




