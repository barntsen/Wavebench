
extern "C" {
  #include <stdio.h>                 /* Library interface                 */
  #include <stdlib.h>
}
#include "util.h"
#include "model.h"
#include "src.h"
#include "rec.h"
#include "ac2d.h"
//#include "gpu.h"

int NBLOCKS;
int NTHREADS;



int main(int argc, char *argv[])
{
  float *wavelet;    // Source pulse
  float *vp;         //
  float *rho;        //
  float *Q;          //
  float f0;

  struct model *Model; // Model
  struct rec   *Rec;   // Receiver
  struct src   *Src;   // Source

  int *sx, *sy;     // Source x,y-coordinates
  int *rx, *ry;     // Receiver x,y-coordinates
  struct ac2d  *Ac2d;  // FD solver
  int Nx,Ny;       // Model dimension in x- and y-directions.
  int nt,ntr;          // No of time steps
  int resamp, sresamp; // Resampling factors for data and snapshot
  float dt, dx;        // Time sampling and space sampling intervals
  FILE * fd;           // File desriptor for source pulse              
  int i;               // Timestep no
  float W0;
  int Nb,Rheol;
  int Nr;
  int l,nb;
  float si;
  float t0,tios,tioe,tmods,tmode,tss,tse;         // Time at start

  t0 = Clock();
  // Main modeling parameters

  Nx = atoi(argv[1]);
  Ny = atoi(argv[2]);
  nt = atoi(argv[3]);
  resamp = atoi(argv[4]);
  sresamp = atoi(argv[5]);

  dx=5.0; // grid interval
  dt=0.0005; // Time sampling
  l=8;      // Operator length
  f0=25.0;   // Peak frequency
  W0=f0*3.14159*2.0; // Central angular frequency
  Nb = 30;             // Border for PML attenuation
  Rheol = MAXWELL;

  NTHREADS = atoi(argv[6]);
  NBLOCKS  = atoi(argv[7]);

  tios=Clock();
  // Read the velocity model
  fd=fopen("vp.bin","r");
  vp= (float*)malloc(Nx*Ny*sizeof(float));
  nb=fread(vp, sizeof(float), (size_t)Nx*Ny, fd); 
  fclose(fd);

  // Read the density model
  fd=fopen("rho.bin","r");
  rho= (float*)malloc(Nx*Ny*sizeof(float));
  nb=fread(rho, sizeof(float), (size_t)Nx*Ny, fd); 
  fclose(fd);

  // Read the attenuation model
  fd=fopen("q.bin","r");
  Q= (float*)malloc(Nx*Ny*sizeof(float));
  nb=fread(Q, sizeof(float), (size_t)Nx*Ny, fd); 
  fclose(fd);

  // Read a source signature from file
  fd=fopen("src.bin","r");
  wavelet=(float*)malloc(nt*sizeof(float));
  nb=fread(wavelet, sizeof(float), (size_t)nt, fd); 
  fclose(fd);
  tioe=Clock();

  /* Create a source */
  sx=(int*)malloc(sizeof(int));
  sy=(int*)malloc(sizeof(int));
  sx[0]= Nx/2;
  sy[0]= Ny/2;
  Src=SrcNew(wavelet,nt,sx,sy,1);

  /* Create a model   */
  tmods=Clock();
  Model = ModelNew(vp,rho,Q,dx,dt,W0,Nb,Rheol,Nx,Ny); 
  si=ModelStability(Model);
  fprintf(stderr,"Stability index: ");
  fprintf(stderr,"%f\n",si);
  fflush(stderr);
  tmode=Clock();

  // Create a receiver
  Nr=Nx;
  rx=(int*)malloc(sizeof(int)*Nr);
  ry=(int*)malloc(sizeof(int)*Nr);
  for(i=0; i<Nr; i=i+1){
    rx[i] = i;
    ry[i] = 50;
  }
  if(resamp > 0){
    ntr = nt/resamp;
  }else {
    ntr=0;
  }

  char snpfile [] ="snp.bin";
  Rec= RecNew(rx,ry,ntr,resamp,sresamp,snpfile,Nr);

  /* Create solver    */
  Ac2d = Ac2dNew(Model);

  tss=Clock();
  /* Run solver       */
  Ac2dSolve(Ac2d, Model, Src, Rec, nt,l,sresamp,resamp);
  tse=Clock();

  // Save recording

  if(Rec->resamp > 0){
    char pfile [] ="p.bin";
    RecSave(Rec,pfile);
  }

  printf("%d\n", Nx);
  printf("%d\n", Ny);
  printf("%d\n", nt);
  printf("%f\n",tioe-tios); 
  printf("%f\n",tmode-tmods); 
  printf("%f\n",tse-tss); 
  printf("%f\n",Clock()-t0); 
  fflush(stdout);

  return(OK);
}
