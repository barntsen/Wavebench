// Ac2d object

// Imports
extern "C"
{
#include <stdlib.h>
#include <stdio.h>
}
#include "diff.h"
#include "rec.h"
#include "src.h"
#include "model.h"
#include "ac2d.h"
#include "util.h"
#include "gpu.h"

// Internal functions

  void Ac2dvx(struct ac2d *Ac2d, struct model *Model);
  void Ac2dvy(struct ac2d *Ac2d, struct model *Model);
  void Ac2dstress(struct ac2d *Ac2d, struct model *Model);
  __global__ void ac2dvx(struct ac2d *Ac2d,struct model *Model);
  __global__ void ac2dvy(struct ac2d *Ac2d,struct model *Model);
  __global__ void ac2dstress(struct ac2d *Ac2d,struct model *Model);
                            
// Public functions

// Ac2dNew creates a new Ac2d object
//
// Parameters:
//   - Model : Model object
//
// Return    :Ac2d object  
  struct ac2d *Ac2dNew(struct model *Model)
{
  struct ac2d *Ac2d;
  int i,j;
  int Nx,Ny;

  Nx = Model->Nx;
  Ny = Model->Ny;
  
  Ac2d = (struct ac2d*)GpuNew(sizeof(struct ac2d)); 
  Ac2d->p=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->vx=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->vy=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->exx=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->eyy=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->gammax=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->gammay=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->thetax=(float*)GpuNew(sizeof(float)*Nx*Ny);
  Ac2d->thetay=(float*)GpuNew(sizeof(float)*Nx*Ny);
  
  for (i=0; i<Nx; i=i+1){ 
    for (j=0; j<Ny; j=j+1){ 
      Ac2d->p[idx2(Nx,i,j)]       = 0.0;
      Ac2d->vx[idx2(Nx,i,j)]      = 0.0;
      Ac2d->vy[idx2(Nx,i,j)]      = 0.0;
      Ac2d->exx[idx2(Nx,i,j)]     = 0.0;
      Ac2d->eyy[idx2(Nx,i,j)]     = 0.0;
      Ac2d->gammax[idx2(Nx,i,j)]  = 0.0;
      Ac2d->gammay[idx2(Nx,i,j)]  = 0.0;
      Ac2d->thetax[idx2(Nx,i,j)]  = 0.0;
      Ac2d->thetay[idx2(Nx,i,j)]  = 0.0;
      Ac2d->ts = 0;
    }
  }
  Ac2d->timer = (float*)GpuNew(sizeof(float)*NTIMER);
  for(i=0; i<NTIMER; i=i+1){
    Ac2d->timer[i] = 0.0;
  }
  return(Ac2d);
}
 
// Ac2dPrtime prints the timing of CUDA kernel functions
int Ac2dPrtime(struct ac2d *Ac2d)
{
  printf("Dxplus:   %g\n",Ac2d->timer[DXP]);
  printf("Dyplus:   %g\n",Ac2d->timer[DYP]);
  printf("Dxminus:  %g\n",Ac2d->timer[DXM]);
  printf("Dyminus:  %g\n",Ac2d->timer[DYM]);
  printf("Vx:       %g\n",Ac2d->timer[VX]);
  printf("Vy:       %g\n",Ac2d->timer[VY]);
  printf("Stress:   %g\n",Ac2d->timer[STRESS]);
  
  return(OK);
}

// Ac2dSolve computes the solution of the acoustic wave equation.
// The acoustic equation of motion are integrated using Virieux's (1986) stress-velocity scheme.
// (See the notes.tex file in the Doc directory).
// 
//     vx(t+dt)   = dt/rhox d^+x[ sigma(t)] + dt fx + vx(t)
//                + thetax D[1/rhox]
//     vy(t+dt)   = dt/rhoy d^+y sigma(t) + dt fy(t) + vy(t)
//                + thetay D[1/rhoy]
//
//     dp/dt(t+dt) = dt Kappa[d^-x dexx/dt + d-y deyy/dt + dt dq/dt(t) 
//                 + dt [gammax Dkappa + gammay Dkappa]
//                 + p(t)
//     dexx/dt     =  d^-_x v_x 
//     deyy/dt     =  d^-_z v_y 
//
//     gammax(t+dt) = alpha1x gammax(t) + alpha2x dexx/dt 
//     gammay(t+dt) = alpha1y gammay(t) + alpha2y deyy/dt 
//
//     thetax(t+dt) = eta1x thetax(t) + eta2x d^+x p
//     thetay(t+dt) = eta1y thetay(t) + eta2y d^+y p
//  
//  Parameters:  
//    Ac2d : Solver object
//    Model: Model object
//    Src  : Source object
//    Rec  : Receiver object
//    nt   : Number of timesteps to do starting with current step  
//    l    : The differentiator operator length
int Ac2dSolve(struct ac2d *Ac2d, struct model *Model, struct src *Src, struct rec *Rec,int nt,int l)
{
  int sx,sy;         // Source x,y-coordinates 
  struct diff *Diff;  // Differentiator object
  int ns,ne;         // Start stop timesteps
  int i,k;
  int Nx,Ny;
  float timer;

  float perc,oldperc; // Percentage finished current and old
  int iperc;          // Percentage finished

  Nx = Model->Nx;
  Ny = Model->Ny;

  Diff = DiffNew(l);  // Create differentiator object

  oldperc=0.0;
  ns=Ac2d->ts;         //Get current timestep 
  ne = ns+nt;         
  for(i=ns; i<ne; i=i+1){

    // Compute spatial derivative of stress
    // Use exx and eyy as temp storage

    timer = Clock();
    DiffDxplus(Diff,Ac2d->p,Ac2d->exx,Model->Dx,Nx,Ny); // Forward differentiation x-axis
    Ac2d->timer[DXP] = Clock()-timer + Ac2d->timer[DXP];

    timer = Clock();
    Ac2dvx(Ac2d,Model);                        // Compute vx
    Ac2d->timer[VX] = Clock()-timer + Ac2d->timer[VX];

    timer = Clock();
    DiffDyplus(Diff,Ac2d->p,Ac2d->eyy,Model->Dx,Nx,Ny); // Forward differentiation y-axis
    Ac2d->timer[DYP] = Clock()-timer + Ac2d->timer[DYP];

    timer = Clock();
    Ac2dvy(Ac2d,Model);                        // Compute vy
    Ac2d->timer[VY] = Clock()-timer + Ac2d->timer[VY];

    timer = Clock();
    DiffDxminus(Diff,Ac2d->vx,Ac2d->exx,Model->Dx,Nx,Ny); //Compute exx     
    Ac2d->timer[DXM] = Clock()-timer + Ac2d->timer[DXM];

    timer = Clock();
    DiffDyminus(Diff,Ac2d->vy,Ac2d->eyy,Model->Dx,Nx,Ny); //Compute eyy   
    Ac2d->timer[DYM] = Clock()-timer + Ac2d->timer[DYM];

    // Update stress

      timer = Clock();
      Ac2dstress(Ac2d,Model);  
      Ac2d->timer[STRESS] = Clock()-timer + Ac2d->timer[STRESS];

    // Add source
    for (k=0; k<Src->Ns;k=k+1){
      sx=Src->Sx[k];
      sy=Src->Sy[k];
      Ac2d->p[idx2(Nx,sx,sy)] = Ac2d->p[idx2(Nx,sx,sy)]
      + Model->Dt*(Src->Src[i]/(Model->Dx*Model->Dx)) ; 
    }

    // Print progress
    perc=1000.0*(i)/(ne-ns-1);
    if(perc-oldperc >= 10.0){
      iperc=perc/10;
      if(iperc%10==0){
        fprintf(stderr,"%d\n",iperc);
        fflush(stderr);
      }
      oldperc=perc;
    }

    //Record wavefield
    if(Rec->resamp > 0){
      RecReceiver(Rec,i,Ac2d->p,Nx,Ny); 
    }

    // Record Snapshots
    if(Rec->sresamp > 0){
      RecSnap(Rec,i,Ac2d->p,Nx,Ny);
    }

  }
  return(OK);
}
// Ac2vx computes the x-component of particle velocity
//
// Parameters:
//   Ac2d : Solver object 
//   Model: Model object
void Ac2dvx(struct ac2d *Ac2d, struct model *Model)
{
  ac2dvx<<< NBLOCKS, NTHREADS>>>(Ac2d,Model);
  GpuError();
}
__global__ void ac2dvx(struct ac2d *Ac2d, struct model *Model)
                       
{
  int N,p;
  int i,j;
  float Dt;
  int nx, ny;

  nx=Model->Nx;
  ny=Model->Ny;
  Dt=Model->Dt;

  // The derivative of stress in x-direction is stored in exx
  // Scale with inverse density and advance one time step
  //for(j=0; j<ny; j=j+1)
  //{
  //  for(i=0; i<nx; i=i+1)
  //  {

  N=ny*nx; // No of processors
  for(p=blockIdx.x*blockDim.x + threadIdx.x; p<N; p+=blockDim.x*gridDim.x)
  {
    i = p%(nx);
    j = floorf(p/nx);

        Ac2d->vx[idx2(nx,i,j)] = Dt*(Model->Rho[idx2(nx,i,j)])*Ac2d->exx[idx2(nx,i,j)] 
                               + Ac2d->vx[idx2(nx,i,j)]
                               + Dt*Ac2d->thetax[idx2(nx,i,j)]*Model->Drhox[idx2(nx,i,j)];

        Ac2d->thetax[idx2(nx,i,j)]  = Model->Eta1x[idx2(nx,i,j)]*Ac2d->thetax[idx2(nx,i,j)] 
                                    + Model->Eta2x[idx2(nx,i,j)]*Ac2d->exx[idx2(nx,i,j)];
  }
    //}
  //}
}
// Ac2vy computes the y-component of particle velocity
//
// Parameters:
//   Ac2d : Solver object 
//   Model: Model object
void Ac2dvy(struct ac2d *Ac2d, struct model *Model)
{
  ac2dvy<<< NBLOCKS,NTHREADS>>>(Ac2d,Model);
  GpuError();
}

__global__ void ac2dvy(struct ac2d *Ac2d, struct model *Model)
{
  int N,p;
  int i,j;

  int nx,ny;
  float Dt;
  
  nx=Model->Nx;
  ny=Model->Ny;
  Dt=Model->Dt;

  // The derivative of stress in x-direction is stored in exx
  // Scale with inverse density and advance one time step
  //for(j=0; j<ny; j=j+1)
  //{
  //  for(i=0; i<nx; i=i+1)
  //  {

  N=ny*nx; // No of processors
  for(p=blockIdx.x*blockDim.x + threadIdx.x; p<N; p+=blockDim.x*gridDim.x)
  {
    i = p%(nx);
    j = floorf(p/nx);

        Ac2d->vy[idx2(nx,i,j)] = Dt*(Model->Rho[idx2(nx,i,j)])*Ac2d->eyy[idx2(nx,i,j)] 
                               + Ac2d->vy[idx2(nx,i,j)]
                               + Dt*Ac2d->thetay[idx2(nx,i,j)]*Model->Drhoy[idx2(nx,i,j)];

        Ac2d->thetay[idx2(nx,i,j)]  = Model->Eta1y[idx2(nx,i,j)]*Ac2d->thetay[idx2(nx,i,j)] 
                                    + Model->Eta2y[idx2(nx,i,j)]*Ac2d->eyy[idx2(nx,i,j)];
  }
    //}
  //}
}

// Ac2dstress computes acoustic stress 
//
// Parameters:
//   Ac2d : Solver object 
//   Model: Model object
void Ac2dstress(struct ac2d *Ac2d, struct model *Model)
{
  ac2dstress<<<NBLOCKS,NTHREADS>>>(Ac2d, Model);
  GpuError();
}

 __global__ void ac2dstress(struct ac2d *Ac2d, struct model *Model)
 { 
  int N,pno;
  int i,j;
  int nx,ny;
  float Dt;

  nx=Model->Nx;
  ny=Model->Ny;
  Dt=Model->Dt;

  N=ny*nx; // No of processors
  for(pno=blockIdx.x*blockDim.x + threadIdx.x; pno<N; pno+=blockDim.x*gridDim.x)
  {
    i = pno%(nx);
    j = floorf(pno/nx);

      Ac2d->p[idx2(nx,i,j)] = Dt*Model->Kappa[idx2(nx,i,j)]
                             *(Ac2d->exx[idx2(nx,i,j)] + Ac2d->eyy[idx2(nx,i,j)]) 
                             + Ac2d->p[idx2(nx,i,j)]
                             + Dt*(Ac2d->gammax[idx2(nx,i,j)]*Model->Dkappax[idx2(nx,i,j)]
                             +Ac2d->gammay[idx2(nx,i,j)]*Model->Dkappay[idx2(nx,i,j)]);

      Ac2d->gammax[idx2(nx,i,j)] = Model->Alpha1x[idx2(nx,i,j)]*Ac2d->gammax[idx2(nx,i,j)] 
                           + Model->Alpha2x[idx2(nx,i,j)]*Ac2d->exx[idx2(nx,i,j)];
      Ac2d->gammay[idx2(nx,i,j)] = Model->Alpha1y[idx2(nx,i,j)]*Ac2d->gammay[idx2(nx,i,j)] 
                                 + Model->Alpha2y[idx2(nx,i,j)]*Ac2d->eyy[idx2(nx,i,j)];
  }
}
