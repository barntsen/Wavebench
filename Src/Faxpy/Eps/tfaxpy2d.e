# tfaxpy2d runs a test of the faxpy2d matrix addition function.

import libe
import faxpy2d

def int Main(struct MainArg [*] MainArgs):

  # The test runs (a number of times) the basic matrix operation
  #
  # a=bx+y,
  #
  # where a,x,y are 2d matrices
  # for different matrix sizes.

  int         nx,ny
  float [*,*] x  # Input to faxpy2d
  float [*,*] y  # Input to faxpy2d
  float [*,*] a  # Output from faxpy2d
  int i,j,l      # Iteration indices
  int niter      # No of times to call faxpy2d
  float b        # Input to faxpy2d
  float t0,t     # Timing variables
  int nm         # No of matrix sizes to run


  LibeInit(); 

  nx=256
  ny=256
  
  nm=7
  for l in range(0,nm):
    x = new(float[nx,ny])
    y = new(float[nx,ny])
    a = new(float[nx,ny])

    for(i=0; i<nx; i=i+1):
      for(j=0; j<ny; j=j+1):
        x[i,j] = 1.0
        y[i,j] = 1.0

    niter = 1000
    t0 = LibeClock()
    for(i=0; i<niter; i=i+1):
      b=1.0
      faxpy2d(a,x,y,b)
    t=LibeClock()-t0

    LibePi(nx); LibePs("\n")
    LibePi(ny); LibePs("\n")
    LibePf(t);  LibePs("\n")
    nx=2*nx
    ny=2*ny
    delete(x)
    delete(y)
    delete(a)
  return(0)
