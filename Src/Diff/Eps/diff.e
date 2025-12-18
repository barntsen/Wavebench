# Diff contains functions for performing differentiation.

import libe

class diff :
  int  l;             # Differentiator length
  int lmax;           # Differentiator max length
  float [*,*] coeffs; # Differentiator weigts
                      # where row no l contains the
                      # weights for a differentiator with
                      # half-length l. 
                      # the second index is 
  float [*]   w;      # Differentiator weights 


def struct diff DiffNew( int l) :

  # DiffNew creates a new differentiator class.
  #
  # Parameters:
  #   l: Length of differentiator l=1,7
  # 
  # Returns:
  #   Differentiator object.


  struct diff Diff; 
  int i,j,k;

  Diff = new(struct diff);
  Diff.lmax = 8;

  if(l < 1):
    l=1;
  if(l > Diff.lmax):
    l=Diff.lmax;

  Diff.l = l;
  Diff.coeffs = new(float [Diff.lmax,Diff.lmax]);
  Diff.w = new(float [l]);

  # Load coefficients
  for (i=0; i<Diff.lmax; i=i+1):
    for (j=0; j<Diff.lmax; j=j+1):
      Diff.coeffs[i,j] = 0.0;
     
  
  # l=1
  Diff.coeffs[0,0] = 1.0021;

  # l=2
  Diff.coeffs[1,0] = 1.1452;
  Diff.coeffs[1,1] = -0.0492;
  
  # l=3
  Diff.coeffs[2,0] = 1.2036;
  Diff.coeffs[2,1] = -0.0833;
  Diff.coeffs[2,2] = 0.0097;

  # l=4
  Diff.coeffs[3,0] = 1.2316;
  Diff.coeffs[3,1] = -0.1041;
  Diff.coeffs[3,2] = 0.0206;
  Diff.coeffs[3,3] = -0.0035;

  # l=5
  Diff.coeffs[4,0] = 1.2463;
  Diff.coeffs[4,1] = -0.1163;
  Diff.coeffs[4,2] = 0.0290;
  Diff.coeffs[4,3] = -0.0080;
  Diff.coeffs[4,4] = 0.0018;

  # l=6
  Diff.coeffs[5,0] = 1.2542;
  Diff.coeffs[5,1] = -0.1213;
  Diff.coeffs[5,2] = 0.0344;
  Diff.coeffs[5,3] = -0.017;
  Diff.coeffs[5,4] = 0.0038;
  Diff.coeffs[5,5] = -0.0011;

  # l=7
  Diff.coeffs[6,0] = 1.2593;
  Diff.coeffs[6,1] = -0.1280;
  Diff.coeffs[6,2] = 0.0384;
  Diff.coeffs[6,3] = -0.0147;
  Diff.coeffs[6,4] = 0.0059;
  Diff.coeffs[6,5] = -0.0022;
  Diff.coeffs[6,6] = 0.0007;

  # l=8
  Diff.coeffs[7,0] = 1.2626;
  Diff.coeffs[7,1] = -0.1312;
  Diff.coeffs[7,2] = 0.0412;
  Diff.coeffs[7,3] = -0.0170;
  Diff.coeffs[7,4] = 0.0076;
  Diff.coeffs[7,5] = -0.0034;
  Diff.coeffs[7,6] = 0.0014;
  Diff.coeffs[7,7] = -0.0005;


  for(k=0;k<l;k=k+1):
    Diff.w[k] = Diff.coeffs[l-1,k];
  
  return(Diff);

def int DiffDxminus(struct diff Diff, float [*,*] A, float [*,*] dA, float dx):

  # Dxminus computes the backward derivative in the x-direction.
  #
  #  Parameters:
  #    Diff     : Diff object 
  #    float  A : Input 2D array
  #    float dx : Sampling interval
  #    float dA : Output array 
  #
  #  Returns:
  #    The output array, dA, contains the derivative for each point computed
  #    as:
  #    dA[i,j] = (1/dx) sum_:k=1^l w[k](A[i+(k-1)dx,j]-A[(i-kdx,j]
  #
  #    w[k] are weights and l is the length of the differentiator.
  #    (see DiffNew for the definitions of these)
 

  int nx, ny;
  int i,j,k;
  float sum;
  int l;
  float [*] w;

  nx = len(A,0);
  ny = len(A,1);

  #
  # Left border (1 <i < l+1)
  #

  l= Diff.l;
  w = Diff.w;

  parallel(i=0:l,j=0:ny):
    sum=0.0;
    for(k=1; k<i+1; k=k+1):
      sum = -w[k-1]*A[i-k,j] + sum; 
    
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*A[i+(k-1),j] +sum; 
    
    dA[i,j] = sum/dx;

  #
  # Outside border area 
  #
  parallel(i=l:nx-l,j=0:ny) :
    sum=0.0;
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*(-A[i-k,j]+A[i+(k-1),j]) +sum; 
    
    dA[i,j] = sum/dx;
   

  #
  # Right border 
  #
  parallel(i=nx-l:nx,j=0:ny) :
    sum = 0.0;
    for(k=1; k<l+1; k=k+1):
      sum = -w[k-1]*A[i-k,j] + sum;
    

    for(k=1; k<(nx-i+1); k=k+1):
      sum = w[k-1]*A[i+(k-1),j] + sum;
    
    dA[i,j] = sum/dx;
  


def int DiffDxplus(struct diff Diff, float [*,*] A, float [*,*] dA, float dx):

  # Dxplus computes the forward derivative in the x-direction.
  #
  # Parameters: 
  #   Diff     : Diff object 
  #   float  A : Input 2D array
  #   float dx : Sampling interval
  #   float dA : Output array 
  #
  # Returns:
  #   The output array, dA, contains the derivative for each point computed
  #   as:
  #   dA[i,j] = (1/dx) sum_:k=1^l w[k](A[i+kdx,j]-A[(i-(k-1)dx,j]
  #
  #   w[k] are weights and l is the length of the differentiator.
  #   (see DiffNew for the definitions of these)

  int nx, ny;
  int i,j,k;
  float sum;
  int l;
  float [*] w;

  nx = len(A,0);
  ny = len(A,1);

  #
  # Left border (1 <i < l+1)
  #

  l= Diff.l;
  w = Diff.w;

  # Left border

  parallel(i=0:l,j=0:ny) :
    sum=0.0;
    for(k=1; k<i+2; k=k+1):
      sum = -w[k-1]*A[i-(k-1),j] + sum; 
    
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*A[i+k,j] +sum; 
    
    dA[i,j] = sum/dx;
   
  #
  # Between left and right border
  #
  parallel(i=l:nx-l,j=0:ny) :
    sum=0.0;
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*(-A[i-(k-1),j]+A[i+k,j]) +sum; 
    
    dA[i,j] = sum/dx;
   

  #
  # Right border 
  #
  parallel(i=nx-l:nx,j=0:ny):
    sum = 0.0;
    for(k=1; k<l+1; k=k+1):
      sum = -w[k-1]*A[i-(k-1),j] + sum;

    for(k=1; k<nx-i; k=k+1):
      sum = w[k-1]*A[i+k,j] + sum;
    
    dA[i,j] = sum/dx;
  
def int DiffDyminus(struct diff Diff, float [*,*] A, float [*,*] dA, float dx):

  # Dyminus computes the backward derivative in the y-direction.
  #
  # Parameters:
  #   Diff     : Diff object 
  #   float  A : Input 2D array
  #   float dx : Sampling interval
  #   float dA : Output array 
  #
  #  Returns:
  #    The output array, dA, contains the derivative for each point computed
  #    as:
  #    dA[i,j] = (1/dx) sum_:k=1^l w[k](A[i,j+(k-1)dx]-A[i,j-kdx,j]
  #
  #  w[k] are weights and l is the length of the differentiator.
  #  (see DiffNew for the definitions of these)

  int nx, ny;
  int i,j,k;
  float sum;
  int l;
  float [*] w;

  nx = len(A,0);
  ny = len(A,1);

  #
  # Top border (1 <i < l+1)
  #

  l= Diff.l;
  w = Diff.w;

  # Left border 

  parallel(i=0:nx,j=0:l) :
    sum=0.0;
    for(k=1; k<j+1; k=k+1):
      sum = -w[k-1]*A[i,j-k] + sum; 
    
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*A[i,j+(k-1)] +sum; 
    
    dA[i,j] = sum/dx;

  #
  # Outside border area 
  #
  parallel(i=0:nx,j=l:ny-l) :
    sum=0.0;
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*(-A[i,j-k]+A[i,j+(k-1)]) +sum; 
    
    dA[i,j] = sum/dx;

  #
  # Right border 
  #
  parallel(i=0:nx,j=ny-l:ny) :
    sum = 0.0;
    for(k=1; k<l+1; k=k+1):
      sum = -w[k-1]*A[i,j-k] + sum;

    for(k=1; k<(ny-j+1); k=k+1):
      sum = w[k-1]*A[i,j+(k-1)] + sum;
    
    dA[i,j] = sum/dx;
  

def int DiffDyplus(struct diff Diff, float [*,*] A, float [*,*] dA, float dx):

# Dyplus computes the forward derivative in the x-direction

  # Parameters:
  #  Diff     : Diff object 
  #  float  A : Input 2D array
  #  float dx : Sampling interval
  #  float dA : Output array 
  #
  #  Returns:
  #    The output array, dA, contains the derivative for each point computed
  #    as:
  #    dA[i,j] = (1/dx) sum_:k=1^l w[k](A[i+kdx,j]-A[(i-(k-1)dx,j]
  #
  #    w[k] are weights and l is the length of the differentiator.
  #    (see DiffNew for the definitions of these)

  int nx, ny;
  int i,j,k;
  float sum;
  int l;
  float [*] w;

  nx = len(A,0);
  ny = len(A,1);

  #
  # Left border (1 <i < l+1)
  #

  l= Diff.l;
  w = Diff.w;

  # Left border
  parallel(i=0:nx,j=0:l) :
    sum=0.0;
    for(k=1; k<j+2; k=k+1):
      sum = -w[k-1]*A[i,j-(k-1)] + sum; 
    
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*A[i,j+k] +sum; 
    
    dA[i,j] = sum/dx;

  #
  # Between left and right border
  #
  parallel(i=0:nx,j=l:ny-l):
    sum=0.0;
    for(k=1; k<l+1; k=k+1):
      sum = w[k-1]*(-A[i,j-(k-1)]+A[i,j+k]) +sum; 
    
    dA[i,j] = sum/dx;
   

  #
  # Right border 
  #
  parallel(i=0:nx,j=ny-l:ny):
    sum = 0.0;
    for(k=1; k<l+1; k=k+1):
      sum = -w[k-1]*A[i,j-(k-1)] + sum;
    

    for(k=1; k<ny-j; k=k+1):
      sum = w[k-1]*A[i,j+k] + sum;
    
    dA[i,j] = sum/dx;
  

