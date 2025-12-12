import libe     
import diff

def int Main(struct MainArg [*] MainArgs):

  float [*,*] a     # Input array
  float [*,*] out   # Output array
  char  [*] tmp     # Temp buffer ref
  float       x     # Sin argument
  float       dx    # Grid spacing
  float       ddx   # Sampling of sin function 
  float       stdev # Standard deviation
  int         fp    # File pointer
  int         n     # No of bytes to write
  int         nx    # Lengt of arrays in x-direction
  int         ny    # Lengt of arrays in y-direction
  int         i,j   # Loop indices
  int         l     # Differentiator length
  const       PI=3.14    
  float       t0    # Start time
  float       t     # End time
  int         niter # No of calls to kernel functions
  struct diff df    # Diff object

  LibeInit()
  l=6
  nx=8192
  ny=8192
  dx=10.0

  # New differentiator object
  df = DiffNew(l)  

  a = new(float [nx,ny])
  out = new(float [nx,ny])

  # Input data
  ddx = (dx*cast(float,1))*(2.0*PI/(dx*cast(float,nx)))
  for j in range (0,ny):
    for i in range(0,nx):
      x = (dx*cast(float,i))*(2.0*PI/(dx*cast(float,nx)))
      a[i,j] = LibeSin(x)
  
  # Differentiate a, ouput in out

  niter=1500;
  t0=LibeClock()
  for i in range(0,niter):
    DiffDxplus(df,a,out,ddx)
    DiffDxminus(df,a,out,ddx)
    DiffDyplus(df,a,out,ddx)
    DiffDyminus(df,a,out,ddx)

  t=LibeClock()-t0;
  LibePs(" time: "); LibePf(t); LibePs("\n")

  return(0);

