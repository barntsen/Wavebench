import libe     
import diff

def int Main(struct MainArg [*] MainArgs):

  float [*,*] a     # Input array
  float [*,*] b     # Output array
  float [*,*] out   # Output array
  float [*,*] err   # Error
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
  float       t0
  struct diff df    # Diff object

  LibeInit()
  l=1
  nx=8192
  ny=8192
  dx=10.0

  # New differentiator object
  df = DiffNew(l)  

  a = new(float [nx,ny])
  out = new(float [nx,ny])
  err = new(float [nx,ny])

  # Test input data
  ddx = (dx*cast(float,1))*(2.0*PI/(dx*cast(float,nx)))
  for j in range (0,ny):
    for i in range(0,nx):
      x = (dx*cast(float,i))*(2.0*PI/(dx*cast(float,nx)))
      a[i,j] = LibeSin(x)
  
  # Differentiate a, ouput in b

  t0=LibeClock()
  for i in range(0,1500):
    DiffDxplus(df,a,out,ddx)
    DiffDxminus(df,a,out,ddx)
    DiffDyplus(df,a,out,ddx)
    DiffDyminus(df,a,out,ddx)
  LibePs(" time: "); LibePf(LibeClock()-t0); LibePs("\n")

  stdev = 0.0
  for i in range(0,nx):
    for j in range (0,ny):
      x = (dx*cast(float,i))*(2.0*PI/(dx*cast(float,nx)))
      err[i,j] = out[i,j] - LibeCos(x+0.5*ddx)
      stdev = stdev+err[i,j]*err[i,j] 

  stdev=stdev/cast(float,nx)
  stdev=LibeSqrt(stdev)
  LibePs("stdev: "); LibePf(stdev); LibePs("\n")

  return(0);

