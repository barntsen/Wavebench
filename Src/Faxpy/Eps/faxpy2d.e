# Faxpy is a simple matrix addition test 

def int faxpy2d(float [*,*] a, float [*,*] x, float [*,*] y, float b):
 int i,j
 int nx,ny 
  
 nx=len(x,0)
 ny=len(x,1)

 parallel(i=0:nx,j=0:ny):
   a[i,j] = b*y[i,j]+x[i,j]

