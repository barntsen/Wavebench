# Faxpy is a simple matrix addition test 

int faxpy3d(float [*,*,*] a, float [*,*,*] x, float [*,*,*] y, float b):
 int i,j,k
 int nx,ny,nz 
  
 nx=len(x,0)
 ny=len(x,1)
 ny=len(x,2)

 parallel(i=0:nx,j=0:ny,k=0:nz):
   a[i,j,k] = b*y[i,j,k]+x[i,j,k]

