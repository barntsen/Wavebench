# Faxpy is a simple matrix addition test 

struct A :
  float [*,*] a

struct X :
  float [*,*] x

struct Y :
  float [*,*] y

def int faxpy2d(struct A aa, struct X xx, struct Y yy, float b):
 int i,j
 int nx,ny 
  
 nx=len(xx.x,0)
 ny=len(xx.x,1)

 parallel(i=0:nx,j=0:ny):
   aa.a[i,j] = b*yy.y[i,j]+xx.x[i,j]

