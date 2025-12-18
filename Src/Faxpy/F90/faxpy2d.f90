! Faxpy is a simple matrix addition test

module faxpy
  implicit none

contains

  subroutine faxpy2d(a,x,y,b)
    real,dimension(:,:)  :: a
    real,dimension(:,:)  :: x
    real,dimension(:,:)  :: y
    real                 :: b

    integer :: i,j
    integer :: nx,ny 
  
    nx=size(a,1)
    ny=size(a,2)

    do concurrent (i=1:nx,j=1:ny)
      a(i,j) = b*y(i,j)+x(i,j)
    enddo

  end subroutine faxpy2d
  
end module

