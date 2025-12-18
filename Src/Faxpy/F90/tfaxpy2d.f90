program tfaxpy2d
  use faxpy
  implicit none

  integer      :: nx,ny
  integer      :: i,j,k,l
  integer      :: niter 
  real         :: b
  real,dimension(:,:), allocatable  :: x 
  real,dimension(:,:), allocatable  :: y 
  real,dimension(:,:), allocatable  :: a 

  INTEGER :: scount, ecount,rate,err,nm 
  real    :: t


  nx=256
  ny=256
  nm=7
  do l=1,nm
    allocate(x(nx,ny),stat=err)
    allocate(y(nx,ny),stat=err)
    allocate(a(nx,ny),stat=err)

    do i=1,nx
      do j=1,ny
        x(i,j) = 1.0
        y(i,j) = 1.0
      enddo
    enddo

    ! Perform the vector addition 1000 times
    niter = 1000

    call system_clock(scount,rate)
    do i=0,niter 
      b=1.0
      call faxpy2d(a,x,y,b)
      y(1) = 1.0
    enddo
    call system_clock(ecount)
    t = real(ecount-scount,8)/real(rate,8)
    print *, nx 
    print *, ny 
    print *, t
    nx=nx*2
    ny=ny*2
    deallocate(x)
    deallocate(y)
    deallocate(a)
  enddo

end program tfaxpy2d
