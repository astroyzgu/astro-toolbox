      program test
      implicit none 
      integer(4) :: i,j,nh, nhcol 
      real(kind = 8),dimension(:,:),allocatable::x,y
!      external :: sumup
!---------------------------------------------------
      nh= 4
      nhcol = 3
      allocate(y(nh,nhcol))
      do i = 1, nh
         do j = 1, nhcol
            y(i,j) = i - 1 + nhcol*(j-1)
         end do
         write(*,*) y(i,:)
      end do 
      !y = 1
      !allocate( x(Nh,Ns) )
      !print *, shape(x)
      end program test 
      
      subroutine sumup2d()
!      integer(4):: i,j,nx,ny 
!      real(kind = 8) :: t(:,:)
!      real(kind = 8),dimension(:,:),allocatable:: tmp
      print *, 'heall'       
!      nx = size(tmp, 1)
!      ny = size(tmp, 2)
!      print *, size(tmp)
!      do i = 1, arg2
!         arg1(i) 
      !end do  
      !> do something 
      return 
      end subroutine sumup2d  
