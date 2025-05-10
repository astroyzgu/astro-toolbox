       program main
       use userdef
       implicit none
       integer:: a,b
       real(kind = 8),dimension(:),allocatable:: arr1d
       real(kind = 8),dimension(:,:),allocatable:: arr2d
       allocate( arr1d(2), arr2d(3,4) )
       a=1
       b=2
       arr1d = 0
       arr2d = 1
       call test2d(a, b, arr1d)
       call sumup2d(arr2d)
       end program

       module userdef

       contains 

       subroutine test2d(num1, num2, t)
       implicit none
       integer::num1,num2
       real*8 :: t(:)
       real*8 :: tmp(size(t,1))
       tmp = t  ! 
       write(*,*)num1,num2
       write(*,*)tmp, size(tmp)
       !> do something 
       t(1)   = 2 ! 外部程序中的值也会变化
       tmp(1) = 2 ! 外部程序中的值不会变化
       return
       end subroutine test2d

       subroutine sumup2d(t)
       integer(4):: i,j,nx,ny 
       real(kind = 8) :: t(:,:)
       real(kind = 8) :: res  
       real(kind = 8), dimension( size(t,1), size(t,2) ) :: tmp
       tmp = t 
       res = 0 
       do i = 1, size(tmp, 1) 
          do j = 1, size(tmp, 2) 
             res = res + tmp(i,j)
          end do  
       end do  
       !> do something 
       print *, res 
       return 
       end subroutine sumup2d  

       !subroutine hello_world() 
       !  write(*,*) "hello world!!"
       !end subroutine hello_world 

       end module 

