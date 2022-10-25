       module xxx 

       contains 
 
       subroutine sumup2d(t)
       integer(4):: i,j,nx,ny 
       real(kind = 8) :: t(:,:)
       real(kind = 8) :: res  
       real(kind = 8),dimension( size(t,1), size(t,2) ) :: tmp
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

       end module 
       !program hello_world
       !  write(*,*) "hello world!!"
       !end program hello_world 
