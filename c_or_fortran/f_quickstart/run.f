       program hello_world
       use xxx
       real(kind = 8),dimension(:,:),allocatable:: arr
       allocate( arr(3,4) )
       arr = 1
       print *, arr 
       call sumup2d(arr)  
       write(*,*) "hello world!!"
       end program hello_world 
