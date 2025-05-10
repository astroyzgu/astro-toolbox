        program test 
        !Fortran内部定义的和C语言类型互通的模块iso_c_binding
        use, intrinsic :: iso_c_binding 
        implicit none
c------------------------------------------------------------------------------
        interface  
         function sharedata32(x, nx) bind (c)
             use iso_c_binding! fortran                     -->   c --> python 
             integer(4) :: nx ! integer(c_int) ! integer(4) --> int --> int32
             real(8) :: x(nx) ! real(c_double) ! real(8) --> double --> float32
             real(8) :: sharedata32 
         end  
        end interface 
c------------------------------------------------------------------------------
        character*256 :: string 
        integer*4, dimension(10)    :: i1d
        integer*4, dimension(2,3)  :: i2d
        real*8, dimension(6)  ::  x1d
        real*8, dimension(2,3) :: x2d
        !real*8, dimension(:), allocatable :: x2d
        !allocate ( x2d(2,3) ) 
        string = 'Hello world'
        x1d    = RESHAPE([1,2,3,4,5,6], [6])
        i2d    = RESHAPE([1,2,3,4,5,6], [2,3])
        x2d    = RESHAPE([1,2,3,4,5,6], [2,3])
        print *, size(x2d), shape(x2d)
        sharedata32(x2d, size(x2d))
        print *, size(x2d), shape(x2d)
        !call holeinfo(hid,nsub, datah, shid, sid, datas)
	end program test

