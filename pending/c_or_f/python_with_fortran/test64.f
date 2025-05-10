        program test64 
        use, intrinsic :: iso_c_binding !Fortran内部定义的和C语言类型互通的模块iso_c_binding
        implicit none 

        interface  
         subroutine f2py_test64(x, nx, str) bind (c)
             use iso_c_binding    ! fortran    --> c      --> python 
             integer(4) :: nx ! integer(4) --> int    --> int32
             real(8) :: x(nx) ! real(8) --> double --> float32
             character(c_char) :: str
         end subroutine f2py_test64
        end interface 

c------- main -------------------------------------------------
        character*256 ::  char0
        integer(c_int) :: int0  
        integer(4) :: int1
        real*8, dimension(:), allocatable :: x
        allocate ( x(2) ) 
        char0 = 'Hello world' 
        int0 =  2147483647  
        x(1) =  0.123456789
        x(2) =  123456789.0
        write(*,"(F20.11)") x(1)
        print *, '########### start test #############'
        print *, 'in fortan ', int0
        print *, 'in fortan ', x, size(x)
        print *, 'in fortan ', char0(:20)
        print *, 'transit data from fortran into python' 
        call f2py_test32( x, size(x), char0 )
        print *, 'in fortan ', x 
        print *, '########### end   test #############'
	end program test64
