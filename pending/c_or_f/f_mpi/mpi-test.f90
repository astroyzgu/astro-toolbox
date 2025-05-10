program main
    use mpi
    implicit none
    integer*4::ierr,my_id,num_procs
    call MPI_INIT ( ierr )
  ! find out my process ID, and how many processes were started.
    call MPI_COMM_RANK (MPI_COMM_WORLD, my_id, ierr)
    call MPI_COMM_SIZE (MPI_COMM_WORLD, num_procs, ierr) !num_procs--number of process
    write(*,'(1x,i2,a,i2)')my_id,'/',num_procs
    call MPI_FINALIZE ( ierr )
end program

