# Cython-level cimport with PXD file
# this make available the native MPI C API
# with namespace-protection (stuff accessed as mpi.XXX)
# (file: mpi4py/include/mpi4py/libmpi.pxd)

from mpi4py cimport libmpi as mpi

cdef int ierr1=0

cdef int size1 = 0
ierr1 = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD, &size1)

cdef int rank1 = 0
ierr1 = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD, &rank1)

cdef int rlen1=0
cdef char pname1[mpi.MPI_MAX_PROCESSOR_NAME]
ierr1 = mpi.MPI_Get_processor_name(pname1, &rlen1)
pname1[rlen1] = 0 # just in case ;-)

hwmess = "Hello, World! I am process %d of %d on %s."
print (hwmess % (rank1, size1, pname1))
