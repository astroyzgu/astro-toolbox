# Python-level module import
# (file: mpi4py/MPI.so)

from mpi4py import MPI

# Python-level objects and code

size  = MPI.COMM_WORLD.Get_size()
rank  = MPI.COMM_WORLD.Get_rank()
pname = MPI.Get_processor_name()

hwmess = "Hello, World! I am process %d of %d on %s."
print (hwmess % (rank, size, pname))
