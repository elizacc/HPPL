from mpi4py import MPI
import numpy as np 

start =  MPI.Wtime()

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

num_iter = 10000

lin = np.linspace(0,1,num_iter)
trap_int = 0
for i in range(rank*(num_iter-1)//size, (rank+1)*(num_iter-1)//size):
    trap_int += (exponential_pdf(lin[i]) + exponential_pdf(lin[i+1]))*(lin[i+1]-lin[i])/2

recvbuf = comm.reduce(trap_int, op=MPI.SUM, root = 0)


end = MPI.Wtime()

if rank==0:
    print(end-start)
