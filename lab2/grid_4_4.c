#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

#define GRID_SIZE 4
#define U_SIZE 3

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

typedef struct unode_t{
	int i;
	int j;
	float u_array[U_SIZE];
} unode_t;

void update_unode(unode_t unode, int new_u){
	unode_t.u_array[2] = unode_t.u_array[1];
	unode_t.u_array[1] = unode_t.u_array[0];
	unode_t.u_array[0] = new_u;
}

int main(int argc, char *argv[])
{
	if(argc<1)
	{
		printf("Not enough arguments. Input arguments as follows:\n");
		return 0;
	}
	int iterations = atoi(argv[1]);

	// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	// Get the number of processes
	int number_of_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int nodes_per_process = GRID_SIZE * GRID_SIZE / number_of_processes;

	int offset = rank * nodes_per_process;

	// set up node indexes
	unode_t unode[nodes_per_process];
	for (int i = 0; i < nodes_per_process; i++){
		unode[i].i = get_2d (offset + i, GRID_SIZE, 0);
		unode[i].j = get_2d (offset + i, GRID_SIZE, 1);
		printf ("[%d] Initialized node [%d,%d].\n", rank, unode[i].i, unode[i].j);
	}

	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}
// Note: the value of the "eta" constant should be 2e-4, the value of the "rho" constant should be 0.5, and the value of the "G" constant should be 0.75.