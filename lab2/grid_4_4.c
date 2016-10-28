#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

#define GRID_SIZE 4
#define U_SIZE 3
#define ETA 

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

typedef struct unode_t{
	int i;
	int j;
	float u_array[U_SIZE];
} unode_t;

void update_unode(unode_t unode, int new_u){
	unode.u_array[2] = unode.u_array[1];
	unode.u_array[1] = unode.u_array[0];
	unode.u_array[0] = new_u;
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
	unode_t unodes[nodes_per_process];
	int i;
	for (i = 0; i < nodes_per_process; i++){
		unodes[i].i = get_2d (offset + i, GRID_SIZE, 0);
		unodes[i].j = get_2d (offset + i, GRID_SIZE, 1);
		for (int j = 0; j < 3; j++){
			unodes[i].u_array[j] = 0;
		}
		//printf ("[%d] Initialized node [%d,%d].\n", rank, unodes[i].i, unodes[i].j);
	}

	while (iterations-- > 0){
		for (i = 0; i < nodes_per_process; i++){
			int node_num = offset + i;
			if (node_num == 0 || node_num == GRID_SIZE - 1 || node_num == GRID_SIZE * (GRID_SIZE - 1) || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
				// if corner
				// TODO
				printf ("[%d,%d] corner node [%d,%d].\n", rank, node_num, unodes[i].i, unodes[i].j);
			} else if (node_num % GRID_SIZE == 0 || node_num % GRID_SIZE == GRID_SIZE - 1){
				// if edge but not corner
				// TODO
				printf ("[%d,%d] edge node [%d,%d].\n", rank, node_num, unodes[i].i, unodes[i].j);
			} else {
				// if not corner nor edge'
				// TODO
				printf ("[%d,%d] central node [%d,%d].\n", rank, node_num, unodes[i].i, unodes[i].j);
			}
		}
	}

	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}
// Note: the value of the "eta" constant should be 2e-4, the value of the "rho" constant should be 0.5, and the value of the "G" constant should be 0.75.