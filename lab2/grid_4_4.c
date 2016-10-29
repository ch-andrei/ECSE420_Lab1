#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

// Note: the value of the "eta" constant should be 2e-4, the value of the "rho" constant should be 0.5, and the value of the "G" constant should be 0.75.

#define GRID_SIZE 4
#define U_SIZE 3
#define ETA 0.0002
#define RHO 0.5
#define G 0.75

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

void print_nodes(int rank, int adjusted_num_proc, int nodes_per_process, unode_t *nodes){
	float nodes_num[GRID_SIZE * GRID_SIZE];
	if (rank == 0){
		//get own nodes
		for (int i = 0; i < nodes_per_process; i++){
			unode_t *node = nodes + i;
			nodes_num[i] = node->u_array[0];
		}
		// get nodes form other processes
		while (rank++ < adjusted_num_proc-1){
			for (int i = 0; i < nodes_per_process; i++){
				float *ptr = nodes_num + rank * nodes_per_process + i;
				MPI_Recv(ptr, 1, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		for (int i = 0; i < GRID_SIZE; i++){
			for (int j = 0; j < GRID_SIZE; j++){
				printf("[%d,%d] %f, ",i,j,nodes_num[get_1d(i,j,GRID_SIZE)]);
			}
			puts("");
		}
	} else if (rank < adjusted_num_proc){
		// send
		for (int i = 0; i < nodes_per_process; i++){
			unode_t *node = nodes + i;
			float *ptr = node->u_array;
			MPI_Send(ptr, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
	}
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

	// adjust process number if needed
	int adjusted_num_proc = 1;
	while ((adjusted_num_proc <<= 1) <= number_of_processes) {
	}
	adjusted_num_proc >>= 1;
	adjusted_num_proc = (adjusted_num_proc > GRID_SIZE) ? GRID_SIZE: adjusted_num_proc;
	printf("adjusted adjusted_num_proc %d\n", adjusted_num_proc);


	int nodes_per_process = GRID_SIZE * GRID_SIZE / adjusted_num_proc;

	int offset = rank * nodes_per_process;

	// set up node indexes
	unode_t unodes[nodes_per_process];
	int i;
	for (i = 0; i < nodes_per_process; i++){
		unodes[i].i = get_2d (offset + i, GRID_SIZE, 0);
		unodes[i].j = get_2d (offset + i, GRID_SIZE, 1);
		for (int j = 0; j < 3; j++){
			unodes[i].u_array[j] = 1.12;
		}
		//printf ("[%d] Initialized node [%d,%d].\n", rank, unodes[i].i, unodes[i].j);
	}

	print_nodes(rank, adjusted_num_proc, nodes_per_process, unodes);

	/*
	if (rank < adjusted_num_proc){
		float updated[nodes_per_process]; 
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
	} */

	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}
