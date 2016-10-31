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

#define SEND_OP 0
#define RECEIVE_OP 1

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

typedef struct unode_t{
	int i;
	int j;
	float u_array[U_SIZE];
} unode_t;

void update_unode(unode_t *unode, float new_u){
	unode->u_array[2] = unode->u_array[1];
	unode->u_array[1] = unode->u_array[0];
	unode->u_array[0] = new_u;
}

void print_nodes(int rank, int num_proc, int nodes_per_process, unode_t *nodes){
	float nodes_num[GRID_SIZE * GRID_SIZE];
	if (rank == 0){
		//get own nodes
		for (int i = 0; i < nodes_per_process; i++){
			unode_t *node = nodes + i;
			nodes_num[i] = node->u_array[0];
		}
		// get nodes form other processes
		while (rank++ < num_proc-1){
			for (int i = 0; i < nodes_per_process; i++){
				float *ptr = nodes_num + rank * nodes_per_process + i;
				MPI_Recv(ptr, 1, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		// print nodes
		for (int i = 0; i < GRID_SIZE; i++){
			for (int j = 0; j < GRID_SIZE; j++){
				printf("{[%d,%d] %f} \t",i,j,nodes_num[get_1d(i,j,GRID_SIZE)]);
			}
			puts("");
		}
		puts("");
	} else if (rank < num_proc){
		// send
		for (int i = 0; i < nodes_per_process; i++){
			unode_t *node = nodes + i;
			float *ptr = node->u_array;
			MPI_Send(ptr, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
	}
}

int get_node_process_id(int i, int j, int nodes_per_process){
	if (i < 0 || j < 0 || i >= GRID_SIZE || j >= GRID_SIZE) 
		return -1;
	return get_1d(i, j, GRID_SIZE) / nodes_per_process;
}

void get_index_by(int k, int i, int j, int *ii, int *jj){
	switch (k){
		default:
			break;
		case 0:
			*ii = i - 1;
			*jj = j;
			break;
		case 1:
			*ii = i;
			*jj = j + 1;
			break;
		case 2:
			*ii = i + 1;
			*jj = j;
			break;
		case 3:
			*ii = i;
			*jj = j - 1;
			break;
	}
}

void exchange_unode_data(int operation, int rank, int offset, int nodes_per_process, unode_t unodes[], float received_buffer[]){
	// RECEIVE DATA
	int counter;
	for (int i = 0; i < GRID_SIZE; i++){
		for (int j = 0; j < GRID_SIZE; j++){
			int node_num = get_1d(i,j,GRID_SIZE);
			if (node_num >= offset && node_num < offset + nodes_per_process){
				int kcount = 0;
				for (int k = 0; k < 4; k++){
					int ii = -1, jj = -1, tag;
					get_index_by(k, i, j, &ii, &jj);
					if (operation == SEND_OP){
						int dest_rank = get_node_process_id(ii, jj, nodes_per_process);
						if (dest_rank != -1 && dest_rank != rank){
							float num = unodes[node_num - offset].u_array[0];
							tag = get_1d(i,j,GRID_SIZE);
							MPI_Send(&num, 1, MPI_FLOAT, dest_rank, tag, MPI_COMM_WORLD);
							//if (i == 2 && j == 2) 
							//	printf("[%d]:{%d,%d} sent <%f> to [%d]:{%d,%d}; tag %d; node num %d\n", rank, i, j, num, dest_rank, ii, jj, tag, node_num - offset);
							counter++;
						}
					} else if (operation == RECEIVE_OP){
						int source_rank = get_node_process_id(ii, jj, nodes_per_process);
						if (source_rank != -1 && source_rank != rank){
							float num;
							tag = get_1d(ii,jj,GRID_SIZE);
							MPI_Recv(&num, 1, MPI_FLOAT, source_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							received_buffer[node_num-offset+k] = num;
							//if (i == 2 && j == 2) 
							//	printf("[%d]:{%d,%d} received <%f> from [%d]:{%d,%d}; tag %d; node num %d\n", rank, i, j, num, source_rank, ii, jj, tag, node_num - offset);
							counter++;
							kcount++;
						} else if (source_rank != -1){
							int index = node_num - offset;
							if (i == ii){
								if (j + 1 == jj){
									// right
									index += 1;
								} else if (j - 1 == jj){
									// left
									index += -1;
								}
							} else if (j == jj){
								if (i + 1 == ii){
									// down
									index += GRID_SIZE;
								} else if (i - 1 == ii){
									// up
									index += -GRID_SIZE;
								}
							}
							//if (rank == 1) 
							//	printf("[%d] node %d (%d,%d) getting from node %d (%d,%d) {dir %d}: val %f\n", 
							//	rank, node_num - offset, i,j,index, ii,jj, k, unodes[index].u_array[0]);
							received_buffer[4*(node_num - offset) + k] = unodes[index].u_array[0];
							kcount++;
						}
					}
				}
			//if (operation == RECEIVE_OP)
			//	printf("[%d{%d,%d}] kcount %d\n", rank,i,j, kcount);
			} else {
				// do nothing
			}
		}
	}
	//printf("[%d] performed (%d) operations.\n", rank, counter);
	MPI_Barrier(MPI_COMM_WORLD);
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

	if (rank < number_of_processes){
		// init nodes
		unode_t unodes[nodes_per_process];
		int i;
		for (i = 0; i < nodes_per_process; i++){
			unodes[i].i = get_2d (offset + i, GRID_SIZE, 0);
			unodes[i].j = get_2d (offset + i, GRID_SIZE, 1);
			for (int j = 0; j < 3; j++){
				unodes[i].u_array[j] = 0;
			}
		}

		// add perturbation
		int perturb_node_offset = get_1d(GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE);
		if (perturb_node_offset >= offset && perturb_node_offset < offset + nodes_per_process){
			unodes[perturb_node_offset-offset].u_array[0] = 1.0;
			printf("[%d] perturbed at offset global/local (%d/%d)\n", rank, perturb_node_offset, perturb_node_offset-offset);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		print_nodes(rank, number_of_processes, nodes_per_process, unodes);
		MPI_Barrier(MPI_COMM_WORLD);

		// run computation iterations
		while (iterations-- > 0){
			// send
			exchange_unode_data(SEND_OP, rank, offset, nodes_per_process, unodes, NULL);
			// receieve
			float received_buffer[nodes_per_process*4];
			exchange_unode_data(RECEIVE_OP, rank, offset, nodes_per_process, unodes, received_buffer);
			float updated;
			// update central nodes
			for (i = 0; i < nodes_per_process; i++){
				int node_num = offset + i;
				if (node_num == 0 || node_num == GRID_SIZE - 1 || node_num == GRID_SIZE * (GRID_SIZE - 1) || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
					// if corner; do nothing
				} else if (node_num % GRID_SIZE == 0 || node_num % GRID_SIZE == GRID_SIZE - 1 || node_num < GRID_SIZE || node_num > GRID_SIZE * (GRID_SIZE - 1)){
					// if edge but not corner; do nothing
				} else {
					// if central node; update
					//printf ("[%d,%d] central node [%d,%d].\n", rank, node_num, unodes[i].i, unodes[i].j);
					updated = RHO * (-4) * unodes[i].u_array[0] + 2 * unodes[i].u_array[0] - (1 - ETA) * unodes[i].u_array[1];
					updated += RHO * (received_buffer[i] + received_buffer[i+1] + received_buffer[i+2] + received_buffer[i+3]);
					updated /= (1+ETA);
					update_unode(unodes+i, updated);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			// ***************************************************************8
			// send
			exchange_unode_data(SEND_OP, rank, offset, nodes_per_process, unodes, NULL);
			// receieve
			exchange_unode_data(RECEIVE_OP, rank, offset, nodes_per_process, unodes, received_buffer);
			// update central nodes
			for (i = 0; i < nodes_per_process; i++){
				int node_num = offset + i;
				if (node_num == 0 || node_num == GRID_SIZE - 1 || node_num == GRID_SIZE * (GRID_SIZE - 1) || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
					// if corner; do nothing
				} else if (node_num % GRID_SIZE == 0){
					// left edge
					updated = G * received_buffer[4*i+1];
					update_unode(unodes+i, updated);
				} else if (node_num % GRID_SIZE == GRID_SIZE - 1) {
					// right edge
					updated = G * received_buffer[4*i+3];
					update_unode(unodes+i, updated);
				} else if(node_num < GRID_SIZE){
					// top edge
					updated = G * received_buffer[4*i+2];
					update_unode(unodes+i, updated);
				} else if (node_num > GRID_SIZE * (GRID_SIZE - 1)){
					// bottom edge
					updated = G * received_buffer[4*i];
					update_unode(unodes+i, updated);
				} else {
					// if central node; do nothing
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);

			// ***************************************************************8
			// send
			exchange_unode_data(SEND_OP, rank, offset, nodes_per_process, unodes, NULL);
			// receieve
			exchange_unode_data(RECEIVE_OP, rank, offset, nodes_per_process, unodes, received_buffer);
			// update central nodes
			for (i = 0; i < nodes_per_process; i++){
				int node_num = offset + i;
				if (node_num == 0) {
					// left top corner
					updated = G * received_buffer[4*i+2];
					update_unode(unodes+i, updated);
				} else if (node_num == GRID_SIZE * (GRID_SIZE - 1)){
					// left bottom corner
					updated = G * received_buffer[4*i];
					update_unode(unodes+i, updated);
				} else if (node_num == GRID_SIZE - 1 || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
					// right top and bottom corners
					updated = G * received_buffer[4*i+3];
					update_unode(unodes+i, updated);
				} else if (node_num % GRID_SIZE == 0 || node_num % GRID_SIZE == GRID_SIZE - 1 || node_num < GRID_SIZE || node_num > GRID_SIZE * (GRID_SIZE - 1)){
					// if edge but not corner; do nothing
				} else {
					// if central node; do nothing
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			// ***************************************************************8

			print_nodes(rank, number_of_processes, nodes_per_process, unodes);
			}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}
