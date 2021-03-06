#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>
#include "output.h"

// Note: the value of the "eta" constant should be 2e-4, the value of the "rho" constant should be 0.5, and the value of the "G" constant should be 0.75.

#ifdef RUNTIME_SMALL
#define GRID_SIZE		4
#else
#define GRID_SIZE		512
#endif

#define U_SIZE			3
#define ETA				0.0002
#define RHO				0.5
#define G				0.75

#define UPDATE_CORNERS	0
#define UPDATE_EDGES	1
#define UPDATE_CENTRAL	2

#define SEND_OP			0
#define RECEIVE_OP		1

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

typedef struct unode_t{
	int i;
	int j;
	float u_array[U_SIZE];
} unode_t;

/**
   * @brief updates unode values
   */
void update_unode(unode_t *unode, float new_u){
	unode->u_array[2] = unode->u_array[1];
	unode->u_array[1] = unode->u_array[0];
	unode->u_array[0] = new_u;
}

/**
   * @brief Prints the entire unode array to screen
   */
void print_nodes(int rank, int num_proc, int nodes_per_process, unode_t *nodes){
	MPI_Barrier(MPI_COMM_WORLD);
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
				//printf("{[%d,%d] %f} \t",i,j,nodes_num[get_1d(i,j,GRID_SIZE)]);
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
	MPI_Barrier(MPI_COMM_WORLD);
}

/**
   * @brief Gets the rank of the process to which node i,j belongs
   */
int get_node_process_id(int i, int j, int nodes_per_process){
	if (i < 0 || j < 0 || i >= GRID_SIZE || j >= GRID_SIZE) 
		return -1;
	return get_1d(i, j, GRID_SIZE) / nodes_per_process;
}

/**
   * @brief Updates indexes ii, jj based on the direction k and initial position i,j
   */
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

/**
   * @brief Return true if the node needs to be update, false otherwise (return integer as boolean, 0 for false, 1 for true)
   */
int check_update_condition(int update_restriction, int node_num){
	if (node_num == 0 || node_num == GRID_SIZE - 1 || node_num == GRID_SIZE * (GRID_SIZE - 1) || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
		// if corner
		if (update_restriction != UPDATE_CORNERS)
			return 0;
	} else if (node_num % GRID_SIZE == 0 || node_num % GRID_SIZE == GRID_SIZE - 1 || node_num < GRID_SIZE || node_num > GRID_SIZE * (GRID_SIZE - 1)){
		// if edge but not corner
		if (update_restriction != UPDATE_EDGES)
			return 0;
	} else {
		// if central node
		if (update_restriction != UPDATE_CENTRAL)
			return 0;
	}
	return 1;
}

/**
   * @brief Gets all necessary unode data using MPI communication or getting values from the local process as relevant  
   */
void exchange_unode_data(int operation, int update_restriction, int rank, int offset, int nodes_per_process, unode_t unodes[], float node_data_buffer[]){
	int counter;
	// loop over all the nodes; work only on those that are assigned to the current process
	for (int i = 0; i < GRID_SIZE; i++){
		for (int j = 0; j < GRID_SIZE; j++){
			// check if node is assigned to this process
			int node_num = get_1d(i,j,GRID_SIZE);
			if (node_num >= offset && node_num < offset + nodes_per_process){
				// this node is assigned to this process
				// check if it need to be updated
				int kcount = 0;
				for (int k = 0; k < 4; k++){
					int ii = -1, jj = -1, tag;
					get_index_by(k, i, j, &ii, &jj);
					if (operation == SEND_OP){
						int dest_rank = get_node_process_id(ii, jj, nodes_per_process);
						if (dest_rank != -1 && dest_rank != rank){
							int dest_node_num = get_1d(ii,jj,GRID_SIZE);
							if (!check_update_condition(update_restriction, dest_node_num))
								continue;
							// if belongs to another process, send using MPI
							float num = unodes[node_num - offset].u_array[0];
							tag = get_1d(i,j,GRID_SIZE);
							MPI_Send(&num, 1, MPI_FLOAT, dest_rank, tag, MPI_COMM_WORLD);
							//if (i == 2 && j == 2) 
							//	printf("[%d]:{%d,%d} sent <%f> to [%d]:{%d,%d}; tag %d; node num %d\n", rank, i, j, num, dest_rank, ii, jj, tag, node_num - offset);
							counter++;
						} else {
							// belongs to self, no need to send anything
							// do nothing
						}
					} else if (operation == RECEIVE_OP){
						int source_rank = get_node_process_id(ii, jj, nodes_per_process);
						if (source_rank != -1 && source_rank != rank){
							if (!check_update_condition(update_restriction, node_num))
								continue;
							// if belongs to another process, receive using MPI
							float num;
							tag = get_1d(ii,jj,GRID_SIZE);
							MPI_Recv(&num, 1, MPI_FLOAT, source_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							node_data_buffer[4*(node_num - offset) + k] = num;
							//if (i == 2 && j == 2) 
							//	printf("[%d]:{%d,%d} received <%f> from [%d]:{%d,%d}; tag %d; node num %d\n", rank, i, j, num, source_rank, ii, jj, tag, node_num - offset);
							counter++;
							kcount++;
						} else if (source_rank != -1){
							// if belongs to self, get from the unode array
							int index = node_num - offset;
							if (k == 0)
								index += -GRID_SIZE;
							else if (k == 1)
								index += 1;
							else if (k == 2)
								index += GRID_SIZE;
							else if (k == 3)
								index += -1;
							//if (rank == 1) 
							//	printf("[%d] node %d (%d,%d) getting from node %d (%d,%d) {dir %d}: val %f\n", 
							//	rank, node_num - offset, i,j,index, ii,jj, k, unodes[index].u_array[0]);
							node_data_buffer[4*(node_num - offset) + k] = unodes[index].u_array[0];
							kcount++;
						} else {
							// out of bounds
							// do nothing
						}
					}
				}
			//if (operation == RECEIVE_OP)
			//	printf("[%d{%d,%d}] kcount %d\n", rank,i,j, kcount);
			} else {
				// node is not assigned to the current process
				// do nothing
			}
		}
	}
	//printf("[%d] performed (%d) operations.\n", rank, counter);
	MPI_Barrier(MPI_COMM_WORLD);
}

/**
   * @brief performs a sub-iteration (toggle for updating central/edges/corners via @update_restriction)
   */
void simulate_sub_iteration(int update_restriction, int rank, int offset, int nodes_per_process, unode_t unodes[], float node_data_buffer[]){
	// send, with central nodes update restriction
	exchange_unode_data(SEND_OP, update_restriction, rank, offset, nodes_per_process, unodes, NULL);
	// receieve, with central nodes update restriction
	exchange_unode_data(RECEIVE_OP, update_restriction, rank, offset, nodes_per_process, unodes, node_data_buffer);
	float updated;
	// update central nodes
	for (int i = 0; i < nodes_per_process; i++){
		int node_num = offset + i;
		if (update_restriction == UPDATE_CORNERS){
			// if updating corners
			if (node_num == 0) {
				// left top corner
				updated = G * node_data_buffer[4*i+2];
				update_unode(unodes+i, updated);
			} else if (node_num == GRID_SIZE * (GRID_SIZE - 1)){
				// left bottom corner
				updated = G * node_data_buffer[4*i];
				update_unode(unodes+i, updated);
			} else if (node_num == GRID_SIZE - 1 || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
				// right top and bottom corners
				updated = G * node_data_buffer[4*i+3];
				update_unode(unodes+i, updated);
			}
		} else if (update_restriction == UPDATE_EDGES){
			// if updating edges
			if (node_num % GRID_SIZE == 0){
				// left edge
				updated = G * node_data_buffer[4*i+1];
				update_unode(unodes+i, updated);
			} else if (node_num % GRID_SIZE == GRID_SIZE - 1) {
				// right edge
				updated = G * node_data_buffer[4*i+3];
				update_unode(unodes+i, updated);
			} else if(node_num < GRID_SIZE){
				// top edge
				updated = G * node_data_buffer[4*i+2];
				update_unode(unodes+i, updated);
			} else if (node_num > GRID_SIZE * (GRID_SIZE - 1)){
				// bottom edge
				updated = G * node_data_buffer[4*i];
				update_unode(unodes+i, updated);
			}
		} else if (update_restriction == UPDATE_CENTRAL){
			// if updating central nodes
			//printf ("[%d,%d] central node [%d,%d].\n", rank, node_num, unodes[i].i, unodes[i].j);
			updated = RHO * (-4) * unodes[i].u_array[0] + 2 * unodes[i].u_array[0] - (1 - ETA) * unodes[i].u_array[1];
			updated += RHO * (node_data_buffer[4 * i] + node_data_buffer[4 * i + 1] + node_data_buffer[4 * i + 2] + node_data_buffer[4 * i + 3]);
			updated /= (1 + ETA);
			update_unode(unodes+i, updated);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

/**
   * @brief initializes nodes assigned to current process
   */
void nodes_setup(int rank, unode_t unodes[], int nodes_per_process, int offset, int *perturbation_node){
	// init nodes
	for (int i = 0; i < nodes_per_process; i++){
		unodes[i].i = get_2d (offset + i, GRID_SIZE, 0);
		unodes[i].j = get_2d (offset + i, GRID_SIZE, 1);
		for (int j = 0; j < 3; j++){
			unodes[i].u_array[j] = 0;
		}
	}
	// add perturbation
	*perturbation_node = get_1d(GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE);
	if (*perturbation_node >= offset && *perturbation_node < offset + nodes_per_process){
		unodes[*perturbation_node - offset].u_array[0] = 1.0;
		// printf("[%d] perturbed at offset global/local (%d/%d)\n", rank, *perturbation_node, *perturbation_node - offset);
	}
}

/**
   * @brief tests if two floating point numbers are equal
   */
int test_equality(float a, float b, float epsilon)
{
  if (fabs(a-b) < epsilon) return 1;
  return 0;
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

	// setup grid nodes (only those assigned to current process)
	float *node_data_buffer = (float*) malloc(nodes_per_process * 4 * sizeof(float));
	unode_t *unodes = (unode_t *) malloc(nodes_per_process * sizeof(unode_t));
	int perturbation_node;
	nodes_setup(rank, unodes, nodes_per_process, offset, &perturbation_node);

	#ifdef DEBUG
	double runtime; 
	clock_t start, end;
	if(rank==0){
		// record start time 
		start = clock();
	}
	#endif /* DEBUG */

	// run computation iterations
	int counter;
	while (iterations-- > 0){
		// update central nodes
		simulate_sub_iteration(UPDATE_CENTRAL, rank, offset, nodes_per_process, unodes, node_data_buffer);
		// update edges
		simulate_sub_iteration(UPDATE_EDGES, rank, offset, nodes_per_process, unodes, node_data_buffer);
		// update corners
		simulate_sub_iteration(UPDATE_CORNERS, rank, offset, nodes_per_process, unodes, node_data_buffer);
		// print result at the node at N/2, N/2
		if (perturbation_node >= offset && perturbation_node < offset + nodes_per_process){
			printf("%f,\n", GRID_SIZE/2, GRID_SIZE/2, unodes[perturbation_node - offset].u_array[0]);
			#ifdef DEBUG
			printf("completed %d\n", counter);
			if (!test_equality(unodes[perturbation_node - offset].u_array[0], output[counter], 0.00001)){
				printf("MISMATCH\t@%d: %f vs %f\n", counter, unodes[perturbation_node - offset].u_array[0], output[counter]);
			}
			#endif /* DEBUG */
		}
		counter++;
	}
		
	#ifdef DEBUG
	if(rank==0){
		// record end time
		end = clock();
		runtime = ((double) (end-start))/CLOCKS_PER_SEC;
		printf("Runtime is: %.23f seconds. Note that this value wont be accurate if only 1 test was run (which is default).\n", runtime);
	}
	#endif /* DEBUG */

	MPI_Barrier(MPI_COMM_WORLD);

	// free pointers
	free(node_data_buffer);
	free(unodes);

	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}