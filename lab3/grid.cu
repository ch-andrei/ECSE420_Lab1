#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "output.h"

// Note: the value of the "eta" constant should be 2e-4, the value of the "rho" constant should be 0.5, and the value of the "G" constant should be 0.75.

#ifdef RUNTIME_SMALL
#define GRID_SIZE		4
#else
#define GRID_SIZE		512
// GRID_SIZE must be an integer divisible by 2
#endif

#define MAX_THREADS_PER_BLOCK 1024

#define U_SIZE			2
#define ETA				0.0002
#define RHO				0.5
#define G				0.75

#define UPDATE_CORNERS	0
#define UPDATE_EDGES	1
#define UPDATE_CENTRAL	2

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

typedef struct unode_t{
	float u_array[U_SIZE];
} unode_t;

/**
   * @brief updates unode values
   */
__device__ void update_unode(unode_t *unode, float new_u){
	unode->u_array[1] = unode->u_array[0];
	unode->u_array[0] = new_u;
}

/**
   * @brief Updates indexes ii, jj based on the direction k and initial position i,j
   */
__device__ int get_index_by(int k, int i, int j, int *ii, int *jj){
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
	if (*ii < 0 || *jj < 0 || *ii >= GRID_SIZE || *jj >= GRID_SIZE){
		return 1;
	} else {
		return 0;
	}
}

/**
   * @brief tests if two floating point numbers are equal
   */
__host__ int test_equality(float a, float b, float epsilon)
{
  if (fabs(a-b) < epsilon) return 1;
  return 0;
}

/**
   * @brief performs a sub-iteration (toggle for updating central/edges/corners via @update_restriction)
   */
__global__ void simulate_sub_iteration(int update_restriction, unode_t *d_unodes, unode_t *d_temp){
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;		 	
	int node_num = blockId * blockDim.x + threadIdx.x;
	int i = get_2d(node_num, GRID_SIZE, 0);
	int j = get_2d(node_num, GRID_SIZE, 1);

	/*
	#ifdef DEBUG
		if (i == GRID_SIZE/2 && j == GRID_SIZE/2)
			printf("working on blockid %d {%d, %d} and threadId {%d,%d} resulting in node #%d at {%d,%d}\n", 
				blockId,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,node_num,i,j);
	#endif
	*/

	// if want to optimize using __shared__ memory, this will go here
	/*
	__shared__ shmem_unodes[MAX_THREADS_PER_BLOCK];
	// TODO setup shmem_unodes 
	__syncthreads();
	*/

	// get adjacent nodes' values and store locally 
	float node_data_buffer[4];
	for (int k = 0; k < 4; k++){
		int ii, jj;
		if (get_index_by(k, i, j, &ii, &jj)){
			node_data_buffer[k] = 0; // this wont be used, but set to 0 anyway
			continue;
		}
		int node_adj = get_1d(ii,jj,GRID_SIZE); 
		node_data_buffer[k] = d_unodes[node_adj].u_array[0];
	}
	float u0 = d_unodes[node_num].u_array[0];
	float u1 = d_unodes[node_num].u_array[1];
	__syncthreads();

	float updated;
	// update central nodes
	if (update_restriction == UPDATE_CORNERS){
		// if updating corners
		if (node_num == 0) {
			// left top corner
			updated = G * node_data_buffer[2];
			update_unode(&(d_temp[node_num]), updated);
		} else if (node_num == GRID_SIZE * (GRID_SIZE - 1)){
			// left bottom corner
			updated = G * node_data_buffer[0];
			update_unode(&(d_temp[node_num]), updated);
		} else if (node_num == GRID_SIZE - 1 || node_num == (GRID_SIZE * GRID_SIZE - 1)) {
			// right top and bottom corners
			updated = G * node_data_buffer[3];
			update_unode(&(d_temp[node_num]), updated);
		}
	} else if (update_restriction == UPDATE_EDGES){
		// if updating edges
		if (node_num % GRID_SIZE == 0){
			// left edge
			updated = G * node_data_buffer[1];
			update_unode(&(d_temp[node_num]), updated);
		} else if (node_num % GRID_SIZE == GRID_SIZE - 1) {
			// right edge
			updated = G * node_data_buffer[3];
			update_unode(&(d_temp[node_num]), updated);
		} else if(node_num < GRID_SIZE){
			// top edge
			updated = G * node_data_buffer[2];
			update_unode(&(d_temp[node_num]), updated);
		} else if (node_num > GRID_SIZE * (GRID_SIZE - 1)){
			// bottom edge
			updated = G * node_data_buffer[0];
			update_unode(&(d_temp[node_num]), updated);
		}
	} else if (update_restriction == UPDATE_CENTRAL){
		// if updating central nodes
		updated = (RHO * (-4) + 2) * u0 - (1 - ETA) * u1;
		updated += RHO * (node_data_buffer[0] + node_data_buffer[1] + node_data_buffer[2] + node_data_buffer[3]);
		updated /= (1 + ETA);
		update_unode(&(d_temp[node_num]), updated);
	}
}

__host__ void update_unodes(unode_t *d_unodes, unode_t *d_temp, int node_count){
	cudaMemcpy(d_unodes, d_temp, node_count * sizeof(unode_t), cudaMemcpyDeviceToDevice);
}

__host__ void getVal(float *val, unode_t *d_unodes){
	int i = GRID_SIZE/2;
	int j = i;
	int index = get_1d(i,j,GRID_SIZE);
	cudaMemcpy(val, &(d_unodes[index].u_array[0]), sizeof(float), cudaMemcpyDeviceToHost);
}

/**
   * @brief initializes nodes assigned to current process
   */
__host__ void nodes_setup(unode_t *unodes){
	// init nodes
	for (int i = 0; i < GRID_SIZE; i++){
		for (int j = 0; j < GRID_SIZE; j++){
			int node_num = get_1d(i,j,GRID_SIZE);
			for (int k = 0; k < U_SIZE; k++){
				unodes[node_num].u_array[k] = 0;
			}
		}
	}
	unodes[get_1d(GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE)].u_array[0] = 1;
}

int main(int argc, char *argv[])
{
	if(argc<1)
	{
		printf("Not enough arguments.\n");
		return 0;
	}
	int iterations = atoi(argv[1]);

	unsigned node_count = GRID_SIZE * GRID_SIZE;

	// declare GPU memory pointer
	unode_t *h_unodes, *d_unodes, *d_temp;

	h_unodes = (unode_t*) malloc(node_count * sizeof(unode_t));

	// set all nodes to 0 except node in the middle
	nodes_setup(h_unodes);

	// allocate GPU memory
	if (cudaMalloc((void**)(&d_unodes), node_count * sizeof(unode_t)) != cudaSuccess){
		printf("cudaMalloc pooped over\n");
	}
	if (cudaMemcpy(d_unodes, h_unodes, node_count * sizeof(unode_t), cudaMemcpyHostToDevice) != cudaSuccess){
		printf("cudaMemcpy pooped over\n");
	}
	if (cudaMalloc((void**)(&d_temp), node_count * sizeof(unode_t)) != cudaSuccess){
		printf("cudaMalloc pooped over\n");
	}
	if (cudaMemcpy(d_temp, h_unodes, node_count * sizeof(unode_t), cudaMemcpyHostToDevice) != cudaSuccess){
		printf("cudaMemcpy pooped over\n");
	}

	// setup CUDA runtime blocks and threads nums
	int threads_per_block = MAX_THREADS_PER_BLOCK;

	int threads_x, threads_y;
	threads_x = threads_per_block;
	threads_y = 1;

	int blocks_x, blocks_y; 
	blocks_x = node_count / threads_per_block;
	blocks_y = 1;

	// for small GRID_SIZE case
	if (GRID_SIZE * GRID_SIZE < MAX_THREADS_PER_BLOCK)
	{
		blocks_x = 1;
		blocks_y = 1;
		threads_x = GRID_SIZE * GRID_SIZE;
		threads_y = 1;
	}

	// this ensures that any GRID_SIZE will work
	while (blocks_x > MAX_THREADS_PER_BLOCK){
		blocks_x /= 2;
		blocks_y *= 2;
	}

	dim3 numThreadsPerBlock(threads_x, threads_y, 1); // 1024 threads
	dim3 numBlocks(blocks_x, blocks_y, 1);

	#ifdef DEBUG
	printf("blocks [%d,%d,1]: threads [%d,%d,1]\n", blocks_x, blocks_y, threads_x, threads_y);
	#endif /* DEBUG */

	int iteration_num = 0;
	// run computation iterations
	GpuTimer timer;
	// start kernel
	timer.Start();
	while (iteration_num < iterations){
		//printf("iterating\n");
		// update central nodes
		simulate_sub_iteration<<<numBlocks, numThreadsPerBlock>>>(UPDATE_CENTRAL, d_unodes, d_temp);
		update_unodes(d_unodes, d_temp, node_count);
		// update edges
		simulate_sub_iteration<<<numBlocks, numThreadsPerBlock>>>(UPDATE_EDGES, d_unodes, d_temp);
		update_unodes(d_unodes, d_temp, node_count);
		// update corners
		simulate_sub_iteration<<<numBlocks, numThreadsPerBlock>>>(UPDATE_CORNERS, d_unodes, d_temp);
		update_unodes(d_unodes, d_temp, node_count);

		float val;
		getVal(&val, d_unodes);
		printf("%f\n", val);
		#ifdef DEBUG
		if (!test_equality(val, output[iteration_num], 0.00001)){
			printf("MISTMATCH @iteration %d: { %f } vs { %f }\n", iteration_num, val, output[iteration_num]);
		}
		#endif

		iteration_num++;
	}
	timer.Stop();
	#ifdef DEBUG
    printf("Time elapsed = %g ms\n", timer.Elapsed());
	#endif	
	// free pointers
	free(h_unodes);
	cudaFree(d_unodes);

	return 0;
}
