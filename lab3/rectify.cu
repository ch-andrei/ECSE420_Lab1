#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BYTES_PER_PIXEL 4
#define MAX_THREADS_PER_BLOCK 1024

/**
* CUDA kernel method to perform rectification for one pixel
*/
__global__ void rectify(unsigned char *d_image_in, unsigned index_offset)
{
	// pixel is 32bits, 8 bits for each channel (BYTES_PER_PIXEL channels: RGBA)
	// rectify RGB but not A 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	int length_offset = threadId + index_offset;
	int i = BYTES_PER_PIXEL * length_offset;
	unsigned char* d_c = d_image_in + i * sizeof(char);
	for (int j = 0; j < 3; j++){
		signed int val = (int)d_c[j];
		val -= 127;
		val = (val >= 0) ? val : 0;
		val += 127;
		d_c[j] = (unsigned char)val;
	}
}

int main(int argc, char *argv[])
{
	// get arguments from command line
	if(argc<3)
	{
		printf("Not enough arguments.\n");
		return -1;
	}

	char *argv1 = argv[1];
	char *argv2 = argv[2];
	int len1 = strlen(argv1) + 1;
	int len2 = strlen(argv2) + 1;
	const char* input_filename = (char*) malloc (len1*sizeof(char));
	strcpy((char *) input_filename, argv[1]);
	const char* output_filename = (char*) malloc (len2*sizeof(char));
	strcpy((char *) output_filename, argv[2]);

    // vars for rectifying
	unsigned char *h_image_in;
	unsigned width_in, height_in;
	unsigned total_pixels, total_chars;
	int error1 = lodepng_decode32_file(&h_image_in, &width_in, &height_in, input_filename);
	if(error1) {
		printf("error %u: %s\n", error1, lodepng_error_text(error1));
		return -2;
	}

	total_pixels = width_in * height_in;
	total_chars = total_pixels * BYTES_PER_PIXEL;

	// declare GPU memory pointer
	unsigned char* d_image_in;
	// allocate GPU memory
	cudaMalloc((void**)(&d_image_in), total_chars * sizeof(char));

	// transfer the array to the GPU
	cudaMemcpy(d_image_in, h_image_in, total_chars, cudaMemcpyHostToDevice);

	// setup threads
	int threads_x, threads_y, blocks_x, blocks_y;
	int threads_per_block;
	#ifndef DEBUG 
	threads_per_block = MAX_THREADS_PER_BLOCK;
	#else
	threads_per_block = 1;
	while (threads_per_block <= MAX_THREADS_PER_BLOCK){
		#endif /* DEBUG */
		// setup threads
		threads_x = threads_per_block;
		threads_y = 1;
	
		// setup blocks
		blocks_x = total_pixels / threads_per_block;
		blocks_y = 1;
		while (blocks_x > MAX_THREADS_PER_BLOCK){
			blocks_x /= 2;
			blocks_y *= 2;
		}
	
		dim3 numThreadsPerBlock(threads_x, threads_y, 1); // 1024 threads
		dim3 numBlocks(blocks_x, blocks_y, 1);
	
		printf("Spawning {%d,%d} blocks, {%d,%d} threads each.\n", blocks_x, blocks_y, threads_x, threads_y);
	
    	GpuTimer timer;
		// start kernel
		timer.Start();
		rectify<<<numBlocks, numThreadsPerBlock>>> (d_image_in, 0);
		timer.Stop();
	    printf("{tpb:%d} Time elapsed = %g ms\n", threads_per_block, timer.Elapsed());
	    #ifdef DEBUG
	    threads_per_block *= 2;
	}
	#endif /* DEBUG */

	// rectify leftover
	int leftover = total_pixels - (threads_x * threads_y * blocks_x * blocks_y); // this will necessarily be less than 1024
	printf("Leftover %d.\n",leftover);
	int blocks = 1;
	while (leftover > MAX_THREADS_PER_BLOCK){
		leftover /= 2;
		blocks *= 2;
	}
	rectify<<<blocks, leftover>>> (d_image_in, total_pixels - leftover);

	// copy back the result array to the CPU
	cudaMemcpy(h_image_in, d_image_in, total_chars, cudaMemcpyDeviceToHost);
	
	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, h_image_in, width_in, height_in);

	free(h_image_in);
	free((char*)input_filename);
	free((char*)output_filename);
	cudaFree(d_image_in);
	return 0;
}
