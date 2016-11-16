#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POOL_BLOCK_SIZE 2
#define BYTES_PER_PIXEL 4
#define MAX_THREADS_PER_BLOCK 1024

__device__ unsigned get_block_offset(unsigned k, unsigned image_width){
	return BYTES_PER_PIXEL * POOL_BLOCK_SIZE * ( image_width * (k / (image_width / 2)) +  (k % (image_width / 2)));
}

/**
* CUDA kernal to perform pooling for one output pixel
*/
__global__ void max_pool(unsigned char *d_image_buffer, unsigned char *d_out_buffer, unsigned image_width, unsigned index_offset)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x + index_offset;
	unsigned char* c;
	unsigned max, val; 
	// pool on RGBA channels
	for (int rgba = 0; rgba < BYTES_PER_PIXEL; rgba++){
		if (rgba < 3){
			// for RGB channels
			max = 0;
			for (int i = 0; i < POOL_BLOCK_SIZE; i++){
				c = d_image_buffer + get_block_offset(index, image_width) + BYTES_PER_PIXEL * image_width * i;
				for (int j = 0; j < POOL_BLOCK_SIZE; j++){
					c += BYTES_PER_PIXEL * j;
					val = (int)c[rgba];
					max = (max < val) ? val : max;
				}
			}
			val = (unsigned char)max;
		} else {
			// for alpha channel
			val = (unsigned char)255;
		}
		d_out_buffer[BYTES_PER_PIXEL*index + rgba] = val;
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
	unsigned char *h_image_in, *h_image_out;
	unsigned width_in, height_in;
	unsigned total_pixels, total_out_pixels;
	int error1 = lodepng_decode32_file(&h_image_in, &width_in, &height_in, input_filename);
	if(error1) {
		printf("error %u: %s\n", error1, lodepng_error_text(error1));
		return -2;
	}
	unsigned width_in_halved, height_in_halved;
	width_in_halved = width_in / 2;
	height_in_halved = height_in / 2;
	total_pixels = width_in * height_in;
	total_out_pixels = width_in_halved * height_in_halved;

	h_image_out = (unsigned char*) malloc (BYTES_PER_PIXEL * total_out_pixels * sizeof(char));

	// declare GPU memory pointer
	unsigned char *d_image_in, *d_image_out;
	// allocate GPU memory
	cudaMalloc((void**)(&d_image_in), BYTES_PER_PIXEL * total_pixels * sizeof(char));
	cudaMalloc((void**)(&d_image_out), BYTES_PER_PIXEL * total_out_pixels * sizeof(char));

	// transfer the array to the GPU
	cudaMemcpy(d_image_in, h_image_in, BYTES_PER_PIXEL * total_pixels, cudaMemcpyHostToDevice);

	// setup threads
	int threads_per_block = MAX_THREADS_PER_BLOCK;
	int threads_x, threads_y;
	threads_x = threads_per_block;
	threads_y = 1;

	// setup blocks
	int blocks_x, blocks_y; 
	blocks_x = total_out_pixels / threads_per_block;
	blocks_y = 1;
	while (blocks_x > MAX_THREADS_PER_BLOCK){
		blocks_x /= 2;
		blocks_y *= 2;
	}

	dim3 numThreadsPerBlock(threads_x, threads_y, 1); // 1024 threads
	dim3 numBlocks(blocks_x, blocks_y, 1);

    GpuTimer timer;
	// start kernel
	timer.Start();
	max_pool<<<numBlocks, numThreadsPerBlock>>> (d_image_in, d_image_out, width_in, 0);
	timer.Stop();
    printf("Time elapsed = %g ms\n", timer.Elapsed());

	// pull leftover
	int leftover = total_out_pixels - (threads_x * threads_y * blocks_x * blocks_y); // this will necessarily be less than 1024
	printf("leftover %d.\n",leftover);
	int blocks = 1;
	while (leftover - (blocks - 1) * MAX_THREADS_PER_BLOCK > MAX_THREADS_PER_BLOCK){
		blocks++;
	}
	max_pool<<<blocks, leftover>>> (d_image_in, d_image_out, width_in, total_out_pixels - leftover);

	// copy back the result array to the CPU
	cudaMemcpy(h_image_out, d_image_out, BYTES_PER_PIXEL * total_out_pixels, cudaMemcpyDeviceToHost);

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, h_image_out, width_in_halved, height_in_halved);

	free(h_image_in);
	free(h_image_out);
	free((char*)input_filename);
	free((char*)output_filename);
	cudaFree(d_image_in);
	cudaFree(d_image_out);	
	return 0;
}
