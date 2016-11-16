#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wm.h"

#define BLOCK_SIZE 3
#define BYTES_PER_PIXEL 4
#define MAX_THREADS_PER_BLOCK 1024

#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

/**
* gets pointer to the pixel (computes offset based on i, j index of the pixel)
*/
__device__ unsigned char* get_pixel_pointer(unsigned i, unsigned j, unsigned char* image_ptr, unsigned image_width)
{
	return (image_ptr + BYTES_PER_PIXEL * get_1d(i,j,image_width));
}

/**
* get 1-dimensional offset of a 2x2 block at index i,j
*/
__device__ unsigned get_block_offset(unsigned i, unsigned j, unsigned image_width)
{
	return ((i + 1) * image_width + j + 1);
}

/**
* converts offset from input picture to output picture (need this because output has different dimensions)
*/
__device__ unsigned convert_block_to_pixel_offset(unsigned blocks_offset, unsigned image_width)
{
	int i = get_2d(blocks_offset, (image_width - 2), 0);
	int j = get_2d(blocks_offset, (image_width - 2), 1);
	return get_1d(i,j,image_width);
}

/**
* method to perform convolution by a given thread
*/
__global__ void convolve(unsigned char *d_image_buffer, unsigned char *d_out_buffer, unsigned image_width, unsigned image_height, float* weights, unsigned index_offset)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	int blocks_offset = threadId + index_offset;
	blocks_offset = convert_block_to_pixel_offset(blocks_offset, image_width);
	int i = get_2d(blocks_offset, image_width, 0);
	int j = get_2d(blocks_offset, image_width, 1);
	unsigned char* c;
	unsigned val;
	signed convolved;
	// convolve for RGBA channels
	for (int rgba = 0; rgba < BYTES_PER_PIXEL; rgba++) {
		if (rgba != 3){
			// for RGB channels: convolve 
			convolved = 0;	
			for (int ii = 0; ii < BLOCK_SIZE; ii++){
				for (int jj = 0; jj < BLOCK_SIZE; jj++){
					c = get_pixel_pointer(i+ii, j+jj, d_image_buffer, image_width); // not i+ii-1 because using i', where i' = i - 1 already
					val = c[rgba];
					convolved += val * weights[ii*3+jj];
				}
			}
			// clamp
			convolved = (convolved < 0) ? 0 : convolved;
			convolved = (convolved > 255) ? 255 : convolved;
		} else {
			// for alpha channel: dont convolve, just set 255
			convolved = 255;
		}
		// store to output array
		c = get_pixel_pointer(i, j, d_out_buffer, image_width - 2);
		c[rgba] = convolved;
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

	// compute work distribution
	total_pixels = width_in * height_in;
	total_out_pixels = (width_in - 2) * (height_in - 2);
	
	h_image_out = (unsigned char*) malloc(BYTES_PER_PIXEL * total_out_pixels);

	// declare GPU memory pointer
	unsigned char *d_image_in, *d_image_out;
	// allocate GPU memory
	cudaMalloc((void**)(&d_image_in), BYTES_PER_PIXEL * total_pixels * sizeof(char));
	cudaMalloc((void**)(&d_image_out), BYTES_PER_PIXEL * total_out_pixels * sizeof(char));

	float *d_weights;
	cudaMalloc((void**)(&d_weights), 9 * sizeof(float));

	// transfer the array to the GPU
	cudaMemcpy(d_image_in, h_image_in, BYTES_PER_PIXEL * total_pixels, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, w, 9 * sizeof(float), cudaMemcpyHostToDevice);

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
	convolve<<<numBlocks, numThreadsPerBlock>>> (d_image_in, d_image_out, width_in, height_in, d_weights, 0);
	timer.Stop();
    printf("Time elapsed = %g ms\n", timer.Elapsed());

	int leftover = total_out_pixels - (threads_x * threads_y * blocks_x * blocks_y); // this will necessarily be less than 1024
	printf("leftover %d.\n",leftover);
	int blocks = 1;
	while (leftover - (blocks - 1) * MAX_THREADS_PER_BLOCK > MAX_THREADS_PER_BLOCK){
		blocks++;
	}
	convolve<<<blocks, leftover>>> (d_image_in, d_image_out, width_in, height_in, d_weights, total_out_pixels - leftover);

	// copy back the result array to the CPU
	cudaMemcpy(h_image_out, d_image_out, BYTES_PER_PIXEL * total_out_pixels, cudaMemcpyDeviceToHost);

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, h_image_out, width_in - 2, height_in - 2);

	free(h_image_in);
	free(h_image_out);
	free((char*)input_filename);
	free((char*)output_filename);
	cudaFree(d_image_in);
	cudaFree(d_image_out);
	cudaFree(d_weights);
	return 0;
}