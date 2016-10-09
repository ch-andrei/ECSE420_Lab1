#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include "wm.h"
#include <unistd.h>
#include <time.h>

/**
* TODO comment this
*/
#define BLOCK_SIZE 3
#define BYTES_PER_PIXEL 4

/**
* TODO comment this
*/
#define get_1d(i, j, width) ((i)*(width)+(j))
#define get_2d(index, width, ij) (((ij)==0)?((index)/(width)):((index)%(width)))

/**
* TODO comment this
*/
typedef struct {
	unsigned char *image_buffer;
	unsigned char *out_buffer;
	unsigned blocks;
	unsigned blocks_offset;
	unsigned image_width;
	unsigned image_height;
	unsigned id;
} thread_arg_t;

/**
* TODO comment this
*/
unsigned char* get_pixel_pointer(unsigned i, unsigned j, unsigned char* image_ptr, unsigned image_width)
{
	return (image_ptr + BYTES_PER_PIXEL * get_1d(i,j,image_width));
}

/**
* TODO comment this
*/
unsigned get_block_offset(unsigned i, unsigned j, unsigned image_width)
{
	return ((i + 1) * image_width + j + 1);
}

/**
* TODO comment this
*/
unsigned convert_block_to_pixel_offset(unsigned blocks_offset, unsigned image_width)
{
	int i = get_2d(blocks_offset, (image_width - 2), 0);
	int j = get_2d(blocks_offset, (image_width - 2), 1);
	return get_1d(i,j,image_width);
}

/**
* TODO comment this
*/
void *convolve(void *arg)
{
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	unsigned char *image_buffer = thread_arg->image_buffer;
	unsigned char *out_buffer = thread_arg->out_buffer;
	int blocks = thread_arg->blocks;
	unsigned blocks_offset = thread_arg->blocks_offset;
	unsigned image_width = thread_arg->image_width;
	unsigned image_height = thread_arg->image_height;

	blocks_offset = convert_block_to_pixel_offset(blocks_offset, image_width);
	int i = get_2d(blocks_offset, image_width, 0);
	int j = get_2d(blocks_offset, image_width, 1);

	//printf("[id%d], start: i %d, j %d; started with blocks left %d\n", thread_arg->id, i,j, blocks);

	unsigned char* c;
	unsigned val;
	signed convolved;
	while (blocks > 0){
		if (j == image_width-2){
			i++;
			j = 0;
		}
		for (int rgba = 0; rgba < BYTES_PER_PIXEL; rgba++) {
			if (rgba != 3){
				// for RGB channels
				convolved = 0;	
				for (int ii = 0; ii < BLOCK_SIZE; ii++){
					for (int jj = 0; jj < BLOCK_SIZE; jj++){
						c = get_pixel_pointer(i+ii, j+jj, image_buffer, image_width); // not i+ii-1 because using i' where i' = i - 1 already
						val = c[rgba];
						convolved += val * w[ii][jj];
					}
				}
				// clamp
				convolved = (convolved < 0) ? 0 : convolved;
				convolved = (convolved > 255) ? 255 : convolved;
			} else {
				// for alpha channel; dont convolve, just set 255
				convolved = 255;
			}
			// store to output array
			c = get_pixel_pointer(i, j, out_buffer, image_width - 2);
			c[rgba] = convolved;
		}
		// increment
		blocks--;
		j++;
	}
	//printf("[id%d], end: i %d, j %d; blocks left %d\n", thread_arg->id, i,j, blocks);
	//pthread_exit(NULL);
}

/**
* TODO add comments inside main
*/
int main(int argc, char *argv[])
{
	// get arguments from command line
	if(argc<4)
	{
		puts("Error: Not enough arguments. Input arguments as follows:\n"
			"./convolve <name of input png> <name of output png> <# threads>");
		return -1;
	}

	char *argv1 = argv[1];
	char *argv2 = argv[2];
	int argv3 = atoi(argv[3]);

	int len1 = strlen(argv1)+1;
	int len2 = strlen(argv2)+1;

	unsigned char input_filename[len1];
	strcpy((char *) input_filename, argv[1]);

	unsigned char output_filename[len2]; 
	strcpy((char *) output_filename, argv[2]);
	
	int number_of_threads = argv3;
	if (number_of_threads < 1){
		puts("Error: Invalid number of threads. Terminating.");
		return -1;
	}
	// *******************************

    // vars for convolution
	unsigned char *image_buffer, *out_buffer;
	unsigned width_in, height_in;
	unsigned total_pixels, total_out_pixels, blocks_per_thread;

	int error1 = lodepng_decode32_file(&image_buffer, &width_in, &height_in, input_filename);
	if(error1) {
		printf("Error %u: %s\n", error1, lodepng_error_text(error1));
		return -1;
	}

	total_pixels = width_in * height_in;
	total_out_pixels = (width_in - 2) * (height_in - 2);
	blocks_per_thread = total_out_pixels / number_of_threads;
	blocks_per_thread = (blocks_per_thread == 0) ? 1 : blocks_per_thread;

	out_buffer = (unsigned char*) malloc(BYTES_PER_PIXEL * total_out_pixels);

	printf("%d width; %d height; %d total pixels; %d total blocks; using %d threads, computing %d blocks/thread.\n", 
		width_in, height_in, total_pixels, total_out_pixels, number_of_threads, blocks_per_thread);

	pthread_t threads[number_of_threads];
	thread_arg_t thread_args[number_of_threads];

	// record start time
	// TODO
	double runtime; 
	clock_t start, end; 
	start = clock();
	printf("Start: %d \n", start);

	unsigned leftover = total_out_pixels - number_of_threads * blocks_per_thread;
	printf("leftover %d\n",leftover);

	// perform convolution
	for (int i = 0; i < number_of_threads; i++) {
		thread_args[i].image_buffer = image_buffer;
		thread_args[i].out_buffer = out_buffer;
		thread_args[i].blocks = blocks_per_thread; // -248
		thread_args[i].blocks_offset = blocks_per_thread * i;
		thread_args[i].image_width = width_in;
		thread_args[i].image_height = height_in;
		thread_args[i].id = i;
	}

	if (leftover > 0){
		thread_args[0].blocks += leftover;
		for (int i = 1; i < number_of_threads; i++) {
			thread_args[i].blocks_offset += leftover;
		}
	}

	for (int i = 0; i < number_of_threads && i < total_out_pixels; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
		pthread_create(&threads[i], NULL, convolve, (void *)&thread_args[i]);
	}
	// join threads
	for (int i = 0; i < number_of_threads; i++) {
		pthread_join(threads[i], NULL);
	}

	// record ending time
	// TODO
	end = clock();
	printf("End: %d \n", end);
	runtime = ((double) (end-start))/CLOCKS_PER_SEC;
	printf("Runtime is: %.23f seconds\n", runtime);

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, out_buffer, width_in - 2, height_in - 2);

	free(out_buffer);
	free(image_buffer);
	return 0;
}