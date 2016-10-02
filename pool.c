#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define BLOCK_SIZE 2
#define BYTES_PER_PIXEL 4

/**
* TODO comment this
*/
typedef struct {
	unsigned char *image_buffer;
	unsigned char *out_buffer;
	unsigned blocks;
	unsigned blocks_offset;
	unsigned image_width;
} thread_arg_t;

/**
* TODO comment this
*/
void *max_pool(void *arg)
{
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	unsigned char *image_buffer = thread_arg->image_buffer;
	unsigned char *out_buffer = thread_arg->out_buffer;
	unsigned blocks = thread_arg->blocks;
	unsigned blocks_offset = thread_arg->blocks_offset;
	unsigned image_width = thread_arg->image_width;

	unsigned increment;
	unsigned char* c;
	unsigned max, val; 
	for (int k = blocks_offset; k < blocks_offset + blocks; k++) 
	{
		for (int rgba = 0; rgba < BYTES_PER_PIXEL; rgba++){
			increment = (8 * image_width * (k / (image_width / 2))) 
					+ 8 * (k % (image_width / 2));
			c = image_buffer + increment;
			max = 0;
			for (int i = 0; i < BLOCK_SIZE; i++){
				c += BYTES_PER_PIXEL * image_width * i;
				for (int j = 0; j < BLOCK_SIZE; j++){
					c += BYTES_PER_PIXEL * j;
					val = (int)c[rgba];
					max = (max < val) ? val : max;
				}
			}
			out_buffer[BYTES_PER_PIXEL*k + rgba] = (unsigned char)max;
		}
	}
}

/**
* TODO add comments inside main
*/
int main(int argc, char *argv[])
{
	// TODO get arguments from argv
	unsigned char input_filename[] = "test.png";
	unsigned char output_filename[] = "test_pool1.png"; 
	// TODO fix output_filename to be input_filename without .png + "_rectify.png"
	
	unsigned number_of_threads = 8; // TODO get from command line
	// *******************************

    // vars for pooling
	unsigned char *image_buffer, *out_buffer;
	unsigned width_in, height_in;
	unsigned total_pixels, total_out_pixels, blocks_per_thread;

	int error1 = lodepng_decode32_file(&image_buffer, &width_in, &height_in, input_filename);
	if(error1) {
		printf("error %u: %s\n", error1, lodepng_error_text(error1));
		return -1;
	}

	total_pixels = width_in * height_in;
	total_out_pixels = total_pixels / BYTES_PER_PIXEL;
	blocks_per_thread = total_out_pixels / number_of_threads;
	blocks_per_thread = (blocks_per_thread == 0) ? 1 : blocks_per_thread;

	out_buffer = (unsigned char*) malloc(BYTES_PER_PIXEL * total_out_pixels);

	printf("%d total pixels; %d total blocks; using %d threads, computing %d blocks/thread.\n", 
		total_pixels, total_out_pixels, number_of_threads, blocks_per_thread);

	pthread_t threads[number_of_threads];
	thread_arg_t thread_args[number_of_threads];

	// record start time
	// TODO

	// perform 2x2 max pooling
	for (int i = 0; i < number_of_threads && i < total_out_pixels; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
		thread_args[i].image_buffer = image_buffer;
		thread_args[i].out_buffer = out_buffer;
		thread_args[i].blocks = blocks_per_thread;
		thread_args[i].blocks_offset = blocks_per_thread * i;
		thread_args[i].image_width = width_in;
		pthread_create(&threads[i], NULL, max_pool, (void *)&thread_args[i]);
	}
	// join threads
	for (int i = 0; i < number_of_threads; i++) {
		pthread_join(threads[i], NULL);
	}

	// record ending time
	// TODO

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, out_buffer, width_in / 2, height_in / 2);

	free(out_buffer);
	free(image_buffer);
	return 0;
}
