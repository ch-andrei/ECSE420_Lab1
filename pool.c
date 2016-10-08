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
	signed blocks_offset;
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
	// get arguments from command line
	if(argc<4)
	{
		printf("Not enough arguments. Input arguments as follows:\n"
			"./pool <name of input png> <name of output png> <# threads>\n");
		return 0;
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
	
	unsigned number_of_threads = argv3;
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

	printf("%d width; %d height; %d total pixels; %d total blocks; using %d threads, computing %d blocks/thread.\n", 
		width_in, height_in, total_pixels, total_out_pixels, number_of_threads, blocks_per_thread);

	pthread_t threads[number_of_threads];
	thread_arg_t thread_args[number_of_threads];

	// record start time
	// TODO

	unsigned leftover = total_out_pixels - number_of_threads * blocks_per_thread;
	printf("leftover %d\n",leftover);

	// set up parameters
	for (unsigned i = 0; i < number_of_threads; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
		thread_args[i].image_buffer = image_buffer;
		thread_args[i].out_buffer = out_buffer;
		thread_args[i].blocks_offset = i * blocks_per_thread;
		thread_args[i].blocks = blocks_per_thread;
		thread_args[i].image_width = width_in;
	}

	if (leftover > 0){
		thread_args[0].blocks += leftover;
		for (int i = 1; i < number_of_threads; i++) {
			thread_args[i].blocks_offset += leftover;
		}
	}

	for (int i = 0; i < number_of_threads; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
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
