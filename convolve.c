#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "wm.h"

#define BLOCK_SIZE 3
#define BYTES_PER_PIXEL 4
#define get_1d(i, j, width) ((i*width+j))
#define get_2d(index, width, ij) ((ij==0)?(index/width):(index%width))

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

unsigned char* get_pixel_pointer(unsigned i, unsigned j, unsigned char* image, unsigned image_width)
{
	return (image + BYTES_PER_PIXEL * get_1d(i,j,image_width));
}

/**
* TODO comment this
*/
void *convolve(void *arg)
{
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	unsigned char *image_buffer = thread_arg->image_buffer;
	unsigned char *out_buffer = thread_arg->out_buffer;
	unsigned blocks = thread_arg->blocks;
	unsigned blocks_offset = thread_arg->blocks_offset;
	unsigned image_width = thread_arg->image_width;
	unsigned image_height = thread_arg->image_height;
	
	unsigned char* c;
	unsigned val;
	unsigned counter = 0;
	signed convolved;
	unsigned start_index_i = get_2d(blocks_offset, image_width, 0);
	unsigned start_index_j = get_2d(blocks_offset, image_width, 1);
	start_index_i = (start_index_i == 0) ? 1 : start_index_i;
	start_index_j = (start_index_j == 0) ? 2 : start_index_j;
	for (int i = start_index_i; counter < blocks; i++){
		for (int j = start_index_j; counter < blocks; j++){
			if(thread_arg->id == 0){
				printf("c%d, ", counter);
			}
			if (i == 0) {
				i++; 
				continue;
			}
			if (j == 0) continue;
			if (j == image_width - 1) {
				j = 0;
				i++;
				continue;
			}
			for (int rgba = 0; rgba < BYTES_PER_PIXEL; rgba++) {
				if (rgba != 3){
					// for RGB channels
					convolved = 0;	
					for (int ii = 0; ii < BLOCK_SIZE; ii++){
						for (int jj = 0; jj < BLOCK_SIZE; jj++){
							c = get_pixel_pointer(i+ii-1, j+jj-1, image_buffer, image_width);
							val = c[rgba];
							convolved += val * w[ii][jj];
						}
					}
					// clamp
					convolved = (convolved < 0) ? 0 : convolved;
					convolved = (convolved > 255) ? 255 : convolved;
				}
				else {
					// for alpha channel
					convolved = 255;
				}
				// store to output array
				c = get_pixel_pointer(i - 1, j - 1, out_buffer, image_width - 2);
				c[rgba] = convolved;
			}
			// increment to show that convolution for a block was successful 
			counter++;
			if (counter % 10000 == 0) printf("count %d\n", counter);
		}
	}
	printf("[id%d] starti %d, startj %d, counter %d\n", thread_arg->id, start_index_i, start_index_j, counter);
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
	
	unsigned number_of_threads = 128; // TODO get from command line
	// *******************************

    // vars for convolution
	unsigned char *image_buffer, *out_buffer;
	unsigned width_in, height_in;
	unsigned total_pixels, total_out_pixels, blocks_per_thread;

	int error1 = lodepng_decode32_file(&image_buffer, &width_in, &height_in, input_filename);
	if(error1) {
		printf("error %u: %s\n", error1, lodepng_error_text(error1));
		return -1;
	}

	total_pixels = width_in * height_in;
	total_out_pixels = (width_in - 2) * (height_in - 2);
	blocks_per_thread = total_out_pixels / number_of_threads;
	blocks_per_thread = (blocks_per_thread == 0) ? 1 : blocks_per_thread;

	out_buffer = (unsigned char*) malloc(BYTES_PER_PIXEL * total_out_pixels);

	printf("%d total pixels; %d total blocks; using %d threads, computing %d blocks/thread.\n", 
		total_pixels, total_out_pixels, number_of_threads, blocks_per_thread);

	pthread_t threads[number_of_threads];
	thread_arg_t thread_args[number_of_threads];

	// record start time
	// TODO

	// perform convolution
	for (int i = 0; i < number_of_threads && i < total_out_pixels; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
		thread_args[i].image_buffer = image_buffer;
		thread_args[i].out_buffer = out_buffer;
		thread_args[i].blocks = blocks_per_thread;
		thread_args[i].blocks_offset = blocks_per_thread * i;
		thread_args[i].image_width = width_in;
		thread_args[i].image_height = height_in;
		thread_args[i].id = i;
		pthread_create(&threads[i], NULL, convolve, (void *)&thread_args[i]);
	}
	// join threads
	for (int i = 0; i < number_of_threads; i++) {
		pthread_join(threads[i], NULL);
	}

	// record ending time
	// TODO

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, out_buffer, width_in - 2, height_in - 2);

	free(out_buffer);
	free(image_buffer);
	return 0;
}


