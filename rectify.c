#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
//#include <sys/types.h>
//#include <sys/timers.h>

#define BYTES_PER_PIXEL 4

/**
* TODO comment this
*/
typedef struct {
	unsigned char *image_buffer;
	unsigned length;
	unsigned length_offset;
} thread_arg_t;

/**
* TODO comment this
*/
void *rectify_array(void *arg)
{
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	unsigned char* image = thread_arg->image_buffer;
	unsigned length = thread_arg->length;
	unsigned length_offset = thread_arg->length_offset;

	// pixel is 32bits, 8 bits for each channel (BYTES_PER_PIXEL channels: RGBA)
	// rectify RGB but not A 
	for (int i = BYTES_PER_PIXEL * length_offset; i < BYTES_PER_PIXEL * (length_offset + length); i += BYTES_PER_PIXEL) 
	{
		unsigned char* c = image + i * sizeof(char);
		for (int j = 0; j < 3; j++){
			signed int val = (int)c[j];
			val -= 127;
			val = (val >= 0) ? val : 0;
			val += 127;
			c[j] = (unsigned char)val;
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
			"./rectify <name of input png> <name of output png> <# threads>\n");
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

    // for rectifying
	unsigned char *image_buffer;
	unsigned width_in, height_in;
	unsigned total_pixels, pixels_per_thread;
	signed leftover_pixels;

	int error1 = lodepng_decode32_file(&image_buffer, &width_in, &height_in, input_filename);
	if(error1) {
		printf("error %u: %s\n", error1, lodepng_error_text(error1));
		return -1;
	}

	total_pixels = width_in * height_in;
	pixels_per_thread = total_pixels / (number_of_threads);
	pixels_per_thread = (pixels_per_thread == 0) ? 1 : pixels_per_thread;
	leftover_pixels = total_pixels - pixels_per_thread * (number_of_threads); 
	printf("%d total pixels; using %d threads: %d pixels/thread and leftover %d pixels.\n", 
		total_pixels, number_of_threads, pixels_per_thread, leftover_pixels);

	pthread_t threads[number_of_threads];
	thread_arg_t thread_args[number_of_threads];

	// record start time
	// TODO
	double runtime; 
	clock_t start, end; 
	start = clock();
	printf("Start: %d \n", start);

	// perform rectifying
	for (int i = 0; i < number_of_threads && i < total_pixels; i++) {
		//printf("[thread%d]: starting index %d\n", i+1, pixels_per_thread * i);
		thread_args[i].image_buffer = image_buffer;
		thread_args[i].length = pixels_per_thread;
		thread_args[i].length_offset = pixels_per_thread * i;
		pthread_create(&threads[i], NULL, rectify_array, (void *)&thread_args[i]);
	}
	// join threads
	for (int i = 0; i < number_of_threads; i++) {
		pthread_join(threads[i], NULL);
	}
	// if image length is not a multiple of number of threads there will be someleftover bits
	if (leftover_pixels > 0){
		thread_args[0].image_buffer = image_buffer;
		thread_args[0].length = leftover_pixels;
		thread_args[0].length_offset = pixels_per_thread * (number_of_threads);
		*rectify_array((void *)&thread_args[0]);
	}

	// record ending time
	// TODO
	end = clock();
	printf("End: %d \n", end);
	runtime = ((double) (end-start))/CLOCKS_PER_SEC;
	printf("Runtime is: %.23f seconds\n", runtime);

	// save rectified pixel data to file
	lodepng_encode32_file(output_filename, image_buffer, width_in, height_in);

	free(image_buffer);
	return 0;
}
