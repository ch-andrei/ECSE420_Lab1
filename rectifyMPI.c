#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

#define BYTES_PER_PIXEL 4

int main(int argc, char *argv[])
{
	if(argc<3)
	{
		printf("Not enough arguments. Input arguments as follows:\n"
			"./rectify <name of input png> <name of output png> <# threads>\n");

		return 0;
	}
	char *argv1 = argv[1];
	char *argv2 = argv[2];
	int len1 = strlen(argv1)+1;
	int len2 = strlen(argv2)+1;

	unsigned char input_filename[len1];
	strcpy((char *) input_filename, argv[1]);
	unsigned char output_filename[len2]; 
	strcpy((char *) output_filename, argv[2]);

	// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	// Get the number of processes
	int number_of_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// vars for rectifying
	unsigned char *image_buffer;
	unsigned width_in, height_in;
	unsigned total_pixels, pixels_per_process;

	int error;
	if (rank == 0) {
		error = lodepng_decode32_file(&image_buffer, &width_in, &height_in, input_filename);
	}
	// sync barrier
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
		// Finalize the MPI environment.
		MPI_Finalize();
		return -1;
	}

	MPI_Bcast(&width_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&height_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
	total_pixels = width_in * height_in;
	pixels_per_process = total_pixels / (number_of_processes);

	// share the entire image array (would be smarter to share only a part but thats too much work for me)
	if (rank == 0) {
		for (int i = 1; i < number_of_processes; i++){
			// process 0 sends to each other process
			MPI_Send(image_buffer, total_pixels * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
		}
	} else {
		// Allocate a buffer to hold the incoming numbers
		image_buffer = (unsigned char*)malloc(sizeof(char) * total_pixels * BYTES_PER_PIXEL);
		// Now receive the message with the allocated buffer
		MPI_Recv(image_buffer, total_pixels * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, 0, 0,
		         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// compute lengths for each process
	int length_offset = pixels_per_process * rank;
	// if total pixels not multiple of number of processes, there will be leftover pixels due to truncating
	int leftover = total_pixels - number_of_processes * pixels_per_process; 
	if (leftover > 0){
		if (rank == 0){
			pixels_per_process += leftover; // adjust processs 0 work load
		} else {
			length_offset += leftover; // adjust other processes offset 
		}
	}

	// rectify
	// pixel is 32bits, 8 bits for each channel (BYTES_PER_PIXEL channels: RGBA)
	// rectify RGB but not A 
	for (int i = BYTES_PER_PIXEL * length_offset; i < BYTES_PER_PIXEL * (length_offset + pixels_per_process); i += BYTES_PER_PIXEL) 
	{
		unsigned char* c = image_buffer + i;
		for (int j = 0; j < 3; j++){
			signed int val = (int)c[j];
			val -= 127;
			val = (val >= 0) ? val : 0;
			val += 127;
			c[j] = (unsigned char)val;
		}
	}

	// sync barrier
	MPI_Barrier(MPI_COMM_WORLD); 

	// send the results results back to process 0
	if (rank == 0) {
		// process 0 loops receiving data from other processses
		pixels_per_process -= leftover; // without leftover, otherwise will try to access out of bounds memory -> segmentation fault
		for (int i = 1; i < number_of_processes; i++){
			MPI_Recv((image_buffer + BYTES_PER_PIXEL * (pixels_per_process * i + leftover)), BYTES_PER_PIXEL * pixels_per_process, MPI_UNSIGNED_CHAR, i, 0,
			         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	} else {
		// process non 0 sends data
		MPI_Send((image_buffer + BYTES_PER_PIXEL * length_offset),  BYTES_PER_PIXEL * pixels_per_process, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
	}

	// save rectified pixel data to file
	if (rank == 0){
		lodepng_encode32_file(output_filename, image_buffer, width_in, height_in);
	}

	free(image_buffer);

	// Finalize the MPI environment.
	MPI_Finalize();

	return 0;
}