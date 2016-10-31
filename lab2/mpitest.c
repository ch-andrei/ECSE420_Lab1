#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[]){
		// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	// Get the number of processes
	int number_of_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	float number;

	for (int i = 0; i < number_of_processes; i++){
		if (rank != i){
			float num = 0.1;
			MPI_Send(&num, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			printf("[%d] sending\n", rank);
		}
	}

	for (int i = 0; i < number_of_processes; i++){
		if (rank != i){
			float num;
			MPI_Recv(&num, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			number += num;
			printf("[%d] receiving\n", rank);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("[%d] number = %f\n", rank, number);


	// Finalize the MPI environment and return
	MPI_Finalize();
	return 0;
}