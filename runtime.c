#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

int main()
{
	int n=100;
	for(int i=0; i<n; i++)
	{
		system("./rectify test.png rectify.png 2 >> runtime.txt");
	}

	FILE *f = fopen("runtime.txt", "r");
	float sum = 0.0;
	for(int i=0; i<n; i++)
	{
		float line;
		fscanf(f, "%f", &line);
		sum += line;
	}
	printf("sum: %.23f\n", sum);
	sum/=n;
	printf("average: %.23f\n", sum);
	fclose(f);
}