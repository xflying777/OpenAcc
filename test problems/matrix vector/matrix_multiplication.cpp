/* matrix-acc-func.c */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 1500

//void gpuTest(double *a, double *b, double *c, int size)
void gpuTest(double *a, double *b, double *restrict c, int size)
{
	int i,j,k;
	// Compute matrix multiplication.
	#pragma acc data copyin(a[0:size*size],b[0:size*size]) copy(c[0:size*size])
	#pragma acc kernels
	#pragma acc loop independent
	for (i = 0; i < size; ++i) 
	{
	#pragma acc loop independent
		for (j = 0; j < size; ++j) 
		{
	#pragma acc loop seq
			for (k = 0; k < size; ++k) 
			{
				c[i*size+j] += a[i*size+k] * b[k*size+j];
			}
		}
	}
}

void cpuTest(double *a, double *b, double *seq, int size)
{
	int i,j,k;
	// Compute matrix multiplication.
	for (i = 0; i < size; ++i) 
	{
		for (j = 0; j < size; ++j) 
		{
			for (k = 0; k < size; ++k) 
			{
				seq[i*size+j] += a[i*size+k] * b[k*size+j];
			}
		}
	}
}
	
int main()
{
	int i, j;
	int size = SIZE;
	double gpu_times, cpu_times;
	clock_t t1, t2;
	double *a = (double*)malloc(sizeof(double)*size*size);
	double *b = (double*)malloc(sizeof(double)*size*size);
	double *c = (double*)malloc(sizeof(double)*size*size);
	
	
	// Initialize matrices.
	#pragma acc kernels create(a[0:size*size], b[0:size*size], c[0:size*size])
	{
	#pragma acc loop independent
	for (i = 0; i < size; ++i) 
	{
	#pragma acc loop independent
		for (j = 0; j < size; ++j) 
		{
			a[i*size+j] = (double)i + j;
			b[i*size+j] = (double)i - j;
			c[i*size+j] = 0.0;
		}	
	}
	}
	
	t1 = clock();
	gpuTest(a, b, c, size);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	printf("gpu times = %f \n", gpu_times);
	
	
	// ****************
	// double-check the OpenACC result sequentially on the host
	// ****************
	double *seq= (double*)malloc(sizeof(double)*size*size);
	// Initialize the seq matrix
	for(i = 0; i < size; ++i) 
		for(j = 0; j < size; ++j) 
			seq[i*SIZE+j] = 0.f;
	
	t1 = clock();
	// Perform the multiplication
	cpuTest(a, b, seq, size);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	printf("cpu times = %f \n", cpu_times);
	
	
	// check all the OpenACC matrices
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			if(c[i*size+j] != seq[i*size+j]) 
			{
				printf("Error (%d %d) (%g, %g)\n", i,j, c[i*size+j], seq[i*size+j]);
				exit(1);
			}
	printf("OpenACC matrix multiplication test was successful!\n");
	printf("cpu times / gpu times = %f \n",cpu_times/gpu_times);
	return 0;
}
