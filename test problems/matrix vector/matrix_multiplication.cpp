#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* matrix-acc-check.c */
#define SIZE 1500
 
int main()
{
	int i, j, k;
	double t1, t2, gpu_time, cpu_time;
	double **a, **b, **c, **seq;
	
	a = (double**) malloc(SIZE*sizeof(double*));
	b = (double**) malloc(SIZE*sizeof(double*));
	c = (double**) malloc(SIZE*sizeof(double*));
	seq = (double**) malloc(SIZE*sizeof(double*));
	a[0] = (double*) malloc(SIZE*SIZE*sizeof(double));
	for(i = 1; i < SIZE; i++) a[i] = a[i-1] + SIZE;
	b[0] = (double*) malloc(SIZE*SIZE*sizeof(double));
	for(i = 1; i < SIZE; i++) b[i] = b[i-1] + SIZE;
	c[0] = (double*) malloc(SIZE*SIZE*sizeof(double));
	for(i = 1; i < SIZE; i++) c[i] = c[i-1] + SIZE;
	seq[0] = (double*) malloc(SIZE*SIZE*sizeof(double));
	for(i = 1; i < SIZE; i++) seq[i] = seq[i-1] + SIZE;
	
	// Initialize matrices.
	for(i = 0; i < SIZE; ++i) 
	{
		for (j = 0; j < SIZE; ++j) 
		{
			a[i][j] = (double)i + j;
			b[i][j] = (double)i - j;
			c[i][j] = 0.0;
		}
	}
	
	t1 = clock(); 
	// Compute matrix multiplication.
	#pragma acc kernels copyin(a,b) copy(c)
	for (i = 0; i < SIZE; ++i) 
		for (j = 0; j < SIZE; ++j) 
			for (k = 0; k < SIZE; ++k) 
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
	t2 = clock();
	gpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	// ****************
	// double-check the OpenACC result sequentially on the host
	// ****************
	// Initialize the seq matrix
	for(i = 0; i < SIZE; ++i) 
		for(j = 0; j < SIZE; ++j) 
			seq[i][j] = 0.0;
	
	t1 = clock();
	// Perform the multiplication
	for (i = 0; i < SIZE; ++i) 
		for (j = 0; j < SIZE; ++j) 
			for (k = 0; k < SIZE; ++k) 
				seq[i][j] = seq[i][j] + a[i][k] * b[k][j];
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	// check all the OpenACC matrices
	for (i = 0; i < SIZE; ++i)
		for (j = 0; j < SIZE; ++j)
			if(c[i][j] != seq[i][j]) 
				{
					printf("Error %d %d\n", i,j);
				}
	printf("OpenACC matrix multiplication test was successful!\n");
	printf("gpu times = %f, cpu times = %f \n", gpu_time, cpu_time); 
	return 0;
}
