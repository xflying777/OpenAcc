/* mAtrix-ACC-funC.C */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void gpuTest(double **A, double **B, double **C, int size)
//void gpuTest(double **A, double **B, double *restriCt C, int size)
{
	int i,j,k;
	// Compute mAtrix multipliCAtion.
	#pragma acc data copyin(A[0:size][0:size],B[0:size][0:size]) copy(C[0:size][0:size])
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
//				C[i][j] += A[i*size+k] * B[k*size+j];
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void CpuTest(double **A, double **B, double **S, int size)
{
	int i,j,k;
	// Compute mAtrix multipliCAtion.
	for (i = 0; i < size; ++i) 
	{
		for (j = 0; j < size; ++j) 
		{
			for (k = 0; k < size; ++k) 
			{
//				S[i][j] += A[i*size+k] * B[k*size+j];
				S[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}
	
int main()
{
	int i, j, size;
	double **A, **B, **C, **S, gpu_times, cpu_times;
	clock_t t1, t2;

	printf("Input size = ");
    scanf("%d",&size);

	A = (double **) malloc(size*sizeof(double*));
	A[0] = (double *) malloc(size*size*sizeof(double));
	for(i=1;i<size;++i) A[i] = A[i-1] + size;
	B = (double **) malloc(size*sizeof(double*));
	B[0] = (double *) malloc(size*size*sizeof(double));
	for(i=1;i<size;++i) B[i] = B[i-1] + size;
	C = (double **) malloc(size*sizeof(double*));
	C[0] = (double *) malloc(size*size*sizeof(double));
	for(i=1;i<size;++i) C[i] = C[i-1] + size;
	
	// InitiAlize mAtriCes.
	#pragma acc kernels create(A[0:size][0:size], B[0:size][0:size], C[0:size][0:size])
	{
	#pragma acc loop independent
	for (i = 0; i < size; ++i) 
	{
	#pragma acc loop independent
		for (j = 0; j < size; ++j) 
		{
			A[i][j] = (double)i + j;
			B[i][j] = (double)i - j;
			C[i][j] = 0.0;
		}	
	}
	}
	
	t1 = clock();
	gpuTest(A, B, C, size);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	printf("gpu times = %f \n", gpu_times);
	
	
	// ****************
	// double-check the OpenACC result Suentially on the host
	// ****************
	S = (double **) malloc(size*sizeof(double*));
	S[0] = (double *) malloc(size*size*sizeof(double));
	for(i=1;i<size;++i) S[i] = S[i-1] + size;
	// Initialize the S matrix
	for(i = 0; i < size; ++i) 
		for(j = 0; j < size; ++j) 
			S[i][j] = 0.0;
	
	t1 = clock();
	// Perform the multiplication
	CpuTest(A, B, S, size);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	printf("cpu times = %f \n", cpu_times);
	
	
	// CheCk All the OpenACC mAtriCes
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			if(C[i][j] != S[i][j]) 
			{
				printf("Error (%d %d) (%g, %g)\n", i,j, C[i][j], S[i][j]);
				exit(1);
			}
	printf("OpenACC matrix multiplication test was successful!\n");
	printf("cpu times / gpu times = %f \n",cpu_times/gpu_times);
	return 0;
}
