#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void gputest(double **A, double **B, double **C, int N);
void cputest(double **A, double **B, double **C, int N);
void error(double **C, double **D, int N);

int main()
{
	int i, j, N;
	double **A, **B, **C, **D, t1, t2, cpu_times, gpu_times;
	
	N = 1000;
	
	A = (double**) malloc(N*sizeof(double*));
	A[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) A[i] = A[i-1] + N;
	B = (double**) malloc(N*sizeof(double*));
	B[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) B[i] = B[i-1] + N;
	C = (double**) malloc(N*sizeof(double*));
	C[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) C[i] = C[i-1] + N;
	D = (double**) malloc(N*sizeof(double*));
	D[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) D[i] = D[i-1] + N;
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) 
		{
			A[i][j] = i + j;
			B[i][j] = i - j;
			C[i][j] = 0.0;
			D[i][j] = 0.0;
		}
	}		
	
	t1 = clock();
	cputest(A, B, C, N);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	gputest(A, B, D, N);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	
	printf("cpu times = %f secs \n", cpu_times);
	printf("gpu times = %f secs \n", gpu_times);
	printf("cpu times/gpu times = %f \n", cpu_times/gpu_times);
	error(C,D,N);
}

void gputest(double **A, double **B, double **C, int N)
{
	int i, j, k;
	#pragma acc kernels
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			for(k=0;k<N;k++) C[j][i] = C[j][i] + A[j][k] * B[k][j];
		}
		
	}
}

void cputest(double **A, double **B, double **C, int N)
{
	int i, j, k;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			for(k=0;k<N;k++) C[j][i] = C[j][i] + A[j][k] * B[k][j];
		}
		
	}
}

void error(double **C, double **D, int N)
{
	int i, j, k;
	double temp;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			if(D[i][j] != C[i][j])
			printf("error D[%d][%d] = %f, C[%d][%d] = %f \n", i, j, D[i][j], i, j, C[i][j]);
			exit(1);
		}
	}
}
