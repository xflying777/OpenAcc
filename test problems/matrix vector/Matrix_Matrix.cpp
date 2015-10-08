#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double Matrix_Vector(double **A, double *x, double *b, int N);
double Matrix_Matrix(double **A, double **B, double **C, int N);

int main()
{
	int i, j, N;
	double **A, **B, **C, t1, t2;
	
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
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) A[i][j] = i + j;
	}		
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) B[i][j] = i - j;
	}
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) C[i][j] = 0.0;
	}
	
	t1 = clock();
	Matrix_Matrix(A, B, C, N);
	t2 = clock();
	
	printf("times = %f secs", 1.0*(t1-t2)/CLOCKS_PER_SEC);
}

double Matrix_Matrix(double **A, double **B, double **C, int N)
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
