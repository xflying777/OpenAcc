#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double Matrix_Vector(double **A, double *x, double *b, int N);
int print_vector(double *x, int N);
int print_matrix(double **A, int N);

int main()
{
	int i, j, N, M;
	double **A, *x, *b, t1, t2, *times, t;
	
	N = 20000;
	M = 10;
	
	x = (double*) malloc(N*sizeof(double));
	b = (double*) malloc(N*sizeof(double));
	A = (double**) malloc(N*sizeof(double*));
	A[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) A[i] = A[i-1] + N;
	
	for(i=0;i<N;i++) x[i] = i;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) A[i][j] = i + j;
	}
	
	times = (double*) malloc(M*sizeof(double));
	for(i=0;i<M;i++)
	{
		t1 = clock();
		Matrix_Vector(A, x, b, N);
		t2 = clock();
		times[i] = 1.0 * (t1 - t2) /CLOCKS_PER_SEC;
	}
	t = 0.0;
	for(i=0;i<M;i++) t = t + times[i];
	t = t / M;
/*	
	print_vector(x, N);
	print_matrix(A, N);
	print_vector(b, N);
*/	
	printf("everage times = %f secs", t);
}

double Matrix_Vector(double **A, double *x, double *b, int N)
{
	#pragma acc parallel loop
	int i, j;

	for(i=0;i<N;i++)
	{
		b[i] = 0.0;
		for(j=0;j<N;j++) b[i] = b[i] + A[i][j] * x[j];

	}
}

int print_vector(double *x, int N)
{
	int i;
	for(i=0;i<N;i++) printf("x[%d] = %f \n", i, x[i]);
}

int print_matrix(double **A, int N)
{
	int i, j;
	for(i=0;i<N;i++)
	{
		printf("A[%d][:] = ", i);
		for(j=0;j<N;j++) printf("%f ",A[i][j]);
		printf("\n");
	}
}
