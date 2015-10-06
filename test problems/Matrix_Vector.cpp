#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double Matrix_Vector(double **A, double *x, double *b, int N);

int main()
{
	int i, j, N;
	double **A, *x, *b, t1, t2;
	
	N = 10000;
	x = (double*) malloc(N*sizeof(double));
	b = (double*) malloc(N*sizeof(double));
	A = (double**) malloc(N*sizeof(double*));
	A[0] = (double*) malloc(N*N*sizeof(double));
	for(i=1;i<N;i++) A[i] = A[i-1] + N;
	
	for(i=0;i<N;i++) b[i] = 0.0;
	for(i=0;i<N;i++) x[i] = i;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) A[i][j] = i + j;
	}
	
	t1 = clock();
	Matrix_Vector(A, x, b, N);
	t2 = clock();
	
/*	for(i=0;i<N;i++) printf("x[%d] = %f \n", i, x[i]);
	for(i=0;i<N;i++)
	{
		printf("A[%d][:] = ", i);
		for(j=0;j<N;j++) printf("%f ",A[i][j]);
		printf("\n");
	}
	for(i=0;i<N;i++) printf("b[%d] = %f \n", i, b[i]);
*/	
	printf("times = %f secs", 1.0*(t1-t2)/CLOCKS_PER_SEC);
}

double Matrix_Vector(double **A, double *x, double *b, int N)
{
	#pragma acc parallel loop
	int i, j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) b[i] = b[i] + A[i][j] * x[j];
	}
}
