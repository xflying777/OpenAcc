//****************************************
// Test: Matrix-vector multiplication
//	dgemv
//****************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

void initial(double *a, double *b, double *c1, double *c2, int N);
double error(double *x, double *y, int N);
void gemv(double *a, double *b, double *c1, int N);
void gemv_cblas(double *a, double *b, double *c2, int N);

int main()
{
	printf("\n");
	int N;
	printf(" Input matrix size N x N, N = ");
	scanf("%d", &N);
	printf("\n");

	int N2;
	N2 = N * N;

	double *a, *b, *c1, *c2;
	double t1, t2, time1, time2;

	a = (double *) malloc(N2*sizeof(double));
	b = (double *) malloc(N*sizeof(double));
	c1 = (double *) malloc(N*sizeof(double));
	c2 = (double *) malloc(N*sizeof(double));

	initial(a, b, c1, c2, N);

	t1 = clock();
	gemv(a, b, c1, N);
	t2 = clock();
	time1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	gemv_cblas(a, b, c2, N);
	t2 = clock();
	time2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf(" gemv spends %f seconds. \n", time1);
	printf(" gemv_cblas spends %f seconds. \n", time2);
	printf(" gemv times / gemv_blas times = %f . \n", time1/time2);
	printf(" error = %e \n", error(c1, c2, N));
	printf(" \n");

	return 0;
}

//******************************************************************

void initial(double *a, double *b, double *c1, double *c2, int N)
{
	int i;
	for (i=0; i<N*N; i++)
	{
		a[i] = sin(i);
	}

	for( i=0; i<N; i++)
	{
		b[i] = cos(i);
		c1[i] = c2[i] = 0.0;
	}
}

double error(double *x, double *y, int N)
{
	int i;
	double e, temp;
	e = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > e)	e = temp;
	}

	return e;
}

//*******************************************************************

void gemv(double *a, double *b, double *c1, int N)
{
	int i, j;

	for (i=0; i<N; i++)
	{
		c1[i] = 0.0;
		for (j=0; j<N; j++)
		{
			c1[i] += a[N*i+j] * b[j];
		}
	}
}

void gemv_cblas(double *a, double *b, double *c2, int N)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, a, N, b, 1, 0.0, c2, 1);
}
