//******************************************************************
// Test: Matrix multiplication for matrix size N x N.
// 	1. dgemm
//	2. for i=1:N	dgemv
//******************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

void initial(double *a, double *b, double *c1, double *c2, int N2);
double error(double *x, double *y, int N);
void dgemm(double *a, double *b, double *c1, int N);
void dgemv(double *a, double *b, double *c2, int N);

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
	b = (double *) malloc(N2*sizeof(double));
	c1 = (double *) malloc(N2*sizeof(double));
	c2 = (double *) malloc(N2*sizeof(double));

	initial(a, b, c1, c2, N2);

	t1 = clock();
	dgemm(a, b, c1, N);
	t2 = clock();
	time1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	dgemv(a, b, c2, N);
	t2 = clock();
	time2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf(" dgemm spends %f seconds. \n", time1);
	printf(" dgemv spends %f seconds. \n", time2);
	printf(" dgemv times / dgemm times = %f . \n", time2/time1);
	printf(" error = %e \n", error(c1, c2, N2));
	printf(" \n");

	return 0;
}

//*******************************************************************

void initial(double *a, double *b, double *c1, double *c2, int N2)
{
	int i;
	for( i=0; i<N2; i++)
	{
		a[i] = sin(i);
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

void dgemm(double *a, double *b, double *c1, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, a, N, b, N, 0.0, c1, N);
}

void dgemv(double *a, double *b, double *c2, int N)
{
	int i, j;
	double *tempb, *tempc;
	tempb = (double *) malloc(N*sizeof(double));
	tempc = (double *) malloc(N*sizeof(double));

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			tempb[j] = b[N*j+i];
			tempc[j] = 0.0;
		}

		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, a, N, tempb, 1, 0.0, tempc, 1);

		for (j=0; j<N; j++)
		{
			c2[N*j+i] = tempc[j];
		}
	}
}
