//****************************************
// Test: Matrix-vector multiplication
//	1. dgemv by cublas
//	2. dgemv by cblas
//****************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"
#include "cublas_v2.h"

void initial(double *a, double *b, double *c1, double *c2, int N);
double error(double *x, double *y, int N);
void gemv_gpu(double *a, double *b, double *c1, int N);
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

	#pragma acc data copyout(a[0:N*N], b[0:N], c1[0:N], c2[0:N])
	initial(a, b, c1, c2, N);

	t1 = clock();
	#pragma acc data copyin(a[0:N*N], b[0:N]) copy(c1[0:N2])
	{
//		t1 = clock();
		gemv_gpu(a, b, c1, N);
//		t2 = clock();
	}
	t2 = clock();
	time1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	gemv_cblas(a, b, c2, N);
	t2 = clock();
	time2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf(" gemv_cublas spends %f seconds. \n", time1);
	printf(" gemv_cblas spends %f seconds. \n", time2);
	printf(" gemv_cblas / gemv_gpu times = %f . \n", time2/time1);
	printf(" error = %e \n", error(c1, c2, N));
	printf(" \n");

	return 0;
}

//******************************************************************

void initial(double *a, double *b, double *c1, double *c2, int N)
{
	#pragma acc data present(a, b, c1, c2)
	{
		int i;
		#pragma acc parallel loop independent
		for (i=0; i<N*N; i++)
		{
			a[i] = sin(i);
		}

		#pragma acc parallel loop independent
		for( i=0; i<N; i++)
		{
			b[i] = cos(i);
			c1[i] = c2[i] = 0.0;
		}
	} // end pragma data
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

void gemv_gpu(double *a, double *b, double *c1, int N)
{
	#pragma acc data present(a, b, c1)
	{
		#pragma acc host_data use_device(a, b, c1)
		{
			cublasHandle_t handle;
			cublasCreate(&handle);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemv(handle, CUBLAS_OP_T, N, N, &alpha, a, N, b, 1, &beta, c1, 1);
			cublasDestroy(handle);
		}
	} // end pragma data
}

void gemv_cblas(double *a, double *b, double *c2, int N)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, a, N, b, 1, 0.0, c2, 1);
}
