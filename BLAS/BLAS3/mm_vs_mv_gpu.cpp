//*********************************************************
//	Test: Matrix multiplication between dgemm and dgemv
//		1.dgemm
//		2.dgemv
//	Matrix size: N x N
//*********************************************************

#include "cublas_v2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

	#pragma acc data copyout(a[0:N2], b[0:N2], c1[0:N2], c2[0:N2])
	initial(a, b, c1, c2, N2);

	#pragma acc data copyin(a[0:N2], b[0:N2]) copyout(c1[0:N2])
	{
		t1 = clock();
		dgemm(a, b, c1, N);
		t2 = clock();
	} // end pragma data
	time1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	#pragma acc data copyin(a[0:N2], b[0:N2]) copyout(c2[0:N2])
	{
		t1 = clock();
		dgemv(a, b, c2, N);
		t2 = clock();
	} // end pragma data
	time2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf(" dgemm spends %f seconds. (gpu) \n", time1);
	printf(" dgemv spends %f seconds. (gpu) \n", time2);
	printf(" dgemv times / dgemm times = %f . (gpu) \n", time2/time1);
	printf(" error = %e \n", error(c1, c2, N2));
	printf(" \n");

	return 0;
}

//*******************************************************************

void initial(double *a, double *b, double *c1, double *c2, int N2)
{
	int i;
	#pragma acc data present(a, b, c1, c2)
	{
		#pragma acc parallel loop independent
		for( i=0; i<N2; i++)
		{
			a[i] = sin(i);
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

void dgemm(double *a, double *b, double *c1, int N)
{
	#pragma acc data present(a, b, c1)
	{
		#pragma acc host_data use_device(a, b, c1)
		{
			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &1.0, a, N, b, N, &0.0, c1, N);
			cublasDestroy(handle);
		}
	} // end pragma data
}

void dgemv(double *a, double *b, double *c2, int N)
{
	int i, j;
	double *tempb, *tempc;
	tempb = (double *) malloc(N*sizeof(double));
	tempc = (double *) malloc(N*sizeof(double));

	#pragma acc data present(a, b, c2) create(tempb[0:N], tempc[0:N])
	{
		cublasHandle_t handle;
		cublasCreate(&handle);
		for (i=0; i<N; i++)
		{
			#pragma acc parallel loop independent
			for (j=0; j<N; j++)
			{
				tempb[j] = b[N*j+i];
				tempc[j] = 0.0;
			}
			#pragma acc host_data use_device(a, b, c2)
			{
				cublasDgemv(handle, CUBLAS_OP_N, N, N, &1.0, a, N, tempb, 1, &0.0, tempc, 1);
			} // end pragma host_data
			#pragma acc parallel loop independent
			for (j=0; j<N; j++)
			{
				c2[N*j+i] = tempc[j];
			}
		}
		cublasDestroy(handle);
	} // end pragma data
}

