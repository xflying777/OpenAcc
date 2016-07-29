#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"
#include "cublas_v2.h"

void initial(double *x, double *y_cpu, double *y_gpu, int N);
double error(double *x, double *y, int N);
void axpy_cpu(double *x, double *y, double alpha, int N);
void axpy_gpu(double *x, double *y, double alpha, int N);

//****************************************************************************

int main()
{
	int N, N2;
	printf(" \n Input matrix size N x N, N = ");
	scanf("%d", &N);
	N2 = N*N;

	double *x, *y_cpu, *y_gpu, alpha;

	x = (double *) malloc(N2*sizeof(double));
	y_cpu = (double *) malloc(N2*sizeof(double));
	y_gpu = (double *) malloc(N2*sizeof(double));

	double t1, t2, cpu_times, gpu_times;

	alpha = 1.0;
	initial(x, y_cpu, y_gpu, N);

	t1 = clock();
	axpy_cpu(x, y_cpu, alpha, N2);
	t2 = clock();
	cpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

//	t1 = clock();
	#pragma acc data copyin(x[0:N2]) copy(y_gpu[0:N2])
	{
		t1 =clock();
		axpy_gpu(x, y_gpu, alpha, N2);
		t2 = clock();
	}
//	t2 = clock();
	gpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf("\n");
	printf(" error = %e \n", error(y_cpu, y_gpu, N2));
	printf(" cpu times = %f sec \n gpu times = %f sec \n", cpu_times, gpu_times);
	printf(" cpu times / gpu times = %f \n\n", cpu_times/gpu_times);

	return 0;
}

//****************************************************************************

void initial(double *x, double *y_cpu, double *y_gpu, int N)
{
	int i, j;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			x[i*N+j] = sin(i+j);
			y_cpu[i*N+j] = y_gpu[i*N+j] = cos(i-j);
		}
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

//******************************************************************************

// y = alpha * x + y
void axpy_cpu(double *x, double *y, double alpha, int N)
{
	cblas_daxpy(N, alpha, x, 1, y, 1);
}

void axpy_gpu(double *x, double *y, double alpha, int N)
{
	#pragma acc data present(x, y)
	{
		#pragma acc host_data use_device(x, y)
		{
			cublasHandle_t handle;
			cublasCreate(&handle);

			cublasDaxpy(handle, N, &alpha, x, 1, y, 1);
//			printf(" gpu axpy success \n");

			cublasDestroy(handle);
		}
	}
}

