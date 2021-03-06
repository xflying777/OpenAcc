//***************************************************
// Test cublasDgemm and cblas_dgemm
//***************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "cblas.h"
#include "cublas_v2.h"


void cpu_dgemm(double *A, double *x, double *b, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, x, N, 0.0, b, N);
}

// Ax = b
void gpu_dgemm(double *A, double *x, double *b, int N)
{
	#pragma acc data present(A, x, b)
	{
		#pragma acc host_data use_device(A, x, b)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, x, N, A, N, &beta, b, N);
			cublasDestroy(h);
		}
	}
}

void Dgemm(double *A, double *x, double *b, int N)
{
	int i, j, k;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			b[N*i+j] = 0.0;
			for (k=0; k<N; k++)	b[N*i+j] += A[N*i+k]*x[N*k+j];
		}
	}
}

void initial(double *A, double *x, int N)
{
	int i;
	for (i=0; i<N*N; i++)
	{
		A[i] = sin(1.0*i);
		x[i] = (1.0*i)/N;
	}
	printf(" A[0:2] = %f %f %f \n", A[0], A[1], A[2]);
	printf(" x[0:2] = %f %f %f \n", x[0], x[1], x[2]);
}

double error(double *x, double *y, int N)
{
	int i;
	double error, temp;

	error = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > error)	error = temp;
	}
	return error;
}

int main()
{
	int N;
	printf(" Input N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);

	double *A, *x, *b_cpu, *b_gpu;
	double *b_check;
	double t1, t2, cpu_time, gpu_time;

	A = (double *) malloc(N*N*sizeof(double));
	x = (double *) malloc(N*N*sizeof(double));
	b_cpu = (double *) malloc(N*N*sizeof(double));
	b_gpu = (double *) malloc(N*N*sizeof(double));
	b_check = (double *) malloc(N*N*sizeof(double));

	initial(A, x, N);

	t1 = clock();
	cpu_dgemm(A, x, b_cpu, N);
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin(A[0:N*N], x[0:N*N]) copyout(b_gpu[0:N*N])
	gpu_dgemm(A, x, b_gpu, N);
	t2 = clock();
	gpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	Dgemm(A, x, b_check, N);
	printf(" b_check[0:2] = %f %f %f \n", b_check[0], b_check[1], b_check[2]);
	printf(" cpu error = %f \n", error(b_cpu, b_check, N*N));
	printf(" gpu error = %f \n", error(b_gpu, b_check, N*N));
	printf(" blas error = %f \n", error(b_cpu, b_gpu, N*N));
	printf(" cpu times = %f \n", cpu_time);
	printf(" gpu times = %f \n", gpu_time);
	printf(" cpu times / gpu times = %f \n", cpu_time/gpu_time);

	return 0;
}
