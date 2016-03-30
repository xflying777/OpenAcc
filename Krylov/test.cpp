//*****************************************************************************
//	Test:
//	Krylov subspace Kn = {b, Ab, A^2b, ..., A^nb}
//
//	Test the computing times of cpu and gpu.
//	Then check the Kn between cpu and gpu.
//*****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"
#include "cblas.h"

void initial_A(double *A, int N);
void initial_b(double *b, int N);
double error(double *x, double *y, int N);
void dgemm_gpu(double *A, double *x, double *b, int N);
void dgemm_cpu(double *A, double *x, double *b, int N);
void cpu_test(double *A, double *b, double *K, int N, int iter);
void gpu_test(double *A, double *b, double *K, int N, int iter);

//******************************************************************************

int main()
{
	int N, iter;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *A, *K_cpu, *K_gpu, *b, *v_gpu, *v_cpu;
	double t1, t2, cpu_time, gpu_time;

	A = (double *) malloc(N*N*sizeof(double));
	K_cpu = (double *) malloc(N*N*(iter+1)*sizeof(double));
	K_gpu = (double *) malloc(N*N*(iter+1)*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));

	v_gpu = (double *) malloc(N*N*sizeof(double));
	v_cpu = (double *) malloc(N*N*sizeof(double));

	initial_A(A, N);
	initial_b(b, N);

	dgemm_cpu(A, b, v_cpu, N);
	#pragma acc data copyin(A[0:N*N], b[0:N*N]) copyout(v_gpu[0:N*N])
	{
		dgemm_gpu(A, b, v_gpu, N);
	}
	printf(" Dgemm max error = %e \n", error(v_cpu, v_gpu, N*N));

	t1 = clock();
	cpu_test(A, b, K_cpu, N, iter);
	t2 = clock();
	cpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
//	printf(" K_cpu[0:2] = %f %f %f , K_cpu[N*N-1] = %f \n", K_cpu[0], K_cpu[1], K_cpu[2], K_cpu[N*N-1]);

	t1 = clock();
	gpu_test(A, b, K_gpu, N, iter);
	t2 = clock();
	gpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
//	printf(" K_gpu[0:2] = %f %f %f , K_gpu[N*N-1] = %f \n", K_gpu[0], K_gpu[1], K_gpu[2], K_gpu[N*N-1]);

	printf(" cpu times = %f, gpu times = %f \n", cpu_time, gpu_time);
	printf(" Error between cpu and gpu, max error = %e \n", error(K_cpu, K_gpu, N*N*(iter+1)));

	return 0;

}

//******************************************************************************

void initial_A(double *A, int N)
{
	int i;
	double h, h2;
	h = 1.0/(N+1);
	h2 = h*h;

	for (i=0; i<N*N; i++)	A[i] = 0.0;
	for (i=0; i<N; i++)	A[N*i+i] = -2.0;
	for (i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = 1.0;
		A[N*i+(i+1)] = 1.0;
	}
//	printf(" A[0:2] = %f %f %f \n", A[0], A[1], A[2]);
}

void initial_b(double *b, int N)
{
	int i;
	for (i=0; i<N*N; i++)	b[i] = sin(M_PI*i);
//	printf(" b[0:2] = %f %f %f \n", b[0], b[1], b[2]);
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
//		if (temp != 0.0)
//		{
//			printf(" Break at %d index ! \n", i);
//			break;
//		}
	}
	return error;
}

//******************************************************************************

void dgemm_gpu(double *A, double *x, double *b, int N)
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

void dgemm_cpu(double *A, double *x, double *b, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, x, N, 0.0, b, N);
}

//******************************************************************************

void cpu_test(double *A, double *b, double *K, int N, int iter)
{
	int i, j;
	double *q, *v;

	q = (double *) malloc(N*N*sizeof(double));
	v = (double *) malloc(N*N*sizeof(double));

	for (i=0; i<N*N; i++)	K[i] = b[i];

	for (i=0; i<iter; i++)
	{
		for (j=0; j<N*N; j++)	q[j] = K[N*N*i+j];
		dgemm_cpu(A, q, v, N);
		for (j=0; j<N*N; j++)	K[N*N*(i+1)+j] = v[j];
	}
}

void gpu_test(double *A, double *b, double *K, int N, int iter)
{
	int i, j;
	double *q, *v;

	q = (double *) malloc(N*N*sizeof(double));
	v = (double *) malloc(N*N*sizeof(double));

	#pragma acc data copyin(A[0:N*N], b[0:N*N]) copy(K[0:N*N*(iter+1)]) create(q[0:N*N], v[0:N*N])
	{
		#pragma acc parallel loop independent
		for (i=0; i<N*N; i++)	K[i] = b[i];

		for (i=0; i<iter; i++)
		{
			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	q[j] = K[N*N*i+j];
			dgemm_gpu(A, q, v, N);
			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	K[N*N*(i+1)+j] = v[j];
		}
	}
}

