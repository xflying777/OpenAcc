//*******************************************************************************************************************
//	Test cublas level-1 (on gpu), compare with cpu after matrix mutiplication.
//	Step :
//	1. C = A * B
//	2. r = dot(C, B)
//	3. C = - r * B + C
//	4. nrmC = norm(C)
// 	4. D = C;
//	6. D = D / nrmC;
//*******************************************************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"

//************************************************************************
void initial(double *A, double *B, int N);
void error(double *A, double *B, int N);
//************************************************************************
void cublas_gemm(const double *A, const double *B, double *C, int N);
//************************************************************************
void gpu_cublas1(double *A, double *B, double *C, double *D, double *r, double *nrmC, int N, int N2);
//************************************************************************
double norm_cpu(double *A, int N);
double dot_cpu(double *A, double *B, int N);
void axpy_cpu(double alpha, double *A, double *B, int N);
void copy_cpu(double *A, double *B, int N);
void scal_cpu(double alpha, double *A, int N);
//************************************************************************


int main()
{
	int N, N2;
	printf(" \n Input matrix size N x N, N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);
	N2 = N*N;

	double *A, *B, *C_cpu, *C_gpu, *D_cpu, *D_gpu, t1, t2, cpu_time, gpu_time;
	double r_cpu, *r_gpu, nrmC_cpu, *nrmC_gpu;

	A = (double *) malloc(N2*sizeof(double));
	B = (double *) malloc(N2*sizeof(double));
	C_cpu = (double *) malloc(N2*sizeof(double));
	C_gpu = (double *) malloc(N2*sizeof(double));
	D_cpu = (double *) malloc(N2*sizeof(double));
	D_cpu = (double *) malloc(N2*sizeof(double));
	
	r_gpu = (double *) malloc(1*sizeof(double));
	nrmC_gpu = (double *) malloc(1*sizeof(double));

	initial(A, B, N);

	t1 = clock();
	
	#pragma acc data copyin(A[0:N2], B[0:N2]) copyout(C_cpu[0:N2])
	{
		cublas_gemm(A, B, C_cpu, N);
	}
	r_cpu = dot_cpu(C_cpu, B, N2);
	axpy_cpu(-1.0*r_cpu, B, C_cpu, N2);
	nrmC_cpu = norm(C_cpu, N2);
	copy_cpu(C_cpu, D, N2);
	scal_cpu(1.0/nrmC_cpu, D, N2);
	
	t2 = clock();
	cpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	
	#pragma acc enter data copyin(A[0:N2], B[0:N2]) create(C_gpu[0:N2], D_gpu[0:N2], r_gpu[0], nrmC_gpu[0]) 
	{
		gpu_cublas1(A, B, C_gpu, D_gpu, r_gpu, nrmC_gpu, N, N2);
	}
	#pragma acc update host(D_gpu[0:N2])
	
	t2 = clock();
	gpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf(" error = %f \n", error(D_cpu, D_gpu, N2));
	printf(" gpu time = %f, cpu times = %f \n", gpu_time, cpu_time);

	return 0;
}

//************************************************************************

void initial(double *A, double *B, int N)
{
	int i, j;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			A[i*N+j] = sin(i+j);
			B[i*N+j] = cos(i-j);
		}
	}
}

void error(double *A, double *B, int N)
{
	int i;
	double error, temp;
	
	error = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(A[i] - B[i]);
		if (temp > error)	error = temp;
	}
	
	return error;
}
//************************************************************************

void cublas_gemm(const double *A, const double *B, double *C, int N)
{
	#pragma acc data present(A, B, C)
	{
		#pragma acc host_data use_device(A, B, C)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, B, N, &beta, C, N);
			cublasDestroy(h);
		}
	}
}

//************************************************************************

void gpu_cublas1(double *A, double *B, double *C, double *D, double *r, double *nrmC, int N, int N2)
{
	#pragma acc data present(A, B, C, D, r, nrmC)
	{
		#pragma acc host_data use_device(A, B, C, D)
		{
			cublasHandle_t handle;
			cublasCreate(&handle);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, B, N, &beta, C, N);
			cublasDdot(handle, N2, C, 1, B, 1, r);
			*r = -1.0 * *r;
			cublasDaxpy(handle, N2, r, B, 1, C, 1);
			cublasDnrm2(handle, N2, C, 1, nrmC);
			cublasDcopy(hnadle, N2, C, 1, D, 1);
			*nrmC = 1.0 / *nrmC;
			cublasDscal(handle, N2, nrmC, D, 1);
			cublasDestroy(handle);
		}
	}
}

//************************************************************************

//	\sum_{i=1}^{N} A[i] * A[i]
double norm_cpu(double *A, int N)
{
	int i;
	double norm;

	norm = 0.0;
	for (i=0; i<N*N; i++)	norm += A[i]*A[i];

	norm = sqrt(norm);

	return norm;
}

//	\sum_{i=1}^{N} A[i] * B[i] 
double dot_cpu(double *A, double *B, int N)
{
	int i;
	double result;
	
	result = 0.0;
	for (i=0; i<N; i++)	result += A[i]*B[i];
	
	return result;
}

//	B[i] = alpha * A[i] + B[i], for i = 1, ..., N
void axpy_cpu(double alpha, double *A, double *B, int N)
{
	int i;
	
	for (i=0; i<N; i++)	B[i] = alpha*A[i] + B[i];
}

//	B[i] = A[i], for i = 1, ..., N
void copy_cpu(double *A, double *B, int N)
{
	int i;
	
	for (i=0; i<N; i++)	B[i] = A[i];
}

//	A[i] = alpha * A[i], for i = 1, ..., N
void scal_cpu(double alpha, double *A, int N)
{
	int i;
	
	for (i=0; i<N; i++)	A[i] = alpha * A[i];
}

//************************************************************************
