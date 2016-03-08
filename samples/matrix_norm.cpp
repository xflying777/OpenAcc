#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"

void initial(double *A, double *B, int N);
void cublas_gemm(const double *A, const double *B, double *C, int N);
double norm_cpu(double *A, int N);
double norm_gpu(double *A, int N);
double cublas_gemm_norm(const double *A, const double *B, double *C, int N);

int main()
{
	int N, N2;
	printf(" \n Input matrix size N x N, N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);
	N2 = N*N;

	double *A, *B, *C_cpu, *C_gpu1, *C_gpu2, t1, t2, cpu_time, gpu1_time, gpu2_time;
	double norm1, norm2, norm3;

	A = (double *) malloc(N2*sizeof(double));
	B = (double *) malloc(N2*sizeof(double));
	C_cpu = (double *) malloc(N2*sizeof(double));
	C_gpu1 = (double *) malloc(N2*sizeof(double));
        C_gpu2 = (double *) malloc(N2*sizeof(double));

	initial(A, B, N);

	t1 = clock();
	#pragma acc data copyin(A[0:N2], B[0:N2]) copyout(C_gpu1[0:N2])
	{
		cublas_gemm(A, B, C_gpu1, N);
		norm1 = norm_gpu(C_gpu1, N);
	}
	t2 = clock();
	gpu1_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

        t1 = clock();
        #pragma acc data copyin(A[0:N2], B[0:N2]) create(C_gpu2[0:N2])
        {
                norm2 = cublas_gemm_norm(A, B, C_gpu2, N);
        }
        t2 = clock();
        gpu2_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin(A[0:N2], B[0:N2]) copyout(C_cpu[0:N2])
	{
		cublas_gemm(A, B, C_cpu, N);
	}
	norm3 = norm_cpu(C_cpu, N);
	t2 = clock();
	cpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	printf("\n norm1 = %f , norm2 = %f , norm3 = %f \n", norm1, norm2, norm3);
	printf(" gpu1 times = %f, gpu2 time = %f, cpu times = %f \n", gpu1_time, gpu2_time, cpu_time);

	return 0;
}

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

double cublas_gemm_norm(const double *A, const double *B, double *C, int N)
{
	double *norm;
	norm = (double *) malloc(1*sizeof(double));

        #pragma acc data present(A, B, C) copyout(norm[0])
        {
                #pragma acc host_data use_device(A, B, C)
                {
                        cublasHandle_t h;
                        cublasCreate(&h);
                        const double alpha = 1.0;
                        const double beta = 0.0;
                        cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, B, N, &beta, C, N);
						cublasDnrm2(h, N*N, C, 1, norm);
                        cublasDestroy(h);
                }
        }
	return *norm;
}

double norm_cpu(double *A, int N)
{
	int i;
	double norm;

	norm = 0.0;
	for (i=0; i<N*N; i++)	norm += A[i]*A[i];

	norm = sqrt(norm);

	return norm;
}

double norm_gpu(double *A, int N)
{
        int i;
        double norm;

	#pragma acc data present(A)
	{
		norm = 0.0;
		#pragma acc parallel loop reduction(+:norm)
        	for (i=0; i<N*N; i++)   norm += A[i]*A[i];
	}
        norm = sqrt(norm);

        return norm;
}

