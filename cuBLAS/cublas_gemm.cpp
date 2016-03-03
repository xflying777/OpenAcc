#include "cublas_v2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/*
	C = alpha * A * B + beta * C
*/


void gpu_cublas3(const int n, const double *a, const double *b, double *c)
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
	#pragma acc data present(a, b, c)
	{
		#pragma acc host_data use_device(a, b, c)
		{
			cublasHandle_t handle;
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n,n,n, &alpha, a, n, b, n, &beta, c, n);
			cublasDestroy(handle);
		}
	}
}

void gpu_oacc(int n, double *a, double *b, double *c)
{
	int i, j, k;
	// Compute mAtrix multipliCAtion.
	#pragma acc parallel loop independent presnet(a, b, c)
	for (i = 0; i < n; ++i) 
	{
		#pragma acc loop independent
		for (j = 0; j < n; ++j) 
		{
			#pragma acc loop seq
			for (k = 0; k < n; ++k) 
			{
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
}

int main()
{
	int i, j, n;
	double gpu_cublas3_times, gpu_oacc_times;
	clock_t t1, t2;

	printf("Input size n = ");
	scanf("%d",&n);

	double *a = (double*)malloc(sizeof(double)*n*n);
	double *b = (double*)malloc(sizeof(double)*n*n);
	double *c_gpu_cublas3 = (double*)malloc(sizeof(double)*n*n);
	double *c_gpu_oacc = (double*)malloc(sizeof(double)*n*n);

	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			a[i*n+j] = i;
			b[i*n+j] = j;
		}
	}
	t1 = clock();
	#pragma acc data copyin(a[0:n*n], b[0:n*n]) create(c_gpu_oacc[0:n*n])
	{
		gpu_oacc( n, a, b, c_gpu_oacc);
	}
	#pragma acc update host(c_gpu_oacc)
	t2 = clock();
	gpu_oacc_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin(a[0:n*n], b[0:n*n]) copyout(c_gpu_cublas3[0:n*n])
	{
		gpu_cublas3(n, a, b, c_gpu_cublas3);
	}
	t2 = clock();
	gpu_cublas3_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	printf(" matrix a = \n");
	for (i=0; i<n; i++)
	{
		for (j=0; j<n; j++)	printf(" %f ", a[i*n+j]);
		printf("\n");
	}

	int nfailures = 0;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j) 
		{
	    	if (c_gpu_oacc[i*n+j] != c_gpu_cublas3[i*n+j]) nfailures++;
		}
	}
	if (nfailures != 0)
      	printf(" Test FAILED\n");
	else
	{
      	printf(" Test tblas6 PASSED\n");
		printf(" gpu oacc times = %f \n", gpu_oacc_times);
		printf(" gpu cublas3 times = %f \n", gpu_cublas3_times);
		printf(" gpu oacc times/gpu cublas3 times = %f \n", gpu_oacc_times/gpu_cublas3_times);
	}
	return nfailures;
}
