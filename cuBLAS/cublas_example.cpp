#include "cublas_v2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/*
 Note : cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n)
 		means matrix C = B * A
 		
 		In cublasDgemm ,we should set matrix as array.
*/

int gpu_cublas3( const int n, const double *a, const double *b, double *c )
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
	#pragma acc data pcopyin( a[0:n*n], b[0:n*n] ) pcopyout( c[0:n*n] )
	{
		#pragma acc host_data use_device( a, b, c )
		{
			cublasHandle_t handle;
			stat = cublasCreate(&handle);
			if ( CUBLAS_STATUS_SUCCESS != stat ) {
				printf("CUBLAS initialization failed\n");
			}
			
			if ( CUBLAS_STATUS_SUCCESS == stat )
			{
				const double alpha = 1.0;
				const double beta = 0.0;
				stat = cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n);
				if (stat != CUBLAS_STATUS_SUCCESS) {
					printf("cublasDgemm failed\n");
				}
			}
			cublasDestroy(handle);
		}
	}
	return CUBLAS_STATUS_SUCCESS == stat;
}

int main()
{
	int i, j, n;
	double gpu_cublas3_times;
	clock_t t1, t2;

	printf("Input size n = ");
	scanf("%d",&n);

	double *a = (double*)malloc(sizeof(double)*n*n);
	double *b = (double*)malloc(sizeof(double)*n*n);
	double *c_gpu_cublas3 = (double*)malloc(sizeof(double)*n*n);

	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			a[i*n+j] = i + j;
			b[i*n+j] = i - j;
		}
	}

	t1 = clock();
	#pragma acc data copyin( a[0:n*n], b[0:n*n] ) copyout( c_gpu_cublas3[0:n*n] )
	{
		gpu_cublas3( n, a, b, c_gpu_cublas3);
	}
	t2 = clock();
	gpu_cublas3_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	for ( i = 0 ;i < n; ++i) 
		for ( j = 0; j < n; ++j)
			printf(" c[%d][%d] = %f \n", i, j, c_gpu_cublas3[i*n + j]);

	printf("cublas times = %f \n ", gpu_cublas3_times);
	return 0;
}
