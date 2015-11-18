#include "cublas_v2.h"
#include <time.h>


int matmul( const int n, const double* const a, const double* const b, double* const c )
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

void cpu_gemm( int n, const double* const a, const double* const b, double* const expct)
{
	int i, j, k;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			for(k = 0; k < n; k++ )
			{
				expct[i*n+j] += a[j*n+k]*b[j+n*k];
			}
}


int main()
{
	const int n = 2000;
	double a[n*n];
	double b[n*n];
	double c[n*n];
	double expct[n*n];
	double cpu_times, gpu_times;
	int error = 0;
	clock_t t1, t2;
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
//			expct[i*n+j] = 1.0;
			a[i*n+j] = 1.0;
			b[i*n+j] = 1.0/n;
		}
	}
	
	t1 = clock();
	cpu_gemm( n, a, b, expct);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin( a[0:n*n], b[0:n*n] ) copyout( c[0:n*n] )
	{
//		error = !matmul( n, a, b, c );
		matmul( n, a, b, c);
	}
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;

//        if (error) {
//          printf(" Test FAILED\n");
//        } else {
	    int nfailures = 0;
	    printf("%lf %lf\n", c[0], c[n*n-1]);
	    for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
		    if (expct[i*n+j] != c[i*n+j]) nfailures++;
		}
	    }
	    if (nfailures)
          	printf(" Test FAILED\n");
	    else
		{
          	printf(" Test tblas6 PASSED\n");
		printf(" cpu times = %f \n", cpu_times);
		printf(" gpu times = %f \n", gpu_times);
		printf(" cpu times/gpu times = %f \n", cpu_times/gpu_times);
		}
//  	}
	return error;
}
