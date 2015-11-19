#include "cublas_v2.h"
#include <time.h>


int gpu_cublas3( const int n, const double *a, const double *b, double *c )
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
	#pragma acc data pcopyin( a[0:n2], b[0:n2] ) pcopyout( c[0:n2] )
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
void gpu_oacc(int n, double *a, double *b, double *c)
//void gpuTest(double **A, double **B, double *restriCt C, int n)
{
	int i,j,k;
	// Compute mAtrix multipliCAtion.
	#pragma acc data copyin(A[0:n][0:n],B[0:n][0:n]) copy(C[0:n][0:n])
	#pragma acc kernels
	#pragma acc loop independent
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


void cpu_gemm( int n, const double *a, const double *b, double *c_cpu)
{
	int i, j, k;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			for(k = 0; k < n; k++ )
			{
				c_cpu[i*n+j] += a[i*n+k]*b[j+n*k];
			}
}


int main()
{
	int i, j, size;
	double gpu_cublas3_times, gpu_oacc_times, cpu_times;
	clock_t t1, t2;

	printf("Input size n = ");
    scanf("%d",&n);
	int n2 = n*n;
	
	double *a = (double*)malloc(sizeof(double)*n2);
	double *b = (double*)malloc(sizeof(double)*n2);
	double *c_gpu_cublas3 = (double*)malloc(sizeof(double)*n2);
	double *c_gpu_oacc = (double*)malloc(sizeof(double)*n2);
	double *c_cpu = (double*)malloc(sizeof(double)*n2);
	
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			a[i*n+j] = 1.0;
			b[i*n+j] = 1.0/n;
		}
	}
	
	t1 = clock();
	cpu_gemm( n, a, b, c_cpu);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	gpu_oacc( n, a, b, c_gpu_oacc);
	t2 = clock();
	gpu_oacc_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	#pragma acc data copyin( a[0:n2], b[0:n2] ) copyout( c[0:n2] )
	{
		gpu_cublas( n, a, b, c);
	}
	t2 = clock();
	gpu_cublas3_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;

    int nfailures = 0;
    printf("%lf %lf\n", c[0], c[n2-1]);
    for (int i = 0; i < n; ++i) 
	{
		for (int j = 0; j < n; ++j) 
		{
	    	if (c_cpu[i*n+j] != c_gpu_cublas3[i*n+j]) nfailures++;
		}
    }
    for (int i = 0; i < n; ++i) 
	{
		for (int j = 0; j < n; ++j) 
		{
	    	if (c_cpu[i*n+j] != c_gpu_oacc[i*n+j]) nfailures++;
		}
    }
    
    if (nfailures != 0)
      	printf(" Test FAILED\n");
    else
	{
      	printf(" Test tblas6 PASSED\n");
		printf(" cpu times = %f \n", cpu_times);
		printf(" gpu oacc times = %f \n", gpu_oacc_times);
		
		printf(" cpu times/gpu oacc times = %f \n", cpu_times/gpu_times);
	}
	
	return error;
}
