#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

extern "C" void for_CUFFT(float *d_data, int n, void *stream);
void initial(float *data, int N);
void cuda_fft(float *data, int N);
void print_vector(float *data, int N);
void expand_data(float *data, float *data2, int N);
void expand_idata(float *data2, float *data3, int M);

int main()
{
	int p, N, M;
	float *data, *data2, *data3;

	printf(" Please input p such that N = 2^p, p = ");
	scanf("%d",&p);
	N = pow(2, p);
	M = 2*N + 2;

	data = (float* )malloc(N*sizeof(float));
	data2 = (float* )malloc(M*sizeof(float));
	data3 = (float* )malloc(2*M*sizeof(float));

	// Initialize interleaved input data on host
	initial(data, N);
	print_vector(data2, N);
	return 0;
}

void gpu_make_up(float *data, float *data2, float *data3, int N, int M)
{
	expand_data(data, data2, N);
	expand_idata(data2, data3, M);
	cuda_fft(data, M);
	
}

void initial(float *data, int N)
{	
	#pragma acc data create(data[0:2*N])
	#pragma acc kernels
	for(int i=0; i<N; i++)  
	{
		data[i] = i;
	}
}
void print_vector(float *data, int N)
{
	for (int i=0;i<N;i++)
	{
		printf(" data[%d] = %f \n", i, data[i]);
	}
}

void expand_data(float *data, float *data2, int N)
{
	// expand data to 2N + 2 length 
	int i;
	#pragma acc kernels
	{
		data2[0] = data2[N+1] = 0.0;
		#pragma acc loop independent
		for(i=0;i<N;i++)
		{
			data2[i+1] = data[i];
			data2[N+i+2] = -1.0*data[N-1-i];
		}
	}
}

void expand_idata(float *data2, float *data3, int M)
{
	int i;
	#pragma acc parallel loop independent
	for (i=0;i<M;i++)
	{
		data3[2*i] = data2[i];
		data3[2*i+1] = 0.0;
	}
}

void cuda_fft(float *data, int N)
{
	// Copy data to device at start of region and back to host and end of region
	#pragma acc data copyout(data[0:2*N])
	{
		// Inside this region the device data pointer will be used
		#pragma acc host_data use_device(data)
		{
			void *stream = acc_get_cuda_stream(acc_async_sync);
			for_CUFFT(data, N, stream);
		}
	}
}

// Forward declaration of wrapper function that will call CUFFT
// Declared extern "C" to disable C++ name mangling
extern "C" void for_CUFFT(float *d_data, int n, void *stream)
{
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, 1);
	cufftSetStream(plan, (cudaStream_t)stream);
	cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
	cufftDestroy(plan);
}
