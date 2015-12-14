#include <cufft.h>

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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openacc.h"

int main()
{
	int i, N;
	N = 16;
	float *data = (float* )malloc(2*N*sizeof(float));


	#pragma acc data create(data[0:2*N])
	#pragma acc kernels
	// Initialize interleaved input data on host
	for(i=0; i<N; i++)  
	{
		data[2*i] = i/2.0;
		data[2*i+1] = 0.0;
	}

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

	for(i=0; i<2*N; i+=2) 
	{
		printf(" cufft_data[%d] = %f + %f i \n", i/2, data[i],data[i+1]);
	}
	return 0;
}
