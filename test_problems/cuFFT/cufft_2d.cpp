/*
Test : Using cufft to do the matrix fourier transform
*/

#include <cufft.h>
// Forward declaration of wrapper function that will call CUFFT
// Declared extern "C" to disable C++ name mangling
extern "C" void for_CUFFT(float *d_data, int nx, int ny, void *stream)
{
	cufftHandle plan;
//	cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
	cufftPlan1d(&plan, nx, CUFFT_C2C, ny);
	cufftSetStream(plan, (cudaStream_t)stream);
	cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
	cufftDestroy(plan);
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openacc.h"

int main(int argc, char *argv[])
{
	int nx, ny, i, j;
	nx = 16;
	ny = 3;
	float *data = (float* )malloc(2*nx*ny*sizeof(float));
    	
	// Initialize interleaved input data on host
	#pragma acc data copy(data[0:2*nx*ny])
	#pragma acc parallel loop independent
	for(i=0; i<ny; i++)  
	{
		#pragma acc loop independent
		for(j=0; j<nx; j++)
		{
			data[2*nx*i + 2*j] = 1.0*j;
			data[2*nx*i + 2*j + 1] = 0.0;
		}
	}

	// Copy data to device at start of region and back to host and end of region
	#pragma acc data copy(data[0:2*nx*ny])
	{
		// Inside this region the device data pointer will be used
		#pragma acc host_data use_device(data)
		{
			void *stream = acc_get_cuda_stream(acc_async_sync);
			for_CUFFT(data, nx, ny, stream);
		}
	}

	for(i=0; i<ny; i++) 
	{
		printf("\n");
		for (j=0; j<nx; j++)	printf(" cufft_data[%d][%d] = %f + %f i \n", i, j, data[2*nx*i+2*j],data[2*nx*i+2*j+1]);
	}
	return 0;
}
