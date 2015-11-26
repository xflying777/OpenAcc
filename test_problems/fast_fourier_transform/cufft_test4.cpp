#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

// Declared extern "C" to disable C++ name mangling
extern "C" void launchCUFFT(float *d_data, int n, void *stream)
{
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    cufftSetStream(plan, (cudaStream_t)stream);
    cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
//    cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_INVERSE);
    cufftDestroy(plan);
}


// Forward declaration of wrapper function that will call CUFFT
//extern void launchCUFFT(float *d_data, int n, void *stream);

int main(int argc, char *argv[])
{
    int n = 8;
    float *data =(float* ) malloc(n*sizeof(float));
    int i;

    // Initialize interleaved input data on host
    for(i=0; i<n; i++) 
    {
		data[i] = i;
    }

    // Copy data to device at start of region and back to host and end of region
    #pragma acc data copy(data[0:n])
    {
        // Inside this region the device data pointer will be used
        #pragma acc host_data use_device(data)
        {
//           void *stream = acc_get_cuda_stream(acc_async_sync);
//           launchCUFFT(data, n, stream);
	    launchCUFFT(data, n, 0);
        }
    }

    for(i=0; i<n; i++)
    {
		printf("data[%d] = %f \n", i, data[i]);
    }

    return 0;
}
