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
#include "openacc.h"

int main(int argc, char *argv[])
{
    int n = 8;
    float *data = (float* )malloc(2*n*sizeof(float));
    int i;
    
    // Initialize interleaved input data on host
    for(i=0; i<2*n; i+=2)  
	{
        data[i] = i/2.0;
        data[i+1] = 0.0;
    }

    // Copy data to device at start of region and back to host and end of region
    #pragma acc data copy(data[0:2*n])
    {
        // Inside this region the device data pointer will be used
        #pragma acc host_data use_device(data)
        {
           void *stream = acc_get_cuda_stream(acc_async_sync);
           for_CUFFT(data, n, stream);
        }
    }

    for(i=0; i<2*n; i+=2) {
	printf(" cufft_data[%d] = %f + %f i \n", i/2, data[i],data[i+1]);
    }

    return 0;
}
