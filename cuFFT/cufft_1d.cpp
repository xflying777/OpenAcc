#include <cufft.h>

// Declared extern "C" to disable C++ name mangling
extern "C" void for_CUFFT(double *d_data, int n, void *stream)
{
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
    cufftSetStream(plan, (cudaStream_t)stream);
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_data, (cufftDoubleComplex*)d_data,CUFFT_FORWARD);
    cufftDestroy(plan);
}



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openacc.h"

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

// Forward declaration of wrapper function that will call CUFFT
extern void launchCUFFT(double *d_data, int n, void *stream);

int main(int argc, char *argv[])
{
    int n = 256;
    double *data = (double* )malloc(2*n*sizeof(double));
    int i;

    // Initialize interleaved input data on host
    double w = 7.0;
    double x;
    for(i=0; i<2*n; i+=2)  {
        x = (double)i/2.0/(n-1);
        data[i] = cos(2*M_PI*w*x);
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
//	   inv_CUFFT(data, n, stream);
        }
    }

    // Find the frequency
    int max_id = 0;
    for(i=0; i<n; i+=2) {
        if( data[i] > data[max_id] )
            max_id = i;
    }
    printf("frequency = %d\n", max_id/2);

    return 0;
}
