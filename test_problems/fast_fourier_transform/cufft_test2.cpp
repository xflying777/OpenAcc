#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX      256
#define BATCH   10

int main(int argc, char *argv[])
{
        cufftHandle plan;
        cufftComplex *devPtr;
        cufftComplex data[NX*BATCH];
        
        int i;
        for(i = 0 ; i < NX*BATCH ; i++){
                data[i].x = 1.0f;
                data[i].y = 1.0f;
        }

        cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*BATCH);
        cudaMemcpy(devPtr, data, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);
        
        cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
        cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD);
        cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE);
        
        cudaMemcpy(data, devPtr, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyDeviceToHost);
        
		cufftDestroy(plan);
        cudaFree(devPtr);

        for(i = 0 ; i < NX*BATCH ; i++){
                printf("data[%d] %f %f\n", i, data[i].x, data[i].y);
        }

        return 0;
}
