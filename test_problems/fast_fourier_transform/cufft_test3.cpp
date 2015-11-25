#include <stdio.h> 
#include <math.h> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <cufft.h> 
#include <sys/time.h>

#define NX 1200 
#define NY 1200 

int main(int argc, char *argv[]) 
{ 
cufftHandle plan; 
cufftComplex *devPtr; 
cufftComplex data[NX*NY]; 
int i; 
struct timeval t1,t2,t3,t4; 
long c_fft,c_all; 

/* source data creation */ 
for(i= 0 ; i < NX*NY ; i++){ 
data[i].x = 1.0f; 
data[i].y = 1.0f; 
} 

gettimeofday(&t1,NULL); 

/* GPU memory allocation */ 
cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*NY); 

/* transfer to GPU memory */ 
cudaMemcpy(devPtr, data, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice); 

gettimeofday(&t2,NULL); 

/* creates 2D FFT plan */ 
cufftPlan2d(&plan, NX, NY, CUFFT_C2C); 

/* executes FFT processes */ 
cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD); 

/* executes FFT processes (inverse transformation) */ 
cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE); 

gettimeofday(&t3,NULL); 
c_fft=(t3.tv_sec-t2.tv_sec)*1000000+(t3.tv_usec-t2.tv_usec); 

/* transfer results from GPU memory */ 
cudaMemcpy(data, devPtr, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost); 

/* deletes CUFFT plan */ 
cufftDestroy(plan); 

/* frees GPU memory */ 
cudaFree(devPtr); 

gettimeofday(&t4,NULL); 
c_all=(t4.tv_sec-t1.tv_sec)*1000000+(t4.tv_usec-t1.tv_usec); 

printf("%13ld microseconds for fft\n",c_fft); 
printf("%13ld microseconds for all\n",c_all); 

/* for(i = 0 ; i < NX*NY ; i++){ 
printf("data[%d] %f %f\n", i, data[i].x, data[i].y); 
} 
*/ 

return 0; 
} 
