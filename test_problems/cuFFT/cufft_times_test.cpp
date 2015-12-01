#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

extern "C" void forward_cuFFT(float *d_data, int N, void *stream);
void FFTr2(float *x_r, float *x_i, float *y_r, float *y_i, int N);
void Initial(float *x, float *y, int N);
void Print_Complex_Vector(float *y_r, float *y_i, float *data, int N);
int Generate_N(int p);
float Error(float *y_r, float *y_i, float *data, int N);

int main()
{
	int i, p, N;
	float *y_r, *y_i, *x_r, *x_i, *data;
	float cpu_FFT_times, gpu_cuFFT_times, error;
	clock_t t1, t2;
	
	printf(" Please input p =");
	scanf("%d", &p);
	N = Generate_N(p);
	printf(" N = 2^%d = %d \n", p, N);
	
	x_r = (float *) malloc(N*sizeof(float));
	x_i = (float *) malloc(N*sizeof(float));
	y_r = (float *) malloc(N*sizeof(float));
	y_i = (float *) malloc(N*sizeof(float));
	data = (float *) malloc(2*N*sizeof(float));
	
	
	// Initialize interleaved input data on host
	for(i=0; i<2*N; i+=2)  
	{
        	data[i] = i/2.0;
        	data[i+1] = 0.0;
    	}
	Initial(x_r, x_i, N);
	
	// Test cpu_FFT and gpu_cuFFT
	// Copy data to device at start of region and back to host and end of region
	t1 = clock();
    	#pragma acc data copy(data[0:2*N])
    	{
        // Inside this region the device data pointer will be used
        #pragma acc host_data use_device(data)
        	{
           		void *stream = acc_get_cuda_stream(acc_async_sync);
           		forward_cuFFT(data, N, stream);
        	}
    	}
    	t2 = clock();
    	gpu_cuFFT_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
    
	t1 = clock();
	FFTr2(x_r, x_i, y_r, y_i, N);
	t2 = clock();
	cpu_FFT_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	// End test
	
	error = Error(y_r, y_i, data, N);
	
	printf(" y_r[0] = %f, data[0] = %f \n", y_r[0], data[0]);
	printf(" cpu FFT: %f secs \n", cpu_FFT_times);
	printf(" gpu cuFFT: %f secs \n", gpu_cuFFT_times);
	printf(" cpu FFT / gpu cuFFT: %f times \n", cpu_FFT_times / gpu_cuFFT_times);
	printf(" error = %f \n", error);
	printf(" \n");
//	Print_Complex_Vector(y_r, y_i, data, N);
	
	return 0;
} 

float Error(float *y_r, float *y_i, float *data, int N)
{
	int i;
	float error, temp;
	
	error = 0.0;
	for (i = 0; i < N; i++)
	{
		temp = fabs(data[2*i] - y_r[i]);
		if (temp > error) 
		{
			error = temp;
			printf(" error at i = %d , y_r = %f, data = %f \n", i, y_r[i], data[2*i]);
		}
//		temp = fabs(data[2*i+1] - y_i[i]);
//		if (temp > error) error = temp; 
	}
	
	return error;
}
void Initial(float *x, float *y, int N)
{
	int i;
	for(i=0;i<N;++i)
	{
		x[i] = i/1.0;
		y[i] = 0.0;
	}
}

void Print_Complex_Vector(float *y_r, float *y_i, float *data, int N)
{
	int i;
	for(i=0; i<N; ++i)
	{
		if (y_i[i] >=0) printf(" cpu_fft%d : %f + %f i\n", i, y_r[i], y_i[i]);
		else printf(" cpu_fft[%d]: %f  %f i \n", i, y_r[i], y_i[i]);
	}
	printf(" \n");
	for(i=0; i<2*N; i+=2) 
	{
		printf(" cufft_data[%d] = %f + %f i \n", i/2, data[i], data[i+1]);
    	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	return N;
}

void FFTr2(float *x_r, float *x_i, float *y_r, float *y_i, int N)
{
	// input : x = x_r + i * x_i
	// output: y = y_r + i * y_i
	int k,n;
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}
	int i, j, M;
	float t_r, t_i;
	i = j = 0;
	while(i < N)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
//			t_i = y_i[i];
			y_r[i] = y_r[j];
//			y_i[i] = y_i[j];
			y_r[j] = t_r;
//			y_i[j] = t_i;
		}
		M = N/2;
		while(j >= M & M > 0)
		{
			j = j - M;
			M = M / 2;
		}
		j = j + M;		
		i = i + 1;
	}
	// Butterfly structure
	float theta, w_r, w_i;
	n = 2;
	while(n <= N)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<N;i+=n)
			{
				j = i + n/2;
				t_r = w_r * y_r[j] - w_i * y_i[j];
				t_i = w_r * y_i[j] + w_i * y_r[j];
				

				y_r[j] = y_r[i] - t_r;
				y_i[j] = y_i[i] - t_i;
				y_r[i] = y_r[i] + t_r;
				y_i[i] = y_i[i] + t_i;

			}
		}
		n = n * 2;
	}
}

// Forward declaration of wrapper function that will call CUFFT
// Declared extern "C" to disable C++ name mangling
extern "C" void forward_cuFFT(float *d_data, int N, void *stream)
{
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftSetStream(plan, (cudaStream_t)stream);
    cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_FORWARD);
    cufftDestroy(plan);
}


