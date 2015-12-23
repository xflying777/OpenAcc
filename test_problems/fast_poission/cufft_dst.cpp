/*
Test : Using cufft to do discrete sine transform and compare with cpu.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

int Generate_N(int p);
void Initial(float *data, int N);
void print_vector(float *data, int N);
void expand_data(float *data, float *data2, int N, int L);
void expand_idata(float *data2, float *data3, int L);
void print_complex_vector(float *data, int N);
extern "C" void cuda_fft(float *d_data, int N, void *stream);
void fdst_gpu(float *data, float *data2, float *data3, int N, int L);

int main()
{
	int p, N, L;
	float *data, *data2, *data3, t1, t2;
	
	printf(" Please input p ( N = 2^p - 1 ) = ");
	scanf("%d",&p);
	N = Generate_N(p);
	L = 2*N + 2;

	data = (float *) malloc(N*sizeof(float));
	data2 = (float *) malloc(L*sizeof(float));
	data3 = (float *) malloc(2*L*sizeof(float));
	
	Generate_N(p);
	Initial(data, N);
	
	t1 = clock();	
	fdst_gpu(data, data2, data3, N, L);
	t2 = clock();

//	print_vector(data, N);
	printf(" data[%d] = %f , data[%d] = %f \n", 0, data[0], N-1, data[N-1]);
	printf(" fdst gpu: %f secs \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
//	print_complex_vector(data3, L);	
	return 0;
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	N = N - 1;
	
	printf(" N = %d \n", N);
	return N;
}

void print_vector(float *data, int N)
{
	for (int i=0;i<N;i++)
	{
		printf(" data[%d] = %f \n", i, data[i]);
	}
}

void print_complex_vector(float *data, int N)
{
        int i;
        for(i=0;i<N;i++)
        {
                if (data[2*i+1] >= 0) printf("%d : %f +%f i\n", i, data[2*i], data[2*i+1]);
                else printf("%d : %f %f i\n", i, data[2*i], data[2*i+1]);
        }

}
void Initial(float *data, int N)
{	
	#pragma acc data copy(data[0:N])
	#pragma acc kernels
	for(int i=0;i<N;i++)
	{
		data[i] = i;
	}
}

void expand_data(float *data, float *data2, int N, int L)
{
	// expand data to 2N + 2 length 
	int i;
//	#pragma acc data copy(data2[0:L])
	data2[0] = data2[N+1] = 0.0;
	#pragma acc loop independent
	for(i=0;i<N;i++)
	{
		data2[i+1] = data[i];
		data2[N+i+2] = -1.0*data[N-1-i];
	}
}

void expand_idata(float *data2, float *data3, int L)
{
	int i;
	#pragma acc loop independent
	for (i=0;i<L;i++)
	{
		data3[2*i] = data2[i];
		data3[2*i+1] = 0.0;
	}
}

extern "C" void cuda_fft(float *d_data, int N, void *stream)
{
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_C2C, 1);
	cufftSetStream(plan, (cudaStream_t)stream);
	cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
	cufftDestroy(plan);
}

void fdst_gpu(float *data, float *data2, float *data3, int N, int L)
{
	int i;
	#pragma acc kernels copyin(data[0:N]), create(data2[0:L]), copy(data3[0:2*L])
	{
	expand_data(data, data2, N, L);
	expand_idata(data2, data3, L);
	}

	#pragma acc data copy(data3[0:2*L])
	// Copy data to device at start of region and back to host and end of region
	// Inside this region the device data pointer will be used
	#pragma acc host_data use_device(data3)
	{
		void *stream = acc_get_cuda_stream(acc_async_sync);
		cuda_fft(data3, L, stream);
	}
	
	#pragma acc data copy(data[0:N]), copyin(data3[0:2*L])
	#pragma acc parallel loop independent
	for(i=0;i<N;i++)
	{
		data[i] = -1.0*data3[2*i+3]/2;
	}

}

