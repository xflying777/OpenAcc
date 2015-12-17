#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

void Initial(float *x_r, float *x_i, float *data, int N);
int Generate_N(int p);
void printf_resault(float *x, float *y, int N);
void fdst_gpu(float *data, float *data2, float *data3, int N, int L);
void fdst_cpu(float *x_r, float *x_i, float *y_r, float *y_i, int N, int L);
void error(float *y_r, float *data, int N);

int main()
{
	int p, N, L;
	float *y_r, *y_i, *x_r, *x_i, *data, *data2, *data3;
	float cpu_times, gpu_times;
	clock_t t1, t2;
	
	printf(" Please input p ( N = 2^p - 1 ) = ");
	scanf("%d",&p);

	N = Generate_N(p);

	printf(" N=2^%d - 1 = %d\n",p,N);
	L = 2*N + 2;
	
	x_r = (float *) malloc(N*sizeof(float));
	x_i = (float *) malloc(N*sizeof(float));
	y_r = (float *) malloc(L*sizeof(float));
	y_i = (float *) malloc(L*sizeof(float));
	data = (float *) malloc(N*sizeof(float));
	data2 = (float *) malloc(L*sizeof(float));
	data3 = (float *) malloc(2*L*sizeof(float));

	Initial(x_r, x_i, data, N);

	t1 = clock();
	fdst_cpu(x_r, x_i, y_r, y_i, N, L);
	t2 = clock();
	cpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	fdst_gpu(data, data2, data3, N, L);
	t2 = clock();
	gpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	
	error(y_r, data, N);
	printf(" fdst CPU: %f secs \n", cpu_times);
	printf(" fdst GPU: %f secs \n", gpu_times);
	printf("CPU / GPU : %f times \n", cpu_times/gpu_times);
	
	printf_resault(y_r, data, N);
	return 0;
}
 
//////////////////
// Initial part //
//////////////////

void printf_resault(float *cpu_fdst, float *gpu_fdst, int N)
{
	int i;
	for (i=0;i<N;i++)
	{
		printf(" cpu_fdst[%d] = %f \n", i, cpu_fdst[i]);
		printf(" gpu_fdst[%d] = %f \n", i, gpu_fdst[i]);
		printf(" \n");
	}
}
void Initial(float *x_r, float *x_i, float *data, int N)
{
	int i;
	for(i=0;i<N;i++)
	{
		x_r[i] = i;
		x_i[i] = 0.0;
		data[i] = i;
	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	N = N - 1;
	return N;
}

void error(float *y_r, float *data, int N)
{
	float error, temp;
	error = 0.0;
	for (int i=0;i<N;i++)
	{
		temp = abs(data[i] - y_r[i]);
		if (temp > error) error = temp;
	}
	printf(" error = %f \n", error);
}

//////////////
// GPU part //
//////////////

// expand the initial data to 2N+2-points for fast fourier discrete sine transformation

void expand_gpu(float *data, float *data2, int N)
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

void expand_idata(float *data2, float *data3, int L)
{
	int i;
	#pragma acc parallel loop independent
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
	#pragma acc data copy(data[0:N]), create(data2[0:L], data3[0:2*L])
	expand_gpu(data, data2, N);
	expand_idata(data2, data3, L);
	
	#pragma acc data copyout(data3[0:2*L])
	// Copy data to device at start of region and back to host and end of region
	// Inside this region the device data pointer will be used
	#pragma acc host_data use_device(data3)
	{
		void *stream = acc_get_cuda_stream(acc_async_sync);
		cuda_fft(data3, L, stream);
	}

	#pragma acc parallel loop independent
	for(i=0;i<N;i++)
	{
		data[i] = -1.0*data3[2*i+3]/2;
	}
}
//////////////
// CPU part //
//////////////

// expand the initial data to 2N+2-points for fast fourier discrete sine transformation 
void expand_cpu(float *x_r, float *x_i, float *y_r, float *y_i, int N)
{
	// expand y[n] to 2N+2-points from x[n]
	// x[n] is the initial data

	int i;
	y_r[0] = y_i[0] = 0.0;
	y_r[N+1] = y_i[N+1] = 0.0;
	for(i=0;i<N;i++)
	{
		y_r[i+1] = x_r[i];
		y_i[i+1] = x_i[i];
		y_r[N+i+2] = -1.0*x_r[N-1-i];
		y_i[N+i+2] = -1.0*x_i[N-1-i];
	}
}

void fdst_cpu(float *x_r, float *x_i, float *y_r, float *y_i, int N, int L)
{
	int i, j, k, n, M;
	float t_r, t_i;

	expand_cpu(x_r, x_i, y_r, y_i, N);
	i = j = 0;
	while(i < L)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
			t_i = y_i[i];
			y_r[i] = y_r[j];
			y_i[i] = y_i[j];
			y_r[j] = t_r;
			y_i[j] = t_i;
		}
		M = L/2;
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
	while(n <= L)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<L;i+=n)
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
	
	// After fft(y[k]), Y[k] = fft(y[k]), Sx[k] = i*Y[k+1]/2
	for(k=0;k<N;k++)
	{
		y_r[k] = -1.0*y_i[k+1]/2;
	}
	
}



