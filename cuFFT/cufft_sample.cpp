#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

extern "C" void cuda_fft(double *x, int N, void *stream);
void print_fft(double *x, int N);

int main()
{
	int i, p, q, r, N;
        clock_t t1, t2;
        printf("\n");
        printf(" Input N = 2^p * 3^q * 5^r - 1, (p, q, r) =  ");
        scanf("%d %d %d", &p, &q, &r);
        N = pow(2, p) * pow(3, q) * pow(5, r) - 1;
        printf("\n N = %d \n \n", N);

	double *x;
	x = (double *) malloc(2*N*sizeof(double));

	for (i=0; i<N; i++)
	{
		x[2*i] = sin(1.0*i);
		x[2*i+1] = 0.0;
	}

	t1 = clock();
	#pragma acc data copy(x[0:2*N])
	{
		#pragma acc host_data use_device(x)
		{
			void *stream = acc_get_cuda_stream(acc_async_sync);
			cuda_fft(x, N, stream);
		}
	}
	t2 = clock();

	print_fft(x, N);
	printf(" Spend %f seconds. \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}

// x = fft(x)
extern "C" void cuda_fft(double *x, int N, void *stream)
{
        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
        cufftSetStream(plan, (cudaStream_t)stream);
        cufftExecZ2Z(plan, (cufftDoubleComplex*)x, (cufftDoubleComplex*)x, CUFFT_FORWARD);
        cufftDestroy(plan);
}

void print_fft(double *x, int N)
{
	int i;
	for (i=0; i<N; i++)
	{
		if (x[2*i+1] >= 0)	printf(" x[%d] = %f +%fi \n", i, x[2*i], x[2*i+1]);
		if (x[2*i+1] < 0)	printf(" x[%d] = %f %fi \n", i, x[2*i], x[2*i+1]);
	}
	printf(" \n");
}


