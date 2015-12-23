/*
Using cufft to do the discrete sine transform and solve the Poisson equation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include "openacc.h"

int main()
{
	int i, N, L; 
	double *x, *u, *b, *data2, *data3;
	clock_t t1, t2;
	
	N = pow(2, 10) - 1;
	// Create memory for solving Ax = b, where r = b-Ax is the residue.
	// M is the total number of unknowns.
	L = 2*N + 2 ;

	// Prepare for two dimensional unknown F
	// where b is the one dimensional vector and 
	// F[i][j] = F[j+i*(N-1)];
	b = (double *) malloc(N*N*sizeof(double));
	x = (double *) malloc(N*N*sizeof(double));
	
	//data2 : prepare for dst.
	//data3 : prepare for complex value to data2 and do the cufft. 
	data2 = (double *) malloc(L*N*sizeof(double));
	data3 = (double *) malloc(2*L*N*sizeof(double));

	// Prepare for two dimensional unknowns U
	// where u is the one dimensional vector and
	// U[i][j] = u[j+i*(N-1)] 
	u = (double *) malloc(L2*sizeof(double));
		
	Exact_Solution(u, N);
	Exact_Source(b, N);
	
	t1 = clock();
	Fast_Poisson_Solver(F, X, L);
	t2 = clock();
	
	printf(" Fast Poisson Solver: %f secs\n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	printf(" For N = %d error = %e \n", N, Error(X, U, L));
	printf(" \n");
	}
	return 0;
}

void Exact_Solution(double *u, int N)
{
	// put the exact solution 
	int i, j;
	double x, y, h;
	h = 1.0/(N+1);
	for(i=0;i<N;++i)
	{
		x = (i + 1)*h;
		for(j=0;j<N;++j)
		{
			//k = j + i*(N-1);
			y = (j + 1)*h;
			u[N*i+j] = sin(M_PI*x)*sin(2*M_PI*y);
		}
	}
}

void Exact_Source(double *b, int N)
{
	int i,j;
	double x, y, h;
	h = 1.0/(N+1);
	for(i=0;i<N;++i)
	{
		x = (i+1)*h;
		for(j=0;j<N;++j)
		{
			//k = j + i*(N-1);
			y = (j+1)*h;
			b[N*i+j] = -(1.0+4.0)*h*h*M_PI*M_PI*sin(M_PI*x)*sin(2*M_PI*y);
		}
	}	
}

double Error(double *x, double *u, int N)
{
	// return max_i |x[i] - u[i]|
	int i, j;
	double v = 0.0, e;
	
	for(i=0;i<N;++i)
	{
		for(j=0;j<N;j++)
		{
			e = fabs(x[N*i+j] - u[N*i+j]);
			if(e > v) v = e;		
		}
	}
	return v;
}

void expand_data(float *data, float *data2, int N)
{
	// expand data to 2N + 2 length and ready for dst.
	int i, j;
	#pragma acc loop independent
	for (i=0;i<N;i++)
	{
		data2[N*i] = data2[N*i+N+1] = 0.0;
		for(j=0;i<N;i++)
		{
			data2[N*i+j+1] = data[N*i+j];
			data2[N*i+N+2+j] = -1.0*data[N*i+N-1-j];
		}
	}
}

void expand_idata(float *data2, float *data3, int N, int L)
{
	int i, j;
	#pragma acc loop independent
	for (i=0;i<N;i++)
	{
		for (j=0;i<L;i++)
		{
			data3[2*L*i+2*j] = data2[L*i+j];
			data3[2*L*i+2*j+1] = 0.0;
		}
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

void fdst_gpu(float *data, float *data2, float *data3, int N, int L, int L2)
{
	int i;
	#pragma acc kernels copyin(data[0:N]), create(data2[0:L]), copy(data3[0:2*L])
	{
	expand_data(data, data2, N);
	expand_idata(data2, data3, N, L, L2);
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
