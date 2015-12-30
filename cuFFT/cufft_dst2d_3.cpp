/*
 Test : Using cufft to do discrete sine transform about a matrix (Nx*Ny).
 Nx : Numbers of row.
 Ny : Numbers of column.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

void Initial(float *data, int Nx, int Ny);
void print_matrix(float *data, int Nx, int Ny);
//void print_complex_vector(float *data, int N);
void fdst_gpu(float *data, float *data2, float *data3, int Nx, int Ny, int Lx);

int main()
{
	int p, Nx, Ny, Lx;
	float *x, *data2, *data3, t1, t2;
	
	printf(" Please input ( Nx = 2^p - 1 ) p = ");
	scanf("%d",&p);
	Nx = pow(2,p) - 1;
	printf(" Nx = %d \n", Nx);
	printf(" Please input Ny = ");
	scanf("%d",&Ny);
	printf(" Ny = %d \n \n", Ny);
	// Expand the length prepared for discrete sine transform.
	Lx = 2*Nx + 2;

	x = (float *) malloc(Nx*Ny*sizeof(float));
	data2 = (float *) malloc(Lx*Ny*sizeof(float));
	data3 = (float *) malloc(2*Lx*Ny*sizeof(float));
	
	Initial(x, Nx, Ny);
	printf(" Initial data[%d][%d] = %f, data[%d][%d] = %f \n", 0, 0, x[0], Ny, Nx, x[Ny*Nx-1]);
	
	t1 = clock();
	fdst_gpu(x, data2, data3, Nx, Ny, Lx);
	fdst_gpu(x, data2, data3, Nx, Ny, Lx);
	t2 = clock();
	printf(" Double dst data[%d][%d] = %f, data[%d][%d] = %f \n", 0, 0, x[0], Ny, Nx, x[Ny*Nx-1]);
	
	printf(" fdst 2d in gpu: %f secs \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}

void print_matrix(float *data, int Nx, int Ny)
{
	for (int i=0;i<Ny;i++)
	{
		for (int j=0;j<Nx;j++)	printf(" %f ", data[Nx*i+j]);
		printf(" \n");
	}
	printf("\n");
}

void Initial(float *data, int Nx, int Ny)
{	
	#pragma acc data copy(data[0:Nx*Ny])
	#pragma acc parallel loop independent
	for(int i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for (int j=0;j<Nx;j++)	data[Nx*i+j] = 1.0*j;
	}
}

void expand_data(float *data, float *data2, int Nx, int Ny, int Lx)
{
	// expand data to 2N + 2 length
	#pragma acc data copyin(data[0:Nx*Ny]), copyout(data2[0:Lx*Ny]) 
	#pragma acc parallel loop independent
	for(int i=0;i<Ny;i++)
	{
		data2[Lx*i] = data2[Lx*i+Nx+1] = 0.0;
		#pragma acc loop independent
		for(int j=0;j<Nx;j++)
		{
			data2[Lx*i+j+1] = data[Nx*i+j];
			data2[Lx*i+Nx+j+2] = -1.0*data[Nx*i+Nx-1-j];
		}
	}
}

void expand_idata(float *data2, float *data3, int Ny, int Lx)
{
	#pragma acc data copyin(data2[0:Lx*Ny]), copyout(data3[0:2*Lx*Ny])
	#pragma acc parallel loop independent
	for (int i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for (int j=0;j<Lx;j++)
		{
			data3[2*Lx*i+2*j] = data2[Lx*i+j];
			data3[2*Lx*i+2*j+1] = 0.0;
		}
	}
}

extern "C" void cuda_fft(float *d_data, int Lx, int Ny, void *stream)
{
	cufftHandle plan;
	cufftPlan1d(&plan, Lx, CUFFT_C2C, Ny);
	cufftSetStream(plan, (cudaStream_t)stream);
	cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
	cufftDestroy(plan);
}

void fdst_gpu(float *data, float *data2, float *data3, int Nx, int Ny, int Lx)
{
	float s;
	s = sqrt(2.0/(Nx+1));
	#pragma acc data copy(data[0:Nx*Ny], data3[0:2*Lx*Ny]), create(data2[0:Lx*Ny])
	{
		expand_data(data, data2, Nx, Ny, Lx);
		expand_idata(data2, data3, Ny, Lx);

		// Copy data to device at start of region and back to host and end of region
		// Inside this region the device data pointer will be used
		#pragma acc host_data use_device(data3)
		{
			void *stream = acc_get_cuda_stream(acc_async_sync);
			cuda_fft(data3, Lx, Ny, stream);
		}
		
		#pragma acc parallel loop independent
		for (int i=0;i<Ny;i++)
		{
			#pragma acc loop independent
			for (int j=0;j<Nx;j++)	data[Nx*i+j] = -1.0*s*data3[2*Lx*i+2*j+3]/2;
		}
	}
}

