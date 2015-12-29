/*
Using cufft to do the discrete sine transform and solve the Poisson equation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"

void fast_poisson_solver_gpu(double *b, double *x, double *data2, double *data3, int Nx, int Ny, int Lx);
void Exact_Solution(double *u, int Nx);
void Exact_Source(double *b, int Nx);
double Error(double *x, double *u, int Nx);
void print_matrix(double *x, int N);

int main()
{
	int p, Nx, Ny, Lx; 
	double *x, *u, *b, *data2, *data3;
	clock_t t1, t2;
	
	// Initialize the numbers of discrete points.
	// Here we consider Nx = Ny. 
	printf(" Please input (Nx = 2^p - 1) p = ");
	scanf("%d",&p);
	Nx = pow(2, p) - 1;
	Ny = Nx;
	printf(" Nx = %d \n", Nx);
	// Prepare the expanded length for discrete sine transform.
	Lx = 2*Nx + 2 ;
	// Create memory for solving Ax = b, where r = b-Ax is the residue.
	// M is the total number of unknowns.
	// Prepare for two dimensional unknown F
	// where b is the one dimensional vector and 
	// F[i][j] = F[j+i*(N-1)];
	b = (double *) malloc(Nx*Ny*sizeof(double));
	x = (double *) malloc(Nx*Ny*sizeof(double));
	
	// data2 : Prepare for dst.
	// data3 : Prepare for complex value to data2 and do the cufft. 
	data2 = (double *) malloc(Lx*Ny*sizeof(double));
	data3 = (double *) malloc(2*Lx*Ny*sizeof(double));

	// Prepare for two dimensional unknowns U
	// where u is the one dimensional vector and
	// U[i][j] = u[j+i*(N-1)] 
	u = (double *) malloc(Nx*Ny*sizeof(double));
		
	Exact_Solution(u, Nx);
	Exact_Source(b, Nx);
	
	t1 = clock();
	fast_poisson_solver_gpu(b, x, data2, data3, Nx, Ny, Lx);
	t2 = clock();
	
	printf(" Fast Poisson Solver: %f secs\n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	printf(" For N = %d error = %e \n", Nx, Error(x, u, Nx));
	
/*	printf(" \n \n");
	printf(" u matrix \n");
	print_matrix(u, Nx);
	printf(" \n \n");
	printf(" x matrix \n");
	print_matrix(x, Nx);
*/	return 0;
}

void print_matrix(double *x, int N)
{
	int i, j;
	for(i=0;i<N;i++)
	{
		for (j=0;j<N;j++) printf(" %f ", x[N*i+j]);
		printf("\n");
	}
}

void Exact_Solution(double *u, int Nx)
{
	// put the exact solution 
	int i, j;
	double x, y, h;
	h = 1.0/(Nx+1);
	for(i=0;i<Nx;++i)
	{
		x = (i + 1)*h;
		for(j=0;j<Nx;++j)
		{
			//k = j + i*(N-1);
			y = (j + 1)*h;
			u[Nx*i+j] = sin(M_PI*x)*sin(2*M_PI*y);
		}
	}
}

void Exact_Source(double *b, int Nx)
{
	int i,j;
	double x, y, h;
	h = 1.0/(Nx+1);
	for(i=0;i<Nx;++i)
	{
		x = (i+1)*h;
		for(j=0;j<Nx;++j)
		{
			//k = j + i*(N-1);
			y = (j+1)*h;
			b[Nx*i+j] = -(1.0+4.0)*h*h*M_PI*M_PI*sin(M_PI*x)*sin(2*M_PI*y);
		}
	}	
}

double Error(double *x, double *u, int Nx)
{
	// return max_i |x[i] - u[i]|
	int i, j;
	double v, e;
	v = 0.0;
	
	for(i=0;i<Nx;++i)
	{
		for(j=0;j<Nx;j++)
		{
			e = fabs(x[Nx*i+j] - u[Nx*i+j]);
			if(e > v) v = e;		
		}
	}
	return v;
}

void expand_data(double *data, double *data2, int Nx, int Ny, int Lx)
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

void expand_idata(double *data2, double *data3, int Nx, int Ny, int Lx)
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

extern "C" void cuda_fft(double *d_data, int Lx, int Ny, void *stream)
{
	cufftHandle plan;
	cufftPlan1d(&plan, Lx, CUFFT_C2C, Ny);
	cufftSetStream(plan, (cudaStream_t)stream);
	cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data,CUFFT_FORWARD);
	cufftDestroy(plan);
}

void fdst_gpu(double *data, double *data2, double *data3, int Nx, int Ny, int Lx)
{
	double s;
	s = sqrt(2.0/(Nx+1));

	expand_data(data, data2, Nx, Ny, Lx);
	expand_idata(data2, data3, Nx, Ny, Lx);

	#pragma acc data copy(data3[0:2*Lx*Ny])
	// Copy data to device at start of region and back to host and end of region
	// Inside this region the device data pointer will be used
	#pragma acc host_data use_device(data3)
	{
		void *stream = acc_get_cuda_stream(acc_async_sync);
		cuda_fft(data3, Lx, Ny, stream);
	}
	
	#pragma acc data copyin(data3[0:2*Lx*Ny]), copyout(data[0:Nx*Ny])
	#pragma acc parallel loop independent
	for (int i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for (int j=0;j<Nx;j++)	data[Nx*i+j] = -1.0*s*data3[2*Lx*i+2*j+3]/2;
	}

}

void transpose(double *data_in, double *data_out, int Nx, int Ny)
{
	int i, j;
	#pragma acc data copyin(data_in[0:Nx*Ny]), copyout(data_out[0:Ny*Nx])
	#pragma acc parallel loop independent
	for(i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for(j=0;j<Nx;j++)
		{
			data_out[i+j*Ny] = data_in[i*Nx+j];
		}
	}
}

void fast_poisson_solver_gpu(double *b, double *x, double *data2, double *data3, int Nx, int Ny, int Lx)
{
	int i, j;
	double h, *lamda, *temp;
	
	temp = (double *) malloc(Nx*Ny*sizeof(double));
	lamda = (double *) malloc(Nx*sizeof(double));
	h = 1.0/(Nx+1);
	
	#pragma acc data copyout(lamda[0:Nx])
	#pragma acc parallel loop independent
	for(i=0;i<Nx;i++)
	{
		lamda[i] = 2 - 2*cos((i+1)*M_PI*h);
	}
	
	fdst_gpu(b, data2, data3, Nx, Ny, Lx);
	transpose(b, temp, Nx, Ny);
	fdst_gpu(temp, data2, data3, Nx, Ny, Lx);
	transpose(temp, b, Ny, Nx);
	
	#pragma acc data copyin(b[0:Nx*Ny]), copyout(x[0:Nx*Ny])
	#pragma acc parallel loop independent
	for(i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for(j=0;j<Nx;j++) 
		{
			x[Nx*i+j] = -b[Nx*i+j]/(lamda[i] + lamda[j]);
		}
	}

	fdst_gpu(x, data2, data3, Nx, Ny, Lx);
	transpose(x, temp, Nx, Ny);
	fdst_gpu(temp, data2, data3, Nx, Ny, Lx);
	transpose(temp, x, Ny, Nx);
}

