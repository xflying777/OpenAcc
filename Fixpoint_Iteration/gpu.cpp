//*************************************************************************
//	Problem :
//		\Delta u + Beta * (\partial u / \partial y ) = f
//
//	Solved with preconditioned fixed-point iteration.
//  Let fast Poisson solver be the preconditioner.
//
//	   Ax = b
//	=> (M + D)x = b
//	=> M^(-1)(M + D)x = M^(-1)b
//	=> x = M^(-1)Dx = M^(-1)b
//	=> x(k+1) = M^(-1)(b - Dx(k))
//
//*************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"
#include "cublas_v2.h"

void initial(double *x, double *b, double *Beta, double *D, double *u, int N);
double error(double *x, double *y, int N);
void fixpoint_iteration(double *Beta, double *D, double *x, double *b, int N, double tol);

int main()
{
	printf("\n");
	int N, p, q, r, s;
	printf(" Input N = 2^p * 3^q * 5^r * 7^s - 1, (p, q, r, s) =  ");
	scanf("%d %d %d %d", &p, &q, &r, &s);
	N = pow(2, p) * pow(3, q) * pow(5, r) * pow(7, s) - 1;
	printf(" N = %d \n\n", N);

	double *x, *b, *Beta, *D, *u;
	double tol, t1, t2;

	x = (double *) malloc(N*N*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));
	Beta = (double *) malloc(N*N*sizeof(double));
	D = (double *) malloc(N*N*sizeof(double));
	u = (double *) malloc(N*N*sizeof(double));


	initial(x, b, Beta, D, u, N);

	tol = 1.0e-6;

	t1 = clock();
	fixpoint_iteration(Beta, D, x, b, N, tol);
	t2 = clock();

	printf(" Spend %f seconds. \n", 1.0*(t2 - t1)/CLOCKS_PER_SEC);
	printf(" Error = %e \n", error(x, u, N*N));

	printf(" \n");
	return 0;
}

//***********************************************************************************************************

double error(double *x, double *y, int N)
{
	int i;
	double e, temp;

	e = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > e)	e = temp;
	}
	return e;
}

void initial(double *x0, double *b, double *Beta, double *D, double *u, int N)
{
	int i, j;
	double *bxxyy, *by;
	double x, y, h, h2, p, temp;

	bxxyy = (double *) malloc(N*N*sizeof(double));
	by = (double *) malloc(N*N*sizeof(double));

	p = M_PI;
	h = 1.0/(N+1);
	h2 = 2.0*h;

	for (i=0; i<N; i++)
	{
		y = (i+1)*h;
		for (j=0; j<N; j++)
		{
			x = (j+1)*h;
			// exact solution
			u[N*i+j] = x*y*sin(2*p*x)*sin(2*p*y);

			// given source
			bxxyy[N*i+j] = -1.0*(8*p*p*x*y*sin(2*p*x)*sin(2*p*y) - 4*p*(y*cos(2*p*x)*sin(2*p*y) + x*sin(2*p*x)*cos(2*p*y)));
			by[N*i+j] = 2*p*x*y*sin(2*p*x)*cos(2*p*y) + x*sin(2*p*x)*sin(2*p*y);
			// Beta = sin(y)
			Beta[N*i+j] =  sin(y);
		}
	}

	for (i=0; i<N*N; i++)
	{
		// initial x
		x0[i] = 0.0;
		// source
		b[i] = bxxyy[i] + Beta[i]*by[i];

		D[i] = 0.0;
	}

	temp = 1.0/h2;
	for (i=0; i<N-1; i++)
	{
		D[N*(i+1)+i] = -1.0*temp;
		D[N*i+(i+1)] = temp;
	}
}

//***********************************************************************************************************

// b = alpha * A * x + beta * b;
// Note : cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, A, n, B, n, &beta, C, n)
// means matrix C = B * A
void dgemm(int n, double *c, double *b, double *a )
{
	#pragma acc data present(a, b, c)
	{
		#pragma acc host_data use_device(a, b, c)
		{
			cublasHandle_t handle;
			cublasCreate(&handle);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n);
			cublasDestroy(handle);
		}
	}
}

void norm(double *x, double *norm, int N)
{
	#pragma acc data present(x)
	{
		#pragma acc host_data use_device(x)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDnrm2(h, N, x, 1, norm);
			cublasDestroy(h);
		}
	}
}

//***********************************************************************************************************


void expand_data(double *data, double *data2, int Nx, int Ny, int Lx) 
{ 
	// expand data to 2N + 2 length 
	#pragma acc parallel loop independent present(data[0:Nx*Ny],data2[0:Lx*Ny]) 
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
	#pragma acc parallel loop independent present(data2[0:Lx*Ny],data3[0:2*Lx*Ny]) 
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
	cufftPlan1d(&plan, Lx, CUFFT_Z2Z, Ny); 
	cufftSetStream(plan, (cudaStream_t)stream); 
	cufftExecZ2Z(plan, (cufftDoubleComplex*)d_data, (cufftDoubleComplex*)d_data,CUFFT_FORWARD); 
	cufftDestroy(plan); 
} 

void fdst_gpu(double *data, double *data2, double *data3, int Nx, int Ny, int Lx) 
{ 
	double s; 
	s = sqrt(2.0/(Nx+1)); 
	#pragma acc data present(data3[0:2*Lx*Ny],data[0:Nx*Ny],data2[0:Lx*Ny]) 
	{ 
		expand_data(data, data2, Nx, Ny, Lx); 
		expand_idata(data2, data3, Nx, Ny, Lx); 

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
			for (int j=0;j<Nx;j++)   data[Nx*i+j] = -1.0*s*data3[2*Lx*i+2*j+3]/2; 
		} 
	}// end data region
} 

void transpose(double *data_in, double *data_out, int Nx, int Ny) 
{ 
	int i, j; 
	#pragma acc parallel loop independent present(data_in[0:Nx*Ny],data_out[0:Ny*Nx]) 
	for(i=0;i<Ny;i++) 
	{ 
		#pragma acc loop independent 
		for(j=0;j<Nx;j++) 
		{ 
			data_out[i+j*Ny] = data_in[i*Nx+j]; 
		} 
	} 
} 

void fastpoisson(double *b, double *x, int N) 
{ 
	int i, j, Nx, Ny, Lx;
	double h, h2, *lamda, *temp, *temp_b, *data2, *data3;

	Nx = Ny = N;
	Lx = 2*Nx + 2;
	data2 = (double *) malloc(Lx*Ny*sizeof(double));
	data3 = (double *) malloc(2*Lx*Ny*sizeof(double));
	temp = (double *) malloc(Nx*Ny*sizeof(double));
	temp_b = (double *) malloc(Nx*Ny*sizeof(double));
	lamda = (double *) malloc(Nx*sizeof(double));

	h = 1.0/(Nx+1);
	h2 = h*h;
	#pragma acc data create(lamda[0:Nx], temp[0:Nx*Ny], temp_b[0:Nx*Ny], data2[0:Lx*Ny], data3[0:2*Lx*Ny]) present(b, x)
	{ 
		#pragma acc parallel loop independent
		for (i=0; i<Nx*Ny; i++)	temp_b[i] = b[i];
		#pragma acc parallel loop independent 
		for(i=0;i<Nx;i++)	lamda[i] = 2 - 2*cos((i+1)*M_PI*h);

		fdst_gpu(temp_b, data2, data3, Nx, Ny, Lx); 
		transpose(temp_b, temp, Nx, Ny); 
		fdst_gpu(temp, data2, data3, Nx, Ny, Lx); 
		transpose(temp, temp_b, Ny, Nx); 

		#pragma acc parallel loop independent 
		for(i=0;i<Ny;i++) 
		{ 
			#pragma acc loop independent 
			for(j=0;j<Nx;j++) 
			{ 
				x[Nx*i+j] = -1.0*h2*temp_b[Nx*i+j]/(lamda[i] + lamda[j]); 
			} 
		}

		fdst_gpu(x, data2, data3, Nx, Ny, Lx); 
		transpose(x, temp, Nx, Ny); 
		fdst_gpu(temp, data2, data3, Nx, Ny, Lx); 
		transpose(temp, x, Ny, Nx); 
	} // end data region
	free(data2);
	free(data3);
	free(temp);
	free(temp_b);
	free(lamda);
}

//***********************************************************************************************************

// x(k+1) = M^(-1) * (b - D * x(k))
void fixpoint_iteration(double *Beta, double *D, double *x, double *b, int N, double tol)
{
	int i, j;

	double *xk, *temp, *error;

	xk = (double *) malloc(N*N*sizeof(double));
	temp = (double *) malloc(N*N*sizeof(double));
	error = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(D[0:N*N], b[0:N*N], Beta[0:N*N]) copy(x[0:N*N]) create(xk[0:N*N], temp[0:N*N])
	{
		for (i=0; i<N*N; i++)
		{
			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	xk[j] = x[j];

			dgemm(N, temp, D, xk);

			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	temp[j] = b[j] - Beta[j]*temp[j];

			fastpoisson(temp, x, N);

			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	temp[j] = x[j] - xk[j];
			norm(temp, error, N*N);

			if ( *error < tol)
			{
				printf(" Converges at %d step ! \n", i+1);
				printf(" residual = %e \n", *error);
				break;
			}
		}
	} // end pragma region
}
