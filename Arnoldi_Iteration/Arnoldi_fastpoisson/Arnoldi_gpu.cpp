//*******************************************************
// Only parallel matrix mutplication.
//*******************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"

void initial(double *A, double *b, int N);
void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter);

//**********************************************************************************

int main()
{
	int N, p, iter;
	printf("\n Input N = 2^p - 1, p = ");
	scanf("%d", &p);
	N = pow(2, p) - 1;
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *A, *Q, *H, *b;
	double t1, t2;

	A = (double *) malloc(N*N*sizeof(double));
	Q = (double *) malloc(N*N*(iter+1)*sizeof(double));
	H = (double *) malloc((iter+1)*iter*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));

	initial(A, b, N);

	t1 = clock();
	Arnoldi_Iteration(A, Q, H, b, N, iter);
	t2 = clock();

	printf(" Times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	return 0;
}

//***********************************************************************************

void initial(double *A, double *b, int N)
{
	int i;
	for (i=0; i<N*N; i++)
	{
		A[i] = sin(i);
		b[i] = cos(i);
	}
}

//***********************************************************************************

void norm(double *x, double *result, int N)
{
	#pragma acc data present(x)
	{
		#pragma acc host_data use_device(x)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDdot(h, N, x, 1, result);
			cublasDestroy(h);
		}
	}
}

// Ax = b
void gemm(double *A, double *x, double *b, int N)
{
	#pragma acc data copyin(A[0:N*N], x[0:N*N]) copyout(b[0:N*N])
	{
		#pragma acc host_data use_device(A, x, b)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, x, N, A, N, &beta, b, N);
			cublasDestroy(h);
		}
	}
//	printf(" cublasDgemm success \n");
}

void dot(double *x, double *y, double *result, int N)
{
	#pragma acc data present(x, y)
	{
		#pragma acc host_data use_device(x, y)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDdot(h, N, x, 1, y, 1, result);
			cublasDestroy(h);
		}
	}
}

//****************************************************************************

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
}

//***********************************************************************************

void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, temp;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(b[0:N*N]) create(q[0:N*N], v[0:N*N]) copy(Q[0:N*N*(iter+1)], H[0:(iter+1)*iter])
	{
		fastpoisson(b, q, N);
		norm(b, nrm, N*N);
		temp = *nrm;
		#pragma acc parallel loop independent
		for (k=0; k<N*N; k++)	Q[k] = q[k]/temp;

		for (i=0; i<iter; i++)
		{
			// v= A*qi
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
			gemm(A, q, v, N);
			fastpoisson(v, v, N);

			// h(j,i) = qj*v
			for (j=0; j<=i; j++)
			{
				#pragma acc parallel loop independent
				for (k=0; k<N*N; k++)	q[k] = Q[N*N*j+k];
				dot(q, v, nrm, N);
				H[iter*j+i] = *nrm;
			}

			// v = v - \sum h(j,i)*qj
			#pragma acc parallel loop seq
			for (j=0; j<=i; j++)
			{
				#pragma acc loop independent
				for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
			}

			// h(i+1,i) = ||v||
			norm(v, nrm, N*N);
			H[iter*(i+1)+i] = *nrm;
			temp = *nrm;

			// qi+1 = v/h(i+1,i)
			#pragma acc parallel loop indepnednet
			for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / temp;
		}
	} // end pragma acc
}

