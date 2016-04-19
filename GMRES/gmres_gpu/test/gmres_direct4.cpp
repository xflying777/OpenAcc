//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the 
// Generalized Minimum Residual method. (A = M + D)
//
// GMRES follows the algorithm described on p. 20 of the 
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//  
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//  
//*****************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "openacc.h"
#include "cublas_v2.h"

void initial_x(double *x, int N);
void initial_A(double *A, int N);
void initial_D(double *D, int N);
void source(double *b, int N);
void exact_solution(double *u, int N);
void gmres(double *A, double *D, double *x, double *b, int N, int max_iter, double tol);
double error(double *x, double *y, int N);

int main()
{
	int p, q, r, N, max_restart, max_iter;
	clock_t t1, t2;
	printf("\n");
	printf(" Input N = 2^p * 3^q * 5^r - 1, (p, q, r) =  ");
	scanf("%d %d %d", &p, &q, &r);
	N = pow(2, p) * pow(3, q) * pow(5, r) - 1;
	printf(" Please input max restart times max_restart = ");
	scanf("%d",&max_restart);
	printf(" Please input max iteration times max_iter = ");
	scanf("%d",&max_iter);
	printf("\n N = %d , max_restart = %d , max_iter = %d \n \n", N, max_restart, max_iter);

	double *A, *D, *x, *b, *u, tol;
	A = (double *) malloc(N*N*sizeof(double));
	D = (double *) malloc(N*N*sizeof(double));
	x = (double *) malloc(N*N*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));
	u = (double *) malloc(N*N*sizeof(double));

	initial_x(x, N);
	initial_A(A, N);
	initial_D(D, N);
	source(b, N);

	tol = 1.0e-6;
	t1 = clock();
	gmres(A, D, x, b, N, max_iter, tol);
	t2 = clock();
	exact_solution(u, N);

	printf(" error = %e \n", error(x, u, N*N));
	printf(" times = %f \n \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}

//****************************************************************************

double error(double *x, double *y, int N)
{
	int i;
	double temp, error;
	error = 0.0;
	for(i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if(temp > error) error = temp;
	}
	return error;
}

void initial_x(double *x, int N)
{
	int i;

	for (i=0; i<N*N; i++)	x[i] = 0.0;
}

void initial_A(double *A, int N)
{
	int i;
	double h, h2, temp;

	h = 1.0/(N+1);
	h2 = h*h;

	for(i=0; i<N*N; i++)	A[i] = 0.0;

	temp = -2.0/h2;
	for(i=0; i<N; i++)
	{
		A[N*i+i] = temp;
	}
	temp = 1.0/h2;
	for(i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = temp;
		A[N*i+(i+1)] = temp;
	}
}

void initial_D(double *D, int N)
{
	int i;
	double h, temp;
	h = 1.0/(N+1);

	for (i=0; i<N*N; i++)	D[i] = 0.0;

	temp = 1.0/(2*h);
	for (i=0; i<N-1; i++)
	{
		D[N*(i+1)+i] = -1.0*temp;
		D[N*i+(i+1)] = temp;
	}
}

void exact_solution(double *u, int N)
{
	int i, j;
	double h, x, y;

	h = 1.0/(N+1);
	for (i=0; i<N; i++)
	{
		y = (i+1)*h;
		for (j=0; j<N; j++)
		{
			x = (j+1)*h;
			u[N*i+j] = x*y*sin(M_PI*x)*sin(M_PI*y);
		}
	}
}

void source(double *b, int N)
{
	int i, j;
	double h, x, y;

	h = 1.0/(N+1);
	for (i=0; i<N; i++)
	{
		y = (i+1)*h;
		for (j=0; j<N; j++)
		{
			x = (j+1)*h;
			b[N*i+j] = (x*sin(M_PI*x)*(2*M_PI*cos(M_PI*y) - M_PI*M_PI*y*sin(M_PI*y)) + y*sin(M_PI*y)*(2*M_PI*cos(M_PI*x) - M_PI*M_PI*x*sin(M_PI*x))) + (x*sin(M_PI*x)*(sin(M_PI*y) + M_PI*y*cos(M_PI*y)));
		}
	}
}

//****************************************************************************

void norm_gpu(double *x, double *norm, int N)
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

void dot_gpu(double *x, double *y, double *result, int N)
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

void q_subQ_gpu(double *q, double *Q, int N, int iter)
{
	int i;
	#pragma acc parallel loop independent present(q, Q)
	for (i=0; i<N; i++)	q[i] = Q[N*iter+i];
}

void subQ_v_gpu(double *Q, double *v, int N, int iter, double norm_v)
{
	int i, N2;
	N2 = N*iter;
	#pragma acc parallel loop independent present(Q, v)
	for (i=0; i<N; i++)	Q[N2+i] = v[i]/norm_v;
}

//***********************************************************************************

void GeneratePlaneRotation(double dx, double dy, double *cs, double *sn, int i)
{
	#pragma acc kernels present(cs, sn)
	{
		double temp;
		if (dy == 0.0) 
		{
			cs[i] = 1.0;
			sn[i] = 0.0;
		} 
		else if (fabs(dy) > fabs(dx)) 
		{
			temp = dx / dy;
			sn[i] = 1.0 / sqrt( 1.0 + temp*temp );
			cs[i] = temp * sn[i];
		} 
		else 
		{
			temp = dy / dx;
			cs[i] = 1.0 / sqrt( 1.0 + temp*temp );
			sn[i] = temp * cs[i];
		}
	}
}

//****************************************************************************

// Note : cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, A, n, B, n, &beta, C, n)
// means matrix C = B * A
void cublas_gemm(int n, double *c, double *b, double *a )
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

//****************************************************************************


void gmres(double *A, double *D, double *x, double *b, int N, int max_iter, double tol)
{
	int i, j, k, l, N2;
	double resid, *normb, *beta, *nrm_temp, temp, *M_temp, *r, *q, *v, *M_b, *w, *cs, *sn, *s, *y, *Q, *H;

	normb = (double *) malloc(1*sizeof(double));
	beta = (double *) malloc(1*sizeof(double));
	nrm_temp = (double *) malloc(1*sizeof(double));

	Q = (double *) malloc(N*N*(max_iter+1)*sizeof(double));
	H = (double *) malloc((N+1)*max_iter*sizeof(double));
	M_temp = (double *) malloc(N*N*sizeof(double));
	M_b = (double *) malloc(N*N*sizeof(double));
	r = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));
	v = (double *) malloc(N*N*sizeof(double));
	w = (double *) malloc(N*N*sizeof(double));
	cs = (double *) malloc((max_iter+1)*sizeof(double));
	sn = (double *) malloc((max_iter+1)*sizeof(double));
	s = (double *) malloc((max_iter+1)*sizeof(double));
	y = (double *) malloc((max_iter+1)*sizeof(double));

	N2 = N*N;

	#pragma acc data copyin(b[0:N2]) copyout(M_b[0:N2], r[0:N2])
	{
		fastpoisson(b, M_b, N);
		norm_gpu(M_b, normb, N2);
		#pragma acc parallel loop independent
		for (k=0; k<N2; k++)	r[k] = M_b[k];
//		norm_gpu(r, beta, N2);
	}

	*beta = *normb;

	if ((resid = *beta / *normb) <= tol)
	{
		tol = resid;
		max_iter = 0;
  	}


	for (i=0; i<N2; i++)	Q[i] = r[i] / *beta;
	for (i=0; i<max_iter; i++)	s[i+1] = 0.0;
	s[0] = *beta;

	for (i=0; i<max_iter; i++)
	{
		#pragma acc data copyin(Q[0:N2*(max_iter+1)], D[0:N2]) copyout(q[0:N2], v[0:N2])
		{
	  		q_subQ_gpu(q, Q, N2, i);
	  		cublas_gemm(N, v, D, q);
		} // end pragma acc
		#pragma acc data copyin(v[0:N2]) copyout(M_temp[0:N2])
		{
			fastpoisson(v, M_temp, N);
		} // end pragma acc

//		printf(" First step pass. \n");

		#pragma acc data copy(Q[0:N2*(max_iter+1)], H[0:(N+1)*max_iter], x[0:N2]) copyin(q[0:N2], M_temp[0:N2]) create(w[0:N2], cs[0:max_iter+1], sn[0:max_iter+1], s[0:max_iter+1], y[0:max_iter+1])
		{
			#pragma acc parallel loop independent present(w, q, M_temp)
	  		for (k=0; k<N*N; k++)	w[k] = q[k] + M_temp[k];

  			// h(k,i) = qk*w
	  		for (k=0; k<=i; k++)
			{
				#pragma acc parallel loop independent
				for (j=0; j<N2; j++)
				{
					q[j] = Q[N2*k+j];
	  			}
				dot_gpu(q, w, nrm_temp, N2);
				H[max_iter*k+i] = *nrm_temp;
			}

			#pragma acc parallel loop seq
			for (k=0; k<=i; k++)
			{
				#pragma acc loop independent
				for (j=0; j<N2; j++)	w[j] -= H[max_iter*k+i]*Q[N2*k+j];
			}

			norm_gpu(w, nrm_temp, N2);
			H[max_iter*(i+1)+i] = *nrm_temp;
			subQ_v_gpu(Q, w, N2, i+1, H[max_iter*(i+1)+i]);

			#pragma acc kernels
			for (k = 0; k < i; k++)
			{
				//ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k))
				temp = cs[k]*H[max_iter*k+i] + sn[k]*H[max_iter*(k+1)+i];
				H[max_iter*(k+1)+i] = -1.0*sn[k]*H[max_iter*k+i] + cs[k]*H[max_iter*(k+1)+i];
				H[max_iter*k+i] = temp;
			}

			GeneratePlaneRotation(H[max_iter*i+i], H[max_iter*(i+1)+i], cs, sn, i);

			#pragma acc kernels
			{
				//ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i))
				H[max_iter*i+i] = cs[i]*H[max_iter*i+i] + sn[i]*H[max_iter*(i+1)+i];
				H[max_iter*(i+1)+i] = 0.0;

				//ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
				temp = cs[i]*s[i];
				s[i+1] = -1.0*sn[i]*s[i];
				s[i] = temp;
				resid = fabs(s[i+1] / *beta);
			} //end kernels

			if (resid < tol)
			{
				#pragma acc host_data use_device(H, y, s)
				{
					cublasHandle_t h;
					cublasCreate(&h);
					cublasDcopy(h, max_iter+1, s, 1, y, 1);
					cublasDtrsv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, i, H, max_iter, y, 1);
					cublasDestroy(h);
				}
//				backsolve(H, s, y, N, max_iter, i);
				#pragma acc parallel loop independent
				for(j=0; j<N; j++)
				{
					#pragma acc loop independent
					for (l=0; l<N; l++)
					{
						#pragma acc loop seq
						for(k=0; k<=i; k++)
						{
							x[N*j+l] += Q[N2*k+N*j+l]*y[k];
						}
					}
				}
				printf(" resid = %e \n", resid);
				printf(" Converges at %d step. \n", i+1);
				break;
			}
		} //end pragma acc
		printf(" Second step pass. \n");
	}//end for
}



