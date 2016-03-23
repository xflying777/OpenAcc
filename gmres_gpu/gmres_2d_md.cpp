//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the 
// Generalized Minimum Residual method
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

void print_vector(double *x, int N);
void matrix_vector(double *A, double *x, double *b, int N);
void print_matrixH(double *x, int N, int k);
void initial(double *A, double *b, double *x0, int N);
void exact_solution(double *u, int N);
void gmres(double *A, double *x, double *b, int N, int max_restart, int max_iter, double tol);
double error(double *x, double *y, int N);

int main()
{
	int N, max_restart, max_iter;
	clock_t t1, t2;
	printf(" Please input N =  ");
	scanf("%d",&N);
	printf(" Please input max restart times max_restart = ");
	scanf("%d",&max_restart);
	printf(" Please input max iteration times max_iter = ");
	scanf("%d",&max_iter);
	printf(" N = %d , max_restart = %d , max_iter = %d \n \n", N, max_restart, max_iter);
	
	double *A, *x, *b, *u, tol;
	A = (double *) malloc(N*N*sizeof(double));
	x = (double *) malloc(N*N*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));
	u = (double *) malloc(N*N*sizeof(double));
	
	initial(A, b, x, N);	
	tol = 1.0e-4;
	t1 = clock();
	gmres(A, x, b, N, max_restart, max_iter, tol);
	t2 = clock();
	exact_solution(u, N);
	
	printf(" error = %e \n", error(x, u, N*N));
	printf(" times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
	return 0;
}

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

void print_vector(double *x, int N)
{
	int i;
	for(i=0; i<N; i++)
	{
		printf(" %f ", x[i]);
	}
	printf("\n");
}

void print_matrix(double *x, int N)
{
	int i, j;
	for(i=0;i<N;i++)
	{
		for (j=0;j<N;j++) printf(" %f ", x[N*i+j]);
		printf("\n");
	}
	printf("\n");
}

void print_matrixH(double *x, int max_iter, int k)
{
	int i, j;
	for(i=0;i<=k;i++)
	{
		for (j=0;j<=k;j++) printf(" %f ", x[max_iter*i+j]);
		printf("\n");
	}
	printf("\n");
}

void initial(double *A, double *b, double *x0, int N)
{
	int i, j;
	double h, h2, temp, x, y;
	
	h = M_PI/(N+1);
	h2 = h*h;
	
	for(i=0; i<N; i++)
	{
		y = (1+i)*h;
		for(j=0; j<N; j++)
		{
			x = (1+j)*h;
			x0[N*i+j] = 0.0;
			A[N*i+j] = 0.0;
			b[N*i+j] = x*sin(x)*(2*cos(y) - y*sin(y)) + y*sin(y)*(2*cos(x) - x*sin(x));
		}
	}
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

void exact_solution(double *u, int N)
{
	int i, j;
	double h, x, y;

	h = M_PI/(N+1);
	for (i=0; i<N; i++)
	{
		y = (i+1)*h;
		for (j=0; j<N; j++)
		{
			x = (j+1)*h;
			u[N*i+j] = x*y*sin(x)*sin(y);
		}
	}
}

//**************************************************************************************

void norm(double *x, double *nrm, int N)
{
	#pragma acc data present(x)
	{
		#pragma acc host_data use_device(x)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDnrm2(h, N, x, 1, nrm);
			cublasDestroy(h);
		}
	}
}

void matrix_matrix(double *A, double *x, double *b, int N)
{
	#pragma acc data present(A, x, b)
	{
		#pragma acc host_data use_device(A, x, b)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, x, N, &beta, b, N);
			cublasDestroy(h);
		}
	}
}

// TU + UT
void matrix_TUUT(double *T, double *U, double *tu, double *ut, int N)
{
	#pragma acc data present(T, U, tu, ut)
	{
		#pragma acc host_data use_device(T, U, tu, ut)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, T, N, U, N, &beta, tu, N);
			cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, U, N, T, N, &beta, ut, N);
			cublasDaxpy(h, N*N, &alpha, tu, 1, ut, 1);
			cublasDestroy(h);
		}
	}
}

void q_subQ(double *q, double *Q, int N, int iter)
{
	int i;
	#pragma acc parallel loop independent present(q, Q)
	for (i=0; i<N; i++)	q[i] = Q[N*iter+i];
}

void subQ_v(double *Q, double *v, int N, int iter, double norm_v)
{
	int i, N2;
	N2 = N*iter;
	for (i=0; i<N; i++)	Q[N2+i] = v[i]/norm_v;
}

void w_shift(double *v, double *q, double h, int N)
{
	int i;
	for(i=0; i<N; i++)
	{
		v[i] -= h*q[i];
	}
}

void backsolve(double *H, double *s, double *y, int N, int max_iter, int i)
{
	// i = iter
	int j, k;
	double temp;
	
	for(j=i; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=i; k++)
		{
			temp -= y[k]*H[max_iter*j+k];
		}
		y[j] = temp/H[max_iter*j+j];
	}

}

void GeneratePlaneRotation(double dx, double dy, double *cs, double *sn, int i)
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

//**************************************************************************************

void gmres(double *A, double *x, double *b, int N, int max_restart, int max_iter, double tol)
{
	int i, j, k, m, N2;
	double resid, *normb, *beta, *nrm, temp, *r, *q, *v, *w, *cs, *sn, *s, *y, *Q, *H;
	
	N2 = N*N;
	Q = (double *) malloc(N2*(max_iter+1)*sizeof(double));
	H = (double *) malloc((N+1)*max_iter*sizeof(double));
	r = (double *) malloc(N2*sizeof(double));
	q = (double *) malloc(N2*sizeof(double));
	v = (double *) malloc(N2*sizeof(double));
	w = (double *) malloc(N2*sizeof(double));
	cs = (double *) malloc((max_iter+1)*sizeof(double));
	sn = (double *) malloc((max_iter+1)*sizeof(double));
	s = (double *) malloc((max_iter+1)*sizeof(double));
	y = (double *) malloc((max_iter+1)*sizeof(double));
	
	normb = (double *) malloc(1*sizeof(double));
	beta = (double *) malloc(1*sizeof(double));
	nrm = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(b[0:N2]) copyout(r[0:N2])
	{
		#pragma acc host_data use_device(b, r)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDnrm2(h, N2, b, 1, normb);
			cublasDcopy(h, N2, b, 1, r, 1);
			cublasDnrm2(h, N2, r, 1, beta);
			cublasDestroy(h);
		}
	} //end pragma acc
	
	for (m=0; m<max_restart; m++)
	{
		#pragma acc data copyin(r[0:N2]) copyout(Q[0:N2*(max_iter+1)])
		{
			#pragma acc parallel loop independent
			for (i=0; i<N2; i++)	Q[i] = r[i] / *beta;
			#pragma acc parallel loop independent
			for (i=0; i<max_iter; i++)	s[i+1] = 0.0;
			s[0] = beta;
		}
		
		for (i = 0; i<max_iter; i++) 
		{
			#pragma acc data copyin(A[0:N2]) copy(Q[0:N2*(max_iter+1)], H[0:(N+1)*max_iter]) create(q[0:N2], w[0:N2], v[0:N2])
			{
				// w = Aq + qA
				#pragma acc parallel loop independent
				for (k=0; k<N2; k++)	q[k] = Q[N2*i+k];
				matrix_TUUT(A, q, v, w, N);

				// h(j,i) = qj*v
				#pragma acc parallel loop independent
				for (j=0; j<=i; j++)
				{
					H[max_iter*j+i] = 0.0;
					#pragma acc loop seq
					for (k=0; k<N*N; k++)	H[max_iter*j+i] += Q[N2*j+k]*v[k];
				}

				// w = w - \sum h(j,i)*qj
				#pragma acc parallel loop seq
				for (j=0; j<=i; j++)
				{
					#pragma acc loop independent
					for (k=0; k<N2; k++)	w[k] -= H[max_iter*j+i]*Q[N*N*j+k];
				}

				// h(i+1,i) = ||w||
				// qi+1 = w/h(i+1,i)
				norm(w, nrm, N2);
				temp = *nrm;
				H[max_iter*(i+1)+i] = temp;
				#pragma acc parallel loop independent
				for (k=0; k<N2; k++)	Q[N2*(i+1)+k] = w[k]/temp;	
			} //end pragma acc
			
	    		for (k = 0; k < i; k++)
	      		{
	      			//ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k))
	      			temp = cs[k]*H[max_iter*k+i] + sn[k]*H[max_iter*(k+1)+i];
				H[max_iter*(k+1)+i] = -1.0*sn[k]*H[max_iter*k+i] + cs[k]*H[max_iter*(k+1)+i];
				H[max_iter*k+i] = temp;
			}
			
	      		GeneratePlaneRotation(H[max_iter*i+i], H[max_iter*(i+1)+i], cs, sn, i);
	      	
	      		//ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i))
			H[max_iter*i+i] = cs[i]*H[max_iter*i+i] + sn[i]*H[max_iter*(i+1)+i];
			H[max_iter*(i+1)+i] = 0.0;
			
		      	//ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
		      	temp = cs[i]*s[i];
		      	s[i+1] = -1.0*sn[i]*s[i];
		      	s[i] = temp;
		      	resid = fabs(s[i+1]/beta);
		     	
		     	if (resid < tol) 
			{
				printf(" resid = %e \n", resid);
				printf(" Converges at %d step ", i+1);
				backsolve(H, s, y, N, max_iter, i);
				#pragma acc data copyin(Q[0:N2*(max_iter+1)], y[0:max_iter+1]) copy(x[0:N2])
				{
					#pragma acc parallel loop seq
					for (k=0; k<=i; k++)
					{
						#pragma acc loop independent
						for(j=0; j<N2; j++)
						{
							x[j] += Q[N2*k+j]*y[k];
						}
					}
				}
				break;
	      		}
		}//end inside for
		
		if (resid < tol)	
		{
			printf(" %d cycle \n", m);
			break;
		}
		
		// Caution : i = i + 1.
		i = i - 1;
		backsolve(H, s, y, N, max_iter, i);
		#pragma acc data copyin(A[0:N2], Q[0:N2*(max_iter+1)], b[0:N2], y[0:max_iter+1]) copy(x[0:N2], r[0:N2]) copy(x[0:N2]) create(v[0:N2], w[0:N2])
		{
			#pragma acc parallel loop seq
			for (k=0; k<=i; k++)
			{
				#pragma acc loop independent
				for(j=0; j<N2; j++)
				{
					x[j] += Q[N2*k+j]*y[k];
				}
			}

			// v = Ax + xA
			// r = b - v
			// beta = ||r||
			maxtrix_TUUT(A, x, w, v);
			#pragma acc host_data use_device(A, x, w, v, b, r)
			{
				cublasHandle_t h;
				cublasCreate(&h);
				const double alpha = 1.0;
				const double beta = 0.0;
				cinst double gama = -1.0;
				cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, x, N, &beta, w, N);
				cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, x, N, A, N, &beta, v, N);
				cublasDaxpy(h, N2, &alpha, w, 1, v, 1);
				cublasDcopy(h, N2, b, 1, r, 1);
				cublasDaxpy(h, N2, &gama, v, 1, r, 1);
				cublasDnrm2(h, N2, r, 1, beta);
				cublasDestroy(h);
			}
		} //end pragma acc

		s[i+1] = beta;
		resid = s[i+1]/normb;
		if ( resid < tol)	
		{
			printf(" resid = %e \n", resid);
			printf(" Converges at %d cycle %d step. \n", m, i);
			break;
		}
	}//end outside for
}



