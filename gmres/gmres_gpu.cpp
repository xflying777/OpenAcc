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

//*****************************************************************
// Compile command : pgc++ -acc -ta=tesla:managed filename.cpp
// managed : 
//*****************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_vector(double *x, int N);
void matrix_vector(double *A, double *x, double *b, int N);
void print_matrixH(double *x, int N, int k);
void exact_solution(double *u, int N);
void initial(double *A, double *b, double *x0, int N);
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
	x = (double *) malloc(N*sizeof(double));
	b = (double *) malloc(N*sizeof(double));
	u = (double *) malloc(N*sizeof(double));
	
	exact_solution(u, N);
	initial(A, b, x, N);	
	tol = 1.0e-4;
	
	t1 = clock();
	gmres(A, x, b, N, max_restart, max_iter, tol);
	t2 = clock();
	
	printf(" error = %e \n", error(x, u, N));
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

void print_matrixH(double *x, int N, int k)
{
	int i, j;
	for(i=0;i<=k;i++)
	{
		for (j=0;j<=k;j++) printf(" %f ", x[N*i+j]);
		printf("\n");
	}
	printf("\n");
}

void exact_solution(double *u, int N)
{
	int i;
	double h, x;
	h = M_PI/(N+1);
	for(i=0; i<N; i++)
	{
		x = (1+i)*h;
		u[i] = x*sin(x);
	}
}

void initial(double *A, double *b, double *x0, int N)
{
	int i, j;
	double h, h2, temp, x;
	
	h = M_PI/(N+1);
	h2 = h*h;
	
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			A[N*i+j] = 0.0;
		}
	}
	temp = -2.0/h2;
	for(i=0; i<N; i++)
	{
		x0[i] = 0.0;
		x = (1+i)*h;
		b[i] = 2*cos(x) - x*sin(x);
			
		A[N*i+i] = temp;
	}
	temp = 1.0/h2;
	for(i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = temp;
		A[N*i+(i+1)] = temp;
	}
}

double norm(double *x, int N)
{
	int i;
	double norm;
	norm=0.0;
	#pragma acc parallel loop seq
	for(i=0; i<N; i++)
	{
		norm += x[i]*x[i];
	}
	norm = sqrt(norm);
	return norm;
}

double inner_product(double *x, double *y, int N)
{
	int i;
	double temp;
	temp = 0.0;
	#pragma acc parallel loop seq present(x[0:N], y[0:N])
	for(i=0; i<N; i++)
	{
		temp += x[i]*y[i];
	}
	return temp;
}

void matrix_vector(double *A, double *x, double *b, int N)
{
	int i, j;
	#pragma acc parallel loop independent present(A[0:N*N], x[0:N], b[0:N])
	for(i=0; i<N; i++)
	{
		b[i] = 0.0;
		#pragma acc loop seq
		for(j=0; j<N; j++)
		{
			b[i] += A[N*i+j]*x[j];
		}
	}
}

void q_subQ(double *q, double *Q, int N, int iter)
{
	int i;
	#pragma acc parallel loop independent present(q[0:N], Q[0:(N+1)*N])
	for(i=0; i<N; i++)
	{
		q[i] = Q[(N+1)*i + iter];
	}
}

void subQ_v(double *Q, double *v, int N, int iter, double norm_v)
{
	int i;
	#pragma acc parallel loop independent present(Q[0:(N+1)*N], v[0:N])
	for(i=0; i<N; i++)
	{
		Q[(N+1)*i + iter] = v[i]/norm_v;
	}
}

void v_shift(double *v, double *q, double h, int N)
{
	int i;
	#pragma acc parallel loop independent present(v[0:N], q[0:N])
	for(i=0; i<N; i++)
	{
		v[i] -= h*q[i];
	}
}

void backsolve(double *H, double *s, double *y, int N, int i)
{
	// i = iter
	int j, k;
	double temp;
	
	for(j=i; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=i; k++)
		{
			temp -= y[k]*H[N*j+k];
		}
		y[j] = temp/H[N*j+j];
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

void gmres(double *A, double *x, double *b, int N, int max_restart, int max_iter, double tol)
{
	int i, j, k, m;
	double resid, normb, beta, temp, *tempv, *r, *q, *v, *cs, *sn, *s, *y, *Q, *H;

	Q = (double *) malloc(N*(N+1)*sizeof(double));
	H = (double *) malloc((N+1)*N*sizeof(double));
	tempv = (double *) malloc(N*sizeof(double));
	r = (double *) malloc(N*sizeof(double));
	q = (double *) malloc(N*sizeof(double));
	v = (double *) malloc(N*sizeof(double));
	cs = (double *) malloc((N+1)*sizeof(double));
	sn = (double *) malloc((N+1)*sizeof(double));
	s = (double *) malloc((N+1)*sizeof(double));
	y = (double *) malloc(N*sizeof(double));

	normb = norm(b, N);

	#pragma acc data create(Q[0:(N+1)*N], H[0:(N+1)*N], tempv[0:N], r[0:N], q[0:N], v[0:N], cs[0:N+1], sn[0:N+1], s[0:N+1], y[0:N]), copyin(A[0:N*N], b[0:N]), copy(x[0:N])
	{
	matrix_vector(A, x, tempv, N);
	#pragma acc parallel loop independent
	for (i=0; i<N; i++)	r[i] = b[i] - tempv[i];
	beta = norm(r, N);

	for (m=0; m<max_restart; m++)
	{
		#pragma acc parallel loop independent
		for (i=0; i<N; i++)
		{
			s[i+1] = 0.0;
			Q[(N+1)*i+0] = r[i]/beta;
		}
		s[0] = beta;

		for (i = 0; i<max_iter; i++)
		{
	  		q_subQ(q, Q, N, i);
	  		matrix_vector(A, q, v, N);

	  		for (k=0; k<=i; k++)
			{
				q_subQ(q, Q, N, k);
				H[N*k+i] = inner_product(q, v, N);
				v_shift(v, q, H[N*k+i], N);
	  		}

			H[N*(i+1)+i] = norm(v, N);
			subQ_v(Q, v, N, i+1, H[N*(i+1)+i]);

			#pragma acc kernels
			for (k = 0; k < i; k++)
			{
				//ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k))
				temp = cs[k]*H[N*k+i] + sn[k]*H[N*(k+1)+i];
				H[N*(k+1)+i] = -1.0*sn[k]*H[N*k+i] + cs[k]*H[N*(k+1)+i];
				H[N*k+i] = temp;
			}

			GeneratePlaneRotation(H[N*i+i], H[N*(i+1)+i], cs, sn, i);
			//ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i))
			H[N*i+i] = cs[i]*H[N*i+i] + sn[i]*H[N*(i+1)+i];
			H[N*(i+1)+i] = 0.0;
			//ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
			temp = cs[i]*s[i];
			s[i+1] = -1.0*sn[i]*s[i];
			s[i] = temp;
			resid = fabs(s[i+1]/beta);

			if (resid < tol)
			{
				printf(" resid = %e \n", resid);
				printf(" Converges at %d step ", i+1);
				backsolve(H, s, y, N, i);
				#pragma acc parallel loop independent
				for(j=0; j<N; j++)
				{
					#pragma acc loop seq
					for(k=0; k<=i; k++)
					{
						x[j] += Q[(N+1)*j+k]*y[k];
					}
				}
				break;
			}
		}//end inside for
		// Caution : i = i + 1.
		if (resid < tol)
		{
			printf(" %d cycle \n", m);
			break;
		}

		backsolve(H, s, y, N, i-1);
		#pragma acc parallel loop independent
		for(j=0; j<N; j++)
		{
			#pragma acc loop seq
			for(k=0; k<=i-1; k++)
			{
				x[j] += Q[(N+1)*j+k]*y[k];
			}
		}
		matrix_vector(A, x, tempv, N);
		#pragma acc parallel loop independent
		for (j=0; j<N; j++)	r[j] = b[j] - tempv[j];
		beta = norm(r, N);
		s[i] = beta;
		resid = s[i]/normb;
		if ( resid < tol)
		{
			printf(" resid = %e \n", resid);
			printf(" Converges at %d cycle %d step. \n", m, i);
			break;
		}
	}//end outside for
	}//end pragma acc
}



