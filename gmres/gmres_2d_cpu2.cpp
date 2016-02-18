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
	printf(" u[%d][%d] = %f \n", N/2, N/2, u[N*N/2+N/2]);
	printf(" x[%d][%d] = %f \n", N/2, N/2, x[N*N/2+N/2]);
	
	printf(" error = %e \n", error(x, u, N*N));
	printf(" times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
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

//****************************************************************************

double norm(double *x, int N)
{
	int i;
	double norm;
	norm=0.0;
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
	for(i=0; i<N; i++)
	{
		temp += x[i]*y[i];
	}
	return temp;
}

void matrix_matrix(double *A, double *x, double *b, int N)
{
	int i, j, k;
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			b[N*i+j] = 0.0;
			for (k=0; k<N; k++)
			{
				b[N*i+j] += A[N*i+k]*x[N*k+j];
			}
		}
	}
}

void q_subQ(double *q, double *Q, int N, int iter)
{
	int i;
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

//****************************************************************************

// Fast Fourier Transform in place for N = 2^p 
void fdst(double *x, int N)
{
	int i, j, k, n, M, K;
	double t_r, t_i, *x_r, *x_i, *y_r, *y_i;
	
	K = 2*N + 2;	
	x_r = (double *) malloc(N*sizeof(double));
	x_i = (double *) malloc(N*sizeof(double));
	y_r = (double *) malloc(K*sizeof(double));
	y_i = (double *) malloc(K*sizeof(double));
	
	for(i=0;i<N;i++)
	{
		x_r[i] = x[i];
		x_i[i] = 0.0;
	}

	// expand y[n] to 2N+2-points from x[n]
	y_r[0] = y_i[0] = 0.0;
	y_r[N+1] = y_i[N+1] = 0.0;
	for(i=0;i<N;i++)
	{
		y_r[i+1] = x_r[i];
		y_i[i+1] = x_i[i];
		y_r[N+i+2] = -1.0*x_r[N-1-i];
		y_i[N+i+2] = -1.0*x_i[N-1-i];
	}
	
	
	i = j = 0;
	while(i < K)
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
		M = K/2;
		while(j >= M & M > 0)
		{
			j = j - M;
			M = M / 2;
		}
		j = j + M;		
		i = i + 1;
	}
	// Butterfly structure
	double theta, w_r, w_i;
	n = 2;
	while(n <= K)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<K;i+=n)
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
		x[k] = -1.0*y_i[k+1]/2;
	}
	
}

void idst(double *x, int N)
{
	int i;
	double s;
	s = 2.0/(N+1);
	fdst(x, N);
	for(i=0;i<N;i++) x[i] = s*x[i];
}

void fast_poisson_solver(double *b, double *x, int N)
{
	int i, j;
	double h, *lamda, *temp;

	temp = (double *) malloc(N*sizeof(double));
	lamda = (double *) malloc(N*sizeof(double));
	h = 1.0/(N+1);

	for(i=0; i<N; i++)
	{
		lamda[i] = 2 - 2*cos((i+1)*M_PI*h);
	}
	
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp[j] = b[N*i+j];
		}
	}
	
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++) 
		{
			x[N*i+j] = -b[N*i+j]/(lamda[i] + lamda[j]);
		}
	}
	
}

//****************************************************************************


void gmres(double *A, double *x, double *b, int N, int max_restart, int max_iter, double tol)
{
	int i, j, k, l, m, N2;
	double resid, normb, beta, temp, *r, *q, *v, *z, *w, *cs, *sn, *s, *y, *Q, *H;
	
	Q = (double *) malloc(N*N*(max_iter+1)*sizeof(double));
	H = (double *) malloc((N+1)*max_iter*sizeof(double));
	r = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));
	v = (double *) malloc(N*N*sizeof(double));
	z = (double *) malloc(N*N*sizeof(double));
	w = (double *) malloc(N*N*sizeof(double));
	cs = (double *) malloc((max_iter+1)*sizeof(double));
	sn = (double *) malloc((max_iter+1)*sizeof(double));
	s = (double *) malloc((max_iter+1)*sizeof(double));
	y = (double *) malloc((max_iter+1)*sizeof(double));
	
	N2 = N*N;
	normb = norm(b, N2);
	
	matrix_matrix(A, x, v, N);
	matrix_matrix(x, A, z, N);
	for (i=0; i<N2; i++)	r[i] = b[i] - (v[i] + z[i]);
	beta = norm(r, N2);
	
	for (m=0; m<max_restart; m++)
	{

		for (i=0; i<N2; i++)	Q[i] = r[i]/beta;
		for (i=0; i<max_iter; i++)	s[i+1] = 0.0;
		s[0] = beta;
		
		for (i = 0; i<max_iter; i++) 
		{
	  		q_subQ(q, Q, N2, i);	
	  		matrix_matrix(A, q, v, N);
	  		matrix_matrix(q, A, z, N);
	  		for (j=0; j<N2; j++)	w[j] = v[j] + z[j];
	  		
	  		for (k=0; k<=i; k++) 
			{
				q_subQ(q, Q, N2, k);
				H[max_iter*k+i] = inner_product(q, w, N2);
				w_shift(w, q, H[max_iter*k+i], N2);
	  		}
	  		
			H[max_iter*(i+1)+i] = norm(w, N2);
			subQ_v(Q, w, N2, i+1, H[max_iter*(i+1)+i]);
			
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
				printf(" Converges at %d cycle %d step. \n", m, i+1);
				backsolve(H, s, y, N, max_iter, i);
				for(j=0; j<N; j++)
				{
					for (l=0; l<N; l++)
					{
						for(k=0; k<=i; k++)
						{
							x[N*j+l] += Q[N2*k+N*j+l]*y[k];
						}
					}
				}
				break;
	      	}
		}//end inside for
		
		if (resid < tol)	
		{
			break;
		}
		
		// Caution : i = i + 1.
		i = i - 1;
		backsolve(H, s, y, N, max_iter, i);
		for(j=0; j<N; j++)
		{
			for (l=0; l<N; l++)
			{
				for(k=0; k<=i; k++)
				{
					x[N*j+l] += Q[N2*k+N*j+l]*y[k];
				}
			}
		}
		matrix_matrix(A, x, v, N);
		matrix_matrix(x, A, z, N);
		for (j=0; j<N2; j++)	r[j] = b[j] - (v[j] + z[j]);
		beta = norm(r, N2);
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



