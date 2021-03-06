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
void initial(double *A, double *b, double *x0, double *u, int N);
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
	
	initial(A, b, x, u, N);	
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

void initial(double *A, double *b, double *x0, double *u, int N)
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
		u[i] = x*sin(x);
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

void matrix_vector(double *A, double *x, double *b, int N)
{
	int i, j;
	for(i=0; i<N; i++)
	{
		b[i] = 0.0;
		for(j=0; j<N; j++)
		{
			b[i] += A[N*i+j]*x[j];
		}
	}
}

void q_subQ(double *q, double *Q, int N, int max_iter, int iter)
{
	int i;
	for(i=0; i<N; i++)
	{
		q[i] = Q[(max_iter+1)*i + iter];
	}
}

void subQ_v(double *Q, double *v, int N, int max_iter, int iter, double norm_v)
{
	int i;
	for(i=0; i<N; i++)
	{
		Q[(max_iter+1)*i + iter] = v[i]/norm_v;
	}
}

void v_shift(double *v, double *q, double h, int N)
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

/*void ApplyPlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
	Real temp  =  cs * dx + sn * dy;
	dy = -sn * dx + cs * dy;
	dx = temp;
}
*/

void gmres(double *A, double *x, double *b, int N, int max_restart, int max_iter, double tol)
{
	int i, j, k, m;
	double resid, normb, beta, temp, *tempv, *r, *q, *v, *cs, *sn, *s, *y, *Q, *H;
	
	Q = (double *) malloc(N*(max_iter+1)*sizeof(double));
	H = (double *) malloc((N+1)*max_iter*sizeof(double));
	tempv = (double *) malloc(N*sizeof(double));
	r = (double *) malloc(N*sizeof(double));
	q = (double *) malloc(N*sizeof(double));
	v = (double *) malloc(N*sizeof(double));
	cs = (double *) malloc((N+1)*sizeof(double));
	sn = (double *) malloc((N+1)*sizeof(double));
	s = (double *) malloc((N+1)*sizeof(double));
	y = (double *) malloc((N+1)*sizeof(double));
		
	normb = norm(b, N);
	matrix_vector(A, x, v, N);
	for (i=0; i<N; i++)	r[i] = b[i] - v[i];
	beta = norm(r, N);
	
	for (m=0; m<max_restart; m++)
	{

		for (i=0; i<N; i++)
		{
			s[i+1] = 0.0;
			Q[(max_iter+1)*i+0] = r[i]/beta;
		}
		s[0] = beta;
		
		for (i = 0; i<max_iter; i++) 
		{
	  		q_subQ(q, Q, N, max_iter, i);
	  		matrix_vector(A, q, v, N);
	  		for (k=0; k<=i; k++) 
			{
				q_subQ(q, Q, N, max_iter, k);
	    		H[max_iter*k+i] = inner_product(q, v, N);
	    		v_shift(v, q, H[max_iter*k+i], N);
	  		}
	  		
			H[max_iter*(i+1)+i] = norm(v, N);
			subQ_v(Q, v, N, max_iter, i+1, H[max_iter*(i+1)+i]);
			
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
				for(j=0; j<N; j++)
				{
					for(k=0; k<=i; k++)
					{
						x[j] += Q[(max_iter+1)*j+k]*y[k];
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
		backsolve(H, s, y, N, max_iter, i-1);
		for(j=0; j<N; j++)
		{
			for(k=0; k<=i-1; k++)
			{
				x[j] += Q[(max_iter+1)*j+k]*y[k];
			}
		}
		matrix_vector(A, x, tempv, N);
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
}



