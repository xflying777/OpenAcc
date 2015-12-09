#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double Error(double **X, double **U, int N);
void Print_Matrix(double **A, int N);
void Exact_Solution(double **U, int N);
void Exact_Source(double **F, int N);
void DST(double *x, int N);
void iDST(double *x, int N);
void Transpose(double **A, int N);
void Fast_Poisson_Solver(double **F, double **X, int N);

int main()
{
	int i, N, L, p; 
	double **X, **U, **F;
	clock_t t1, t2;
	p = pow(2, 12);
	// Create memory for solving Ax = b, where r = b-Ax is the residue.
	for(N=4;N<=p;N*=2)
	{
		// M is the total number of unknowns.
		L = (N-1);
		// Prepare for two dimensional unknown F
		// where b is the one dimensional vector and 
		// F[i][j] = F[j+i*(N-1)];
		F = (double **) malloc(L*sizeof(double*));
		F[0] = (double *) malloc(L*L*sizeof(double));
		for(i=1;i<L;++i) F[i] = F[i-1] + L;

		X = (double **) malloc(L*sizeof(double*));
		X[0] = (double *) malloc(L*L*sizeof(double));
		for(i=1;i<L;++i) X[i] = X[i-1] + L;
				
		// Prepare for two dimensional unknowns U
		// where u is the one dimensional vector and
		// U[i][j] = u[j+i*(N-1)] 
		U = (double **) malloc(L*sizeof(double*));
		U[0] = (double *) malloc(L*L*sizeof(double));
		for(i=1;i<L;++i) U[i] = U[i-1] + L;
		
		Exact_Solution(U, N);
		Exact_Source(F, N);
		t1 = clock();
		Fast_Poisson_Solver(F, X, L);
		t2 = clock();
		printf(" Fast Poisson Solver: %f secs\n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
		printf(" For N = %d error = %e \n", N, Error(X, U, L));
		printf(" \n");
		free(X);
		free(U);
		free(F);
	}
	return 0;
}
void Exact_Solution(double **U, int N)
{
	// put the exact solution 
	int i,j;
	double x, y, h;
	h = 1.0/N;
	for(i=0;i<N-1;++i)
	{
		for(j=0;j<N-1;++j)
		{
			//k = j + i*(N-1);
			x = (i+1)*h;
			y = (j+1)*h;
			U[i][j] = sin(M_PI*x)*sin(2*M_PI*y);
		}
	}
}
void Exact_Source(double **F, int N)
{
	int i,j;
	double x, y, h;
	h = 1.0/N;
	for(i=0;i<N-1;++i)
	{
		for(j=0;j<N-1;++j)
		{
			//k = j + i*(N-1);
			x = (i+1)*h;
			y = (j+1)*h;
			F[i][j] = -(1.0+4.0)*h*h*M_PI*M_PI*sin(M_PI*x)*sin(2*M_PI*y);
		}
	}	
}
void Print_Matrix(double **A, int N)
{
	int i, j;
	for(i=0;i<N;++i)
	{
		for(j=0;j<N;++j)
		{
			printf("%.0f ",A[i][j]);
		}
		printf("\n");
	}
}
double Error(double **X, double **U, int N)
{
	// return max_i |x[i] - u[i]|
	int i, j;
	double v = 0.0, e;
	
	for(i=0;i<N;++i)
	{
		for(j=0;j<N;j++)
		{
			e = fabs(X[i][j] - U[i][j]);
			if(e>v) v = e;		
		}
	}
	return v;
}

// Fast Fourier Transform in place for N = 2^p 
void DST(double *x, int N)
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
void iDST(double *x, int N)
{
	int i;
	double s;
	s = 2.0/(N+1);
	DST(x, N);
	for(i=0;i<N;i++) x[i] = s*x[i];
}
void Transpose(double **A, int N)
{
	int i, j;
	double v;
	for(i=0;i<N;++i)
	{
		for(j=i+1;j<N;++j)
		{
			v = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = v;
		}
	}
}
void Fast_Poisson_Solver(double **F, double **X, int N)
{
	int i, j;
	double h, *lamda, **Xbar;
	Xbar = (double **) malloc(N*sizeof(double*));
	Xbar[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) Xbar[i] = Xbar[i-1] + N;
		
	lamda = (double *) malloc(N*sizeof(double));
	h = 1.0/(N+1);
	
	for(i=0;i<N;i++)
	{
		lamda[i] = 2 - 2*cos((i+1)*M_PI*h);
	}
	for(i=0;i<N;i++) DST(F[i], N);
	Transpose(F, N);
	for(i=0;i<N;i++) iDST(F[i], N);
	Transpose(F, N);
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) 
		{
			Xbar[i][j] = -F[i][j]/(lamda[i] + lamda[j]);
		}
	}
	
	for(i=0;i<N;i++) DST(Xbar[i], N);
	Transpose(Xbar, N);
	for(i=0;i<N;i++) iDST(Xbar[i], N);
	Transpose(Xbar, N);
	
	for(i=0;i<N;i++) X[i] = Xbar[i];
}




