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
#include <fftw3.h>
#include "cblas.h"

void initial(double *x, double *b, double *Beta, double *D, double *u, int N);
double error(double *x, double *y, int N);
void fixpoint_iteration(double *Beta, double *D, double *x, double *b, int N, double tol);

int main()
{
	int N, p;
	printf("\n Input N = 2^p - 1, p = ");
	scanf("%d", &p);
	N = pow(2, p) - 1;
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
//	printf(" Initial finish. \n");
}

//***********************************************************************************************************

// b = alpha * A * x + beta * b;
void dgemm(double *b, double *A, double *x, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, x, N, 0.0, b, N);
}

//***********************************************************************************************************

// Fast Fourier Transform
void fdst(double *x, int N)
{
	int i, K;
	double s;
	fftw_complex *in, *out;
	
	s = sqrt(2.0/(N+1));
	K = 2*N + 2;

	in = (fftw_complex *) malloc(K*sizeof(fftw_complex));
	out = (fftw_complex *) malloc(K*sizeof(fftw_complex));

	in[0][0] = in[0][1] = 0.0;
	in[N+1][0] = in[N+1][1] = 0.0;

	for (i=0; i<N; i++)
	{
		in[i+1][0] = x[i];
		in[i+1][1] = 0.0;
		in[N+i+2][0] = -1.0*x[N-1-i];
		in[N+i+2][1] = 0.0;
	}

	fftw_plan plan;
	plan = fftw_plan_dft_1d(K, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	
	// After fft(y[k]), Y[k] = fft(y[k]), Sx[k] = i*Y[k+1]/2
	for (i=0; i<N; i++)	x[i] = -1.0*s*out[i+1][1]/2.0;	

	free(in);
	free(out);
}

// \Delta x = b
void fastpoisson(double *b, double *x, int N)
{

	int i, j;
	double h, h2, *lamda, *temp, *tempb;

	tempb = (double *) malloc(N*N*sizeof(double));
	temp = (double *) malloc(N*sizeof(double));
	lamda = (double *) malloc(N*sizeof(double));

	h = 1.0/(N+1);
	h2 = h*h;

	for(i=0; i<N*N; i++)	tempb[i] = b[i];

	for(i=0; i<N; i++)
	{
		lamda[i] = 2 - 2*cos((i+1)*M_PI*h);
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	temp[j] = tempb[N*i+j];
		fdst(temp, N);
		for (j=0; j<N; j++)	tempb[N*i+j] = temp[j];
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	temp[j] = tempb[N*j+i];
		fdst(temp, N);
		for (j=0; j<N; j++)	tempb[N*j+i] = temp[j];
	}

	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			x[N*i+j] = -1.0*h2*tempb[N*i+j]/(lamda[i] + lamda[j]);
		}
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	temp[j] = x[N*i+j];
		fdst(temp, N);
		for (j=0; j<N; j++)	x[N*i+j] = temp[j];
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	temp[j] = x[N*j+i];
		fdst(temp, N);
		for (j=0; j<N; j++)	x[N*j+i] = temp[j];
	}
}

//***********************************************************************************************************

// x(k+1) = M^(-1) * (b - D * x(k))
void fixpoint_iteration(double *Beta, double *D, double *x, double *b, int N, double tol)
{
	int i, j;

	double *xk, *temp, error;

	xk = (double *) malloc(N*N*sizeof(double));
	temp = (double *) malloc(N*N*sizeof(double));

	for (i=0; i<N*N; i++)
	{
		for (j=0; j<N*N; j++)	xk[j] = x[j];

		dgemm(temp, D, xk, N);
//		if (i == 0)	printf(" Dgemm finish. \n");

		for (j=0; j<N*N; j++)	temp[j] = b[j] - Beta[j]*temp[j];

		fastpoisson(temp, x, N);
//		if(i == 0)	printf(" Fast Poisson finish. \n");

		for (j=0; j<N*N; j++)	temp[j] = x[j] - xk[j];
		error = cblas_dnrm2(N*N, temp, 1);

//		printf(" Step %d finish. \n", i+1);
		if ( error < tol)
		{
			printf("\n Converges at %d step ! \n", i+1);
			break;
		}
	}
}
