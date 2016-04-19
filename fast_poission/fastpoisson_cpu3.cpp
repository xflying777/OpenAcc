#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

void initial(double *u, double *b, int N);
void fast_poisson_solver(double *b, double *x, int N);
double error(double *u, double *x, int N);

int main()
{
	printf("\n");
	int N, p, q, r, s;
	printf(" Input N = 2^p * 3^q * 5^r * 7^s - 1, (p, q, r, s) =  ");
	scanf("%d %d %d %d", &p, &q, &r, &s);
	N = pow(2, p) * pow(3, q) * pow(5, r) * pow(7, s) - 1;
	printf(" N = %d \n\n", N);

	double *u, *b, *x, t1, t2;
	u = (double *) malloc(N*N*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));
	x = (double *) malloc(N*N*sizeof(double));
	
	initial(u, b, N);
	
	t1 = clock();
	fast_poisson_solver(b, x, N);
	t2 = clock();
	
	printf(" times = %f \n", 1.0*(t2 - t1)/CLOCKS_PER_SEC);
	printf(" error = %e \n", error(u, x, N));
	
	return 0;
}

double error(double *u, double *x, int N)
{
	int i, j;
	double error, temp;
	
	error = 0.0;
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp = fabs(u[N*i+j] - x[N*i+j]);
			if (temp > error)	error = temp;
		}
	}
	
	return error;
}

void initial(double *u, double *b, int N)
{
	int i, j;
	double h, x, y;
	
	h = 1.0/(N+1);
	
	for(i=0; i<N; i++)
	{
		y = (1+i)*h;
		for(j=0; j<N; j++)
		{
			x = (1+j)*h;
			u[N*i+j] = x*y*sin(M_PI*x)*sin(M_PI*y);
			b[N*i+j] = x*sin(M_PI*x)*(2*M_PI*cos(M_PI*y) - M_PI*M_PI*y*sin(M_PI*y)) + y*sin(M_PI*y)*(2*M_PI*cos(M_PI*x) - M_PI*M_PI*x*sin(M_PI*x));
		}
	}
}

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

void fast_poisson_solver(double *b, double *x, int N)
{
	int i, j;
	double h, h2, *lamda, *temp, *tempb;

	tempb = (double *) malloc(N*N*sizeof(double));
	temp = (double *) malloc(N*sizeof(double));
	lamda = (double *) malloc(N*sizeof(double));
	h = 1.0/(N+1);
	h2 = h*h;
	
	for (i=0; i<N*N; i++)	tempb[i] = b[i];

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

