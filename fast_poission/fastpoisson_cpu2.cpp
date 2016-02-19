#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void initial(double *u, double *b, int N);
void fast_poisson_solver(double *b, double *x, int N);
double error(double *u, double *x, int N);
void fdst(double *x, int N);

int main()
{
	int N, p;
	printf(" Input N = 2^p - 1, p = ");
	scanf("%d", &p);
	N = pow(2, p) - 1;
	
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

// Fast Fourier Transform in place for N = 2^p 
void fdst(double *x, int N)
{
	int i, j, k, n, M, K;
	double s, t_r, t_i, *x_r, *x_i, *y_r, *y_i;
	
	s = sqrt(2.0/(N+1));
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
		x[k] = -1.0*s*y_i[k+1]/2;
	}
	
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

