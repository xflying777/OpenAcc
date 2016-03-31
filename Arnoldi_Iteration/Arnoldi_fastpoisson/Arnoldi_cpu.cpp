
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

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

	printf(" Q[0, N*N, N*N*iter] = %f %f %f \n", Q[0], Q[N*N], Q[N*N*iter]);
	printf(" Times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	return 0;
}

//***********************************************************************************

void initial(double *A, double *b, int N)
{
	int i;
	double h, h2;
	h = 1.0/(1+N);
	h2 = h*h;

	for (i=0; i<N*N; i++)
	{
		A[i] = 0.0;
		b[i] = sin(1.0*i);
	}

	for (i=0; i<N; i++)	A[N*i+i] = -2.0/h2;
	for (i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = 1.0/h2;
		A[N*i+(i+1)] = 1.0/h2;
	}
}

//***********************************************************************************

double norm(double *x, int N)
{
	double temp;
	temp = cblas_dnrm2(N, x, 1);
	return temp;
}

void gemm(double *A, double *x, double *b, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, x, N, 0.0, b, N);
}

double dot(double *x, double *y, int N)
{
	double temp;
	temp = cblas_ddot(N, x, 1, y, 1);
	return temp;
}

//***********************************************************************************

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

//***********************************************************************************

void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));


	fastpoisson(b, q, N);
	*nrm = norm(q, N*N);
	for (k=0; k<N*N; k++)	Q[k] = q[k] / *nrm;

	for (i=0; i<iter; i++)
	{
		// v= A*qi
		for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
		gemm(A, q, v, N);
		fastpoisson(v, v, N);

		// h(j,i) = qj*v
		for (j=0; j<=i; j++)
		{
			for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
			H[iter*j+i] = dot(q, v, N*N);
		}

		// v = v - \sum h(j,i)*qj
		for (j=0; j<=i; j++)
		{
			for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
		}

		// h(i+1,i) = ||v||
		*nrm = norm(v, N*N);
		H[iter*(i+1)+i] = *nrm;
		// qi+1 = v/h(i+1,i)
		for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / *nrm;
	}
}

