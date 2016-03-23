
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

int main()
{
        int N, iter;
        printf("\n Input N = ");
        scanf("%d", &N);
        printf(" Input max_iter = ");
        scanf("%d", &iter);
        printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *H_cpu, *H_oacc, *H_cublas, *Q, *v;
	double t1, t2, cpu_time, oacc_time, cublas_time;

	H = (double *) malloc(iter*sizeof(double));
	H_oacc = (double *) malloc(iter*sizeof(double));
	H_cublas = (double *) malloc(iter*sizeof(double));
	Q = (double *) malloc(N*N*(iter+1)*sizeof(double));
	v = (double *) malloc(N*N*sizeof(double));

	initial(Q, v, N, iter);

	return 0;
}

//********************************************************************************************

void initial(double *Q, double *v, int N, int iter)
{
	int i;
	for (i=0; i<N*N*(iter+1); i++)	Q[i] = sin(i);
	for (i=0; i<N*N; i++)	v[i] = cos(i);
}

void dot_cpu(double *x, double *y, double *nrm, int N)
{
	int i;
	double temp;

	temp = 0.0;
	for (i=0; i<N; i++)	temp += x[i]*y[i];
	*nrm = temp;
}

void dot_oacc(double *x, double *y, double *nrm, int N)
{
	int i;
	double temp;

	#pragma acc data present(x, y)
	{
		temp = 0.0;
		#pragma acc parallel loop reduction(+:temp)
		for (i=0; i<N; i++)	temp += x[i]*y[i];
		*nrm = temp;
	}
}

//********************************************************************************************

void cpu_test(double *H, double *Q, double *v, int N, int iter)
{
	int i, j, k;
	double *q, *dot;

	q = (dobule *) malloc(N*N*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

	for (i=0; i<iter; i++)
	{
		for (j=0; j<N*N; j++)	q[j] = Q[N*N*i+j];
		H[i] = dot_cpu(q, v, dot, N*N);
	}
}

void oacc_test(double *H, double *Q, double *v, int N, int iter)
{
	int i, j;
	double *q, *dot;

	q = (dobule *) malloc(N*N*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(Q[0:N*N*(iter+1)], v[0:N*N]) copyout(H[0:iter]) create(q[0:N*N])
	{
		for (i=0; i<iter; i++)
		{
			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	q[j] = Q[N*N*i+j];
			dot_oacc(q, v, dot, N*N);
			H[i] = *dot;
		}
	}
}

void cublas_test(double *H, double *Q, double *v, int N, int iter)
{
	int i, j;
	double *q, *dot;

	q = (dobule *) malloc(N*N*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(Q[0:N*N*(iter+1)], v[0:N*N]) copyout(H[0:iter]) create(q[0:N*N])
	{
		for (i=0; i<iter; i++)
		{
			#pragma acc parallel loop independent
			for (j=0; j<N*N; j++)	q[j] = Q[N*N*i+j];
			#pragma acc host_data use_device(q, v)
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDdot(handle, N*N, q, 1, v, 1, r);
			cublasDestroy(h);
		}
	}
}

