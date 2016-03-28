//**********************************************************************************************
//	Test:
//		hi = dot(qi, v)
//		v = v - \sum hi*qi
//**********************************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"
#include "cublas_v2.h"

void initial(double *Q, double *v_cpu, double *v_gpu, int N, int iter);
double error(double *x, double *y, int N);
void cpu_test(double *Q, double *H, double *v, int N, int iter);
void gpu_test(double *Q, double *H, double *v, int N, int iter);

int main()
{
	int N, iter;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *Q, *H_oacc, *H_cublas, *H_cpu, *v_oacc, *v_cublas, *v_cpu;
	double t1, t2, cpu_time, gpu_time;

	Q = (double *) malloc(N*N*iter*sizeof(double));
	H_cpu = (double *) malloc(iter*sizeof(double));
	H_oacc = (double *) malloc(iter*sizeof(double));
	H_cublas = (double *) malloc(iter*sizeof(double));
	v_cpu = (double *) malloc(N*N*sizeof(double));
	v_oacc = (double *) malloc(N*N*sizeof(double));
	v_cublas = (double *) malloc(N*N*sizeof(double));

	initial(Q, v_cpu, v_gpu, N, iter);
	printf(" v error = %e \n\n", error(v_gpu, v_cpu, N*N));

	t1 = clock();
	cpu_test(Q, H_cpu, v_cpu, N, iter);
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	gpu_test(Q, H_gpu, v_gpu, N, iter);
	t2 = clock();
	gpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	printf(" H error = %e, v error = %e \n", error(H_cpu, H_gpu, iter), error(v_cpu, v_gpu, N*N));
	printf(" cpu time = %f, gpu time = %f \n\n", cpu_time, gpu_time);

	return 0;
}

//**********************************************************************************************

void initial(double *Q, double *v_cpu, double *v_gpu, int N, int iter)
{
	int i;
	for (i=0; i<N*N*iter; i++)	Q[i] = sin(i);
	for (i=0; i<N*N; i++)
	{
		v_cpu[i] = i;
		v_gpu[i] = v_cpu[i];
	}
}

double error(double *x, double *y, int N)
{
	int i;
	double error, temp;

	error = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > error)	error = temp;
	}

	return error;
}

//**********************************************************************************************

void cpu_test(double *Q, double *H, double *v, int N, int iter)
{
	int i, j;

	for (i=0; i<iter; i++)
	{
		H[i] = 0.0;
		for (j=0; j<N*N; j++)	H[i] += Q[N*N*i+j]*v[j];
	}

	for (i=0; i<iter; i++)
	{
		for (j=0; j<N*N; j++)	v[j] -= H[i]*Q[N*N*i+j];
	}
}

void gpu_test(double *Q, double *H, double *v, int N, int iter)
{
	int i, j;

	#pragma acc data copyin(Q[0:N*N*iter]) copy(H[0:iter], v[0:N*N])
	{
		#pragma acc parallel loop independent
		for (i=0; i<iter; i++)
		{
			H[i] = 0.0;
			#pragma acc loop seq
			for (j=0; j<N*N; j++)	H[i] += Q[N*N*i+j]*v[j];
		}

		#pragma acc parallel loop seq
		for (i=0; i<iter; i++)
		{
			#pragma acc loop independent
			for (j=0; j<N*N; j++)	v[j] -= H[i]*Q[N*N*i+j];
		}
	} //end pragma acc
}
