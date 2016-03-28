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
void oacc_test(double *Q, double *H, double *v, int N, int iter);
void cublas_test(double *Q, double *H, double *v, int N, int iter);

int main()
{
	int N, iter;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *Q, *H_oacc, *H_cublas, *H_cpu, *v_oacc, *v_cublas, *v_cpu;
	double t1, t2, cpu_time, oacc_time, cublas_time;

	Q = (double *) malloc(N*N*iter*sizeof(double));
	H_cpu = (double *) malloc(iter*sizeof(double));
	H_oacc = (double *) malloc(iter*sizeof(double));
	H_cublas = (double *) malloc(iter*sizeof(double));
	v_cpu = (double *) malloc(N*N*sizeof(double));
	v_oacc = (double *) malloc(N*N*sizeof(double));
	v_cublas = (double *) malloc(N*N*sizeof(double));

	initial(Q, v_cpu, v_oacc, v_cublas, N, iter);

	t1 = clock();
	cpu_test(Q, H_cpu, v_cpu, N, iter);
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	oacc_test(Q, H_oacc, v_oacc, N, iter);
	t2 = clock();
	oacc_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	cublas_test(Q, H_cublas, v_cublas, N, iter);
	t2 = clock();
	cublas_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	printf(" (oacc) H error = %e, v error = %e \n", error(H_cpu, H_oacc, iter), error(v_cpu, v_oacc, N*N));
	printf(" (cublas) H error = %e, v error = %e \n", error(H_cpu, H_cublas, iter), error(v_cpu, v_cublas, N*N));
	printf(" cpu time = %f, oacc time = %f,  cublas time = %f \n\n", cpu_time, oacc_time, cublas_time);

	return 0;
}

//**********************************************************************************************

void initial(double *Q, double *v_cpu, double *v_oacc, double *v_cublas, int N, int iter)
{
	int i;
	for (i=0; i<N*N*iter; i++)	Q[i] = sin(i);
	for (i=0; i<N*N; i++)
	{
		v_cpu[i] = i;
		v_oacc[i] = v_cpu[i];
		v_cublas[i] = v_cpu[i];
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

void oacc_test(double *Q, double *H, double *v, int N, int iter)
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

void cublas_test(double *Q, double *H, double *v, int N, int iter)
{
	int i, j;
	double *temp, *dot;

	temp = (double *) malloc(N*N*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(Q[0:N*N*iter]) copy(H[0:iter], v[0:N*N]) create(temp[0:N*N])
	{
		#pragma acc host_data use_device(temp, v)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			for (i=0; i<iter; i++)
			{
				#pragma acc parallel loop independent
				for (j=0; j<N*N; j++)	temp[j] = Q[N*N*i+j];

				cublasDdot(h, N*N, temp, 1, v, 1, dot);
				H[i] = *dot;
			}
			cublasDestroy(h);
		}

		#pragma acc parallel loop seq
		for (i=0; i<iter; i++)
		{
			#pragma acc loop independent
			for (j=0; j<N*N; j++)	v[j] -= H[i]*Q[N*N*i+j];
		}
	} //end pragma acc
}
