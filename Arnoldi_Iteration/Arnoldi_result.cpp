//********************************************************
// Check the result of gpu and cpu.
//********************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"
#include "cblas.h"

void initial(double *A, double *b, int N);
void initial_QH(double *Q, double *H, int N, int iter);
double error(double *x, double *y, int N);
void Arnoldi_gpu(double *A, double *Q, double *H, double *b, int N, int iter);
void Arnoldi_cpu(double *A, double *Q, double *H, double *b, int N, int iter);

//**********************************************************************************

int main()
{
	int N, iter;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *A, *Q_gpu, *Q_cpu, *H_gpu, *H_cpu, *b;
	double t1, t2, gpu_time, cpu_time;

	A = (double *) malloc(N*N*sizeof(double));
	Q_gpu = (double *) malloc(N*N*(iter+1)*sizeof(double));
	Q_cpu = (double *) malloc(N*N*(iter+1)*sizeof(double));
	H_gpu = (double *) malloc((iter+1)*iter*sizeof(double));
	H_cpu = (double *) malloc((iter+1)*iter*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));

	initial(A, b, N);
	initial_QH(Q_gpu, H_gpu, N, iter);
	initial_QH(Q_cpu, H_cpu, N, iter);

	t1 = clock();
	Arnoldi_gpu(A, Q_gpu, H_gpu, b, N, iter);
	t2 = clock();
	gpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	printf(" \n");

	t1 = clock();
	Arnoldi_cpu(A, Q_cpu, H_cpu, b, N, iter);
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	printf(" \n");
	printf(" gpu times = %f \n cpu times = %f \n", gpu_time, cpu_time);
	printf(" Q error = %f \n H error = %f \n\n", error(Q_gpu, Q_cpu, N*N*(iter+1)), error(H_gpu, H_cpu, (iter+1)*iter));
	return 0;
}

//***********************************************************************************

void initial(double *A, double *b, int N)
{
	int i;
	for (i=0; i<N*N; i++)
	{
		A[i] = i/N;
		b[i] = cos(i);
	}
}

void initial_QH(double *Q, double *H, int N, int iter)
{
	int i;
	for (i=0; i<N*N*(iter+1); i++)	Q[i] = 0.0;
	for (i=0; i<(iter+1)*iter; i++)	H[i] = 0.0;
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
//***********************************************************************************

void norm_gpu(double *x, double *nrm, int N)
{
	#pragma acc data present(x)
	{
		#pragma acc host_data use_device(x)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDnrm2(h, N, x, 1, nrm);
			cublasDestroy(h);
		}
	}
//	printf(" cublasDnrm2 success \n");
}

// Ax = b
void dgemm_gpu(double *A, double *x, double *b, int N)
{
	#pragma acc data present(A, x, b)
	{
		#pragma acc host_data use_device(A, x, b)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			const double alpha = 1.0;
			const double beta = 0.0;
			cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, x, N, &beta, b, N);
			cublasDestroy(h);
		}
	}
//	printf(" cublasDgemm success \n");
}

//***********************************************************************************

void norm_cpu(double *x, double *nrm, int N)
{
	int i;
	double temp;
	temp = 0.0;
	for (i=0; i<N; i++)	temp += x[i]*x[i];
	*nrm = sqrt(temp);
}

//***********************************************************************************

void Arnoldi_gpu(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, temp, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));

	t1 = clock();
	#pragma acc data copyin(b[0:N*N]) copyout(Q[0:N*N*(iter+1)])
	{
		norm_gpu(b, nrm, N*N);
		temp = *nrm;
		#pragma acc parallel loop independent
		for (k=0; k<N*N; k++)	Q[k] = b[k] / temp;
	}
	t2 = clock();
	printf(" GPU First step times = %f , normb = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC, temp);

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		#pragma acc data copyin(A[0:N*N]) copy(Q[0:N*N*(iter+1)], H[0:(iter+1)*iter]) create(v[0:N*N], q[0:N*N])
		{
			// v= A*qi
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
			dgemm_gpu(A, q, v, N);

			printf(" Q[%d] = %f, Q[%d] = %f \n", 0, Q[0], N*N-1, Q[N*N-1]);
			printf(" v[%d] = %f, v[%d] = %f \n", 0, v[0], N*N-1, v[N*N-1]);
			// h(j,i) = qj*v
			#pragma acc parallel loop independent
			for (j=0; j<=i; j++)
			{
				H[iter*j+i] = 0.0;
				#pragma acc loop seq
				for (k=0; k<N*N; k++)	H[iter*j+i] += Q[N*N*j+k]*v[k];
			}
			printf(" H[%d,%d] = %f ", i, i, H[iter*i+i]);
			// v = v - \sum h(j,i)*qj
			#pragma acc parallel loop seq
			for (j=0; j<=i; j++)
			{
				#pragma acc loop independent
				for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
			}

			// h(i+1,i) = ||v||
			norm_gpu(v, nrm, N*N);
			H[iter*(i+1)+i] = *nrm;
			temp = *nrm;
			// qi+1 = v/h(i+1,i)
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / temp;
		} //end pragma acc
	}
	printf(" \n");
	t2 = clock();
	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}

//***********************************************************************************

void Arnoldi_cpu(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));

	t1 = clock();
	norm_cpu(b, nrm, N*N);
	for (k=0; k<N*N; k++)	Q[k] = b[k] / *nrm;
	t2 = clock();
	printf(" CPU First step times = %f , normb = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC, *nrm);

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		// v= A*qi
		for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, q, N, 0.0, v, N);
		printf(" Q[%d] = %f, Q[%d] = %f \n", 0, Q[0], N*N-1, Q[N*N-1]);
		printf(" v[%d] = %f, v[%d] = %f \n", 0, v[0], N*N-1, v[N*N-1]);

		// h(j,i) = qj*v
		for (j=0; j<=i; j++)
		{
			H[iter*j+i] = 0.0;
			for (k=0; k<N*N; k++)	H[iter*j+i] += Q[N*N*j+k]*v[k];
		}

		printf(" H[%d,%d] = %f ", i, i, H[iter*i+i]);
		// v = v - \sum h(j,i)*qj
		for (j=0; j<=i; j++)
		{
			for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
		}

		// h(i+1,i) = ||v||
		norm_cpu(v, nrm, N*N);
		H[iter*(i+1)+i] = *nrm;
		// qi+1 = v/h(i+1,i)
		for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / *nrm;
	}
	t2 = clock();
	printf(" \n");
	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}
