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
	printf(" Q_gpu[0, N*N, N*N*(iter)] = %f %f %f \n", Q_gpu[0], Q_gpu[N*N], Q_gpu[N*N*iter]);

	printf(" \n");
	t1 = clock();
	Arnoldi_cpu(A, Q_cpu, H_cpu, b, N, iter);
	t2 = clock();
	cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	printf(" Q_cpu[0, N*N, N*N*(iter)] = %f %f %f \n", Q_cpu[0], Q_cpu[N*N], Q_cpu[N*N*iter]);

	printf(" \n");
	printf(" gpu times = %f \n cpu times = %f \n", gpu_time, cpu_time);
	printf(" Q error = %e \n H error = %e \n\n", error(Q_gpu, Q_cpu, N*N*(iter+1)), error(H_gpu, H_cpu, (iter+1)*iter));
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

	for (i=0; i<N; i++)	A[i] = -2.0;
	for (i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = 1.0;
		A[N*i+(i+1)] = 1.0;
	}
//	for (i=0; i<N*N; i++)
//	{
//		A[i] = sin(i);
//		b[i] = cos(i);
//	}
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

void dot_gpu(double *x, double *y, double *dot, int N)
{
	#pragma acc data present(x, y)
	{
		#pragma acc host_data use_device(x, y)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDdot(h, N, x, 1, y, 1, dot);
			cublasDestroy(h);
		}
	}
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
			cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, x, N, A, N, &beta, b, N);
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

void dgemm_cpu(double *A, double *x, double *b, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, x, N, 0.0, b, N);
}

void dgemv_cpu(double *A, double *x, double *b, int N)
{
	int i, j;
	for (i=0; i<N; i++)	x[i] = sin(i);
	for (i=0; i<N; i++)
	{
		b[i] = 0.0;
		for (j=0; j<N; j++)	b[i] += A[N*i+j]*x[j];
	}
}

//***********************************************************************************

void Arnoldi_gpu(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q, *x1, *x2;
	double *nrm, *dot, temp, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	x1 = (double *) malloc(N*sizeof(double));
	x2 = (double *) malloc(N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

        norm_cpu(b, nrm, N*N);
	temp = *nrm;
        for (k=0; k<N*N; k++)   Q[k] = b[k] / temp;

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		#pragma acc data copyin(A[0:N*N]) copy(Q[0:N*N*(iter+1)], H[0:(iter+1)*iter]) create(v[0:N*N], q[0:N*N])
		{
			// v= A*qi
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
			#pragma acc data present(A, q, v)
			dgemm_gpu(A, q, v, N);

			// hj,i = qj*v
			for (j=0; j<=i; j++)
			{
				#pragma acc parallel loop independent
				for (k=0; k<N*N; k++)	q[k] = Q[N*N*j+k];

				dot_gpu(q, v, dot, N*N);
				H[iter*j+i] = *dot;
			}

//			if (i==0)	printf(" %f \n", H[0]);
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
//		dgemv_cpu(A, x1, x2, N);
	}
	t2 = clock();
//	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}

//***********************************************************************************

void Arnoldi_cpu(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q, *x1, *x2;
	double *nrm, temp, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

        x1 = (double *) malloc(N*sizeof(double));
        x2 = (double *) malloc(N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));
//	dot = (double *) malloc(1*sizeof(double));

	norm_cpu(b, nrm, N*N);
	temp = *nrm;
	for (k=0; k<N*N; k++)	Q[k] = b[k]/temp;

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		// v= A*qi
		for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
		dgemm_cpu(A, q, v, N);

		// h(j,i) = qj*v
		for (j=0; j<=i; j++)
		{
			for (k=0; k<N*N; k++)   q[k] = Q[N*N*j+k];
			H[iter*j+i] = cblas_ddot(N*N, q, 1, v, 1);
		}

//		if (i==0)       printf(" %f \n", H[0]);
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

//		dgemv_cpu(A, x1, x2, N);
	}
	t2 = clock();
//	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}
