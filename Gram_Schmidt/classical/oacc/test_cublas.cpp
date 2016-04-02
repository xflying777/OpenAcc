
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"
#include "cblas.h"

//******************************************************************************
void initial(double *A, double *Q, double *R, int N);
void print_matrix(double *A, int N);
double error(double *A, double *Q, double *R, int N);
void Gram_Schmidt(double *A, double *Q, double *R, int N);

//******************************************************************************
int main()
{
	int N;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);

	double t1, t2;
	double *A, *Q, *R;

	A = (double *) malloc(N*N*sizeof(double));
	Q = (double *) malloc(N*N*sizeof(double));
	R = (double *) malloc(N*N*sizeof(double));

	initial(A, Q, R, N);

	t1 = clock();
	Gram_Schmidt(A, Q, R, N);
	t2 = clock();

	printf(" times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	printf(" error = %e \n", error(A, Q, R, N));

	return 0;
}

//******************************************************************************

void initial(double *A, double *Q, double *R, int N)
{
	int i, j;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			A[N*i+j] = sin(i+j);
		}
	}

	for (i=0; i<N*N; i++)
	{
		Q[i] = 0.0;
		R[i] = 0.0;
	}

}

void print_matrix(double *A, int N)
{
	int i, j;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	printf(" %f ", A[N*i+j]);
		printf("\n");
	}
	printf("\n");
}

double error(double *A, double *Q, double *R, int N)
{
	double *C;
	C = (double *) malloc(N*N*sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, Q, N, R, N, 0.0, C, N);
	int i;
	double error, temp;
	error = 0.0, temp;

	for (i=0; i<N*N; i++)
	{
		temp = fabs(A[i] - C[i]);
		if (temp > error)	error = temp;
	}

	return error;
}

//******************************************************************************

//      \sum_{i=1}^{N} A[i] * A[i]
void norm(double *A, double *nrm, int N)
{
	#pragma acc data present(A)
	{
		#pragma acc host_data use_device(A)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDnrm2(h, N, A, 1, nrm);
			cublasDestroy(h);
		}
	}
}

//      \sum_{i=1}^{N} A[i] * B[i] 
void dot(double *A, double *B, double *result, int N)
{
	#pragma acc data present(A, B)
	{
		#pragma acc host_data use_device(A, B)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDdot(h, N, A, 1, B, 1, result);
			cublasDestroy(h);
		}
	}
}

//******************************************************************************

void Gram_Schmidt(double *A, double *Q, double *R, int N)
{
	int i, j, k;
	double *v, *a, *q;
	double *nrm, *result;

	v = (double *) malloc(N*sizeof(double));
	a = (double *) malloc(N*sizeof(double));
	q = (double *) malloc(N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));
	result = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(A[0:N*N]) copyout(Q[0:N*N], R[0:N*N]) create(v[0:N], a[0:N], q[0:N])
	{
		for (j=0; j<N; j++)
		{
			// vj = aj
			#pragma acc kernels loop independent
			for (k=0; k<N; k++)
			{
				a[k] = A[N*k+j];
				v[k] = A[N*k+j];
			}

//			#pragma acc kernels loop independent
			for (i=0; i<j; i++)
			{
				//r(i,j) = qi*aj
				#pragma acc loop parallel independent
				for (k=0; k<N; k++)	q[k] = Q[N*k+i];
				dot(q, a, result, N);
				R[N*i+j] = *result;
			}

			//vj = vj - \sum r(i,j)*qi
			#pragma acc parallel loop seq
			for (i=0; i<j; i++)
			{
				#pragma acc parallel loop independent
				for (k=0; k<N; k++)	v[k] -= R[N*i+j]*q[k];
			}

			//r(j,j) = ||vj||
			//qj = vj/r(j,j)
			norm(v, nrm, N);
			R[N*j+j] = *nrm;
			#pragma acc parallel loop independent
			for (k=0; k<N; k++)	Q[N*k+j] = v[k] / *nrm;
		}
	}// end pragma acc data
}
