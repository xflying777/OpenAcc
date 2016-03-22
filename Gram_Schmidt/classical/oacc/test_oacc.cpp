
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
double norm_cpu(double *A, int N)
{
        int i;
        double norm;

	#pragma acc data present(A)
	{
        	norm = 0.0;
		#pragma acc parallel loop reduction(+:norm)
	        for (i=0; i<N; i++)     norm += A[i]*A[i];
	        norm = sqrt(norm);
	}

        return norm;
}

//      \sum_{i=1}^{N} A[i] * B[i] 
double dot_cpu(double *A, double *B, int N)
{
        int i;
        double result;

	#pragma acc data present(A, B)
	{
        	result = 0.0;
		#pragma acc parallel loop reduction(+:result)
		for (i=0; i<N; i++)     result += A[i]*B[i];
	}

        return result;
}

//******************************************************************************

void Gram_Schmidt(double *A, double *Q, double *R, int N)
{
	int i, j, k;
	double *v, *a, *q, r;

	v = (double *) malloc(N*sizeof(double));
	a = (double *) malloc(N*sizeof(double));
	q = (double *) malloc(N*sizeof(double));

	#pragma acc data copyin(A[0:N*N]) copyout(Q[0:N*N], R[0:N*N]) create(v[0:N], a[0:N], q[0:N])
	{
		for (j=0; j<N; j++)
		{
			// vj = aj
			#pragma acc parallel loop independent
			for (k=0; k<N; k++)
			{
				a[k] = A[N*k+j];
				v[k] = A[N*k+j];
			}

			for (i=0; i<j; i++)
			{
				//r(i,j) = qi*aj
				#pragma acc parallel loop independent
				for (k=0; k<N; k++)	q[k] = Q[N*k+i];
				R[N*i+j] = dot_cpu(q, a, N);

				//vj = vj - r(i,j)*qi
				#pragma acc parallel loop independent
				for (k=0; k<N; k++)	v[k] = v[k] - R[N*i+j]*q[k];
			}

			//r(j,j) = ||vj||
			//qj = vj/r(j,j)
			r = norm_cpu(v, N);
			R[N*j+j] = r;
			#pragma acc parallel loop independent
			for (k=0; k<N; k++)	Q[N*k+j] = v[k]/r;
		}
	}// end pragma acc data
}
