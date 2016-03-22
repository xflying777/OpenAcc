
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

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

/*	printf("\n A = \n");
	print_matrix(A, N);
	printf("\n Q = \n");
	print_matrix(Q, N);
	printf("\n R = \n");
	print_matrix(R, N);
*/
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

        norm = 0.0;
        for (i=0; i<N; i++)     norm += A[i]*A[i];

        norm = sqrt(norm);

        return norm;
}

//      \sum_{i=1}^{N} A[i] * B[i] 
double dot_cpu(double *A, double *B, int N)
{
        int i;
        double result;

        result = 0.0;
        for (i=0; i<N; i++)     result += A[i]*B[i];

        return result;
}

//      B[i] = alpha * A[i] + B[i], for i = 1, ..., N
void axpy_cpu(double alpha, double *A, double *B, int N)
{
        int i;

        for (i=0; i<N; i++)     B[i] = alpha*A[i] + B[i];
}

//      B[i] = A[i], for i = 1, ..., N
void copy_cpu(double *A, double *B, int N)
{
        int i;

        for (i=0; i<N; i++)     B[i] = A[i];
}

//      A[i] = alpha * A[i], for i = 1, ..., N
void scal_cpu(double alpha, double *A, int N)
{
        int i;

        for (i=0; i<N; i++)     A[i] = alpha * A[i];
}

//******************************************************************************

void Gram_Schmidt(double *A, double *Q, double *R, int N)
{
	int i, j, k;
	double *v, *a, *q, r;

	v = (double *) malloc(N*sizeof(double));
	a = (double *) malloc(N*sizeof(double));
	q = (double *) malloc(N*sizeof(double));

	for (j=0; j<N; j++)
	{
		// vj = aj
		for (k=0; k<N; k++)
		{
			a[k] = A[N*k+j];
			v[k] = a[k];
		}

		for (i=0; i<j; i++)
		{
			//r(i,j) = qi*aj
			for (k=0; k<N; k++)	q[k] = Q[N*k+i];
			R[N*i+j] = dot_cpu(q, a, N);

			//vj = vj - r(i,j)*qi
			for (k=0; k<N; k++)	v[k] = v[k] - R[N*i+j]*q[k];
		}

		//r(j,j) = ||vj||
		//qj = vj/r(j,j)
		r = norm_cpu(v, N);
		R[N*j+j] = r;
		for (k=0; k<N; k++)	Q[N*k+j] = v[k]/r;
	}
}
