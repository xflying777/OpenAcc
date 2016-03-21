
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

void print_matrix(double *A, int N);
double error(double *x, double *y, int N);
void initial(double *A, double *B, double *C, double *D, int N);
void matrix_matrix(double *A, double *B, double *C, double alpha, double beta, int N);
void blas_dgemm(double *A, double *B, double *C, double alpha, double beta, int N);

int main()
{
	int N;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);

	double alpha, beta, t1, t2, time_cpu, time_blas;
	double *A, *B, *C, *D;

	A = (double *) malloc(N*N*sizeof(double));
	B = (double *) malloc(N*N*sizeof(double));
	C = (double *) malloc(N*N*sizeof(double));
	D = (double *) malloc(N*N*sizeof(double));

	initial(A, B, C, D, N);
	alpha = 1.0;
	beta = 0.0;

	t1 = clock();
	matrix_matrix(A, B, C, alpha, beta, N);
	t2 = clock();
	time_cpu = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	blas_dgemm(A, B, D, alpha, beta, N);
	t2 = clock();
	time_blas = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	
/*	printf(" A matrix \n");
	print_matrix(A, N);
	printf(" B matrix \n");
	print_matrix(B, N);
	printf(" C matrix \n");
	print_matrix(C, N);
	printf(" D matrix \n");
	print_matrix(D, N);
*/
	printf("\n cpu times = %f , blas times = %f \n", time_cpu, time_blas);
	printf(" error = %f \n", error(C, D, N));

	return 0;
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

double error(double *x, double *y, int N)
{
	int i;
	double error, temp;

	error = 0.0;
	for (i=0; i<N*N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > error)	error = temp;
	}

	return error;
}

void initial(double *A, double *B, double *C, double *D, int N)
{
	int i, j, temp;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp = N*i + j;
			A[temp] = i + j;
			B[temp] = i - j;
			C[temp] = 0.0;
			D[temp] = 0.0;
		}
	}

}

// C = alpha*op(A)*op(B)
void matrix_matrix(double *A, double *B, double *C, double alpha, double beta, int N)
{
	int i, j, k;
	double temp;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp = 0.0;
			for (k=0; k<N; k++)
			{
				temp += A[N*i+k]*B[N*k+j];
			}
			C[N*i+j] = alpha*temp + beta*C[N*i+j];
		}
	}

}

void blas_dgemm(double *A, double *B, double *C, double alpha, double beta, int N)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
}

