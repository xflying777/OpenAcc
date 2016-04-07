#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"
#include "cublas_v2.h"

double error(double *x, double *y, int N);
void print_vector(double *x, int N);
void print_matrix(double *A, int N);
void initial(double *A, double *u, double *b, int N);
void backsolve(double *H, double *s, double *y, int N, int i);
void cblas_backsolver(double *A, double *x, double *b, int N);
void cublas_backsolver(double *A, double *x, double *b, int N);

int main()
{
	int N;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf("\n N = %d \n\n", N);

	double *A, *u, *x, *x_blas, *x_cublas, *b;
	double t1, t2, time1, time2, time3;

	A = (double *) malloc(N*N*sizeof(double));
	u = (double *) malloc(N*sizeof(double));
	x = (double *) malloc(N*sizeof(double));
	x_blas = (double *) malloc(N*sizeof(double));
	x_cublas = (double *) malloc(N*sizeof(double));
	b = (double *) malloc(N*sizeof(double));

	initial(A, u, b, N);

	printf(" A : \n");
	print_matrix(A, N);
	printf(" u : \n");
	print_vector(u, N);
	printf(" b : \n");
	print_vector(b, N);

	t1 = clock();
	backsolve(A, b, x, N, N);
	t2 = clock();
	time1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
//	printf(" x : \n");
//	print_vector(x, N);

	t1 = clock();
	cblas_backsolver(A, x_blas, b, N);
	t2 = clock();
	time2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
//	printf(" x blas : \n");
//	print_vector(x_blas, N);

	t1 = clock();
	#pragma acc data copyin(A[0:N*N], b[0:N]) copyout(x_cublas[0:N])
	{
		cublas_backsolver(A, x_cublas, b, N);
	}
	t2 = clock();
	time3 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	printf(" x cublas : \n");
	print_vector(x_cublas, N);

	printf(" backsolver times = %f, cblas_dtrsv times = %f, cublas_Dtrsv time = %f \n", time1, time2, time3);
	printf(" error of backsolver = %e \n", error(x, u, N));
	printf(" error of cblas backsolver = %e \n", error(x_blas, u, N));
	printf(" error of cublas backsolver = %e \n", error(x_cublas, u, N));
	printf("\n");

	return 0;
}

//***********************************************************************************

double error(double *x, double *y, int N)
{
	int i;
	double temp, error;
	error = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > error)	error = temp;
	}
	return error;
}
		
void print_vector(double *x, int N)
{
	int i;
	for (i=0; i<N; i++)	printf(" %f \n", x[i]);
	printf("\n");
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

void initial(double *A, double *u, double *b, int N)
{
	int i, j;

	for (i=0; i<N*N; i++)	A[i] = 0.0;
	for (i=0; i<N; i++)
	{
		for (j=0; j<=i; j++)	A[N*j+i] = 1.0*(i + j + 1);
	}

	for (i=0; i<N; i++)	u[i] = 1.0*(N - i);

	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, u, 1, 0.0, b, 1);
}

//***********************************************************************************

// Solve H * y = s
void backsolve(double *H, double *s, double *y, int N, int i)
{
	// i = iter
	int j, k;
	double temp;
	
	for(j=i; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=i; k++)
		{
			temp -= y[k]*H[N*j+k];
		}
		y[j] = temp/H[N*j+j];
	}
}

// Solve A * x = b, input A and b.
void cblas_backsolver(double *A, double *x, double *b, int N)
{
//	int i;
//	for (i=0; i<N; i++)	x[i] = b[i];
	cblas_dcopy(N, b, 1, x, 1);
	cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, N, A, N, x, 1);
}

// Solve A * x = b in GPU.
void cublas_backsolver(double *A, double *x, double *b, int N)
{
	#pragma acc data present(A, x, b)
	{
		#pragma host_data use_device(A, x, b)
		{
			cublasHandle_t h;
			cublasCreate(&h);
			cublasDcopy(h, N, b, 1, x, 1);
			cublasDtrsv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUSPARSE_DIAG_TYPE_NON_UNIT, N, A, N, x, 1);
			cublasDestroy(h);
		}
	}
}

