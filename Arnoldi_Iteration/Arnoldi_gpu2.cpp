
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cublas_v2.h"

void initial(double *A, double *b, int N);
void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter);

//**********************************************************************************

int main()
{
	int N, iter;
	printf("\n Input N = ");
	scanf("%d", &N);
	printf(" Input max_iter = ");
	scanf("%d", &iter);
	printf(" N = %d, max_iter = %d \n\n", N, iter);

	double *A, *Q, *H, *b;
	double t1, t2;

	A = (double *) malloc(N*N*sizeof(double));
	Q = (double *) malloc(N*N*(iter+1)*sizeof(double));
	H = (double *) malloc((iter+1)*iter*sizeof(double));
	b = (double *) malloc(N*N*sizeof(double));

	initial(A, b, N);

	t1 = clock();
	Arnoldi_Iteration(A, Q, H, b, N, iter);
	t2 = clock();

	printf(" Times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	return 0;
}

//***********************************************************************************

void initial(double *A, double *b, int N)
{
	int i;
	for (i=0; i<N*N; i++)
	{
		A[i] = sin(i);
		b[i] = cos(i);
	}
}

void norm(double *x, double *nrm, int N)
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
void matrix_matrix(double *A, double *x, double *b, int N)
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

void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter)
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
		norm(b, nrm, N*N);
		temp = *nrm;
		#pragma acc parallel loop independent
		for (k=0; k<N*N; k++)	Q[k] = b[k] / temp;
	}
	t2 = clock();
	printf(" First step times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		#pragma acc data copyin(A[0:N*N]) copy(Q[0:N*N*(iter+1)], H[0:(iter+1)*iter]) create(v[0:N*N], q[0:N*N])
		{
			// v= A*qi
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
			matrix_matrix(A, q, v, N);

			// h(j,i) = qj*v
			#pragma acc parallel loop independent
			for (j=0; j<=i; j++)
			{
				temp = 0.0;
				#pragma acc loop reduction(+:temp)
				for (k=0; k<N*N; k++)	temp += Q[N*N*j+k]*v[k];
				H[iter*j+i] = temp;
			}

			// v = v - \sum h(j,i)*qj
			#pragma acc parallel loop seq
			for (j=0; j<=i; j++)
			{
				#pragma acc loop independent
				for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
			}

			// h(i+1,i) = ||v||
			norm(v, nrm, N*N);
			H[iter*(i+1)+i] = *nrm;
			temp = *nrm;
			// qi+1 = v/h(i+1,i)
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / temp;
		} //end pragma acc
	}
	t2 = clock();
	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}

