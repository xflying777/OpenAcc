
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

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
	int i;
	double temp;
	temp = 0.0;
	for (i=0; i<N; i++)	temp += x[i]*x[i];
	*nrm = sqrt(temp);
}

//***********************************************************************************

void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, t1, t2;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));

	norm(b, nrm, N*N);
	for (k=0; k<N*N; k++)	Q[k] = b[k] / *nrm;

	t1 = clock();
	for (i=0; i<iter; i++)
	{
		// v= A*qi
		for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, q, N, 0.0, v, N);

		// h(j,i) = qj*v
		for (j=0; j<=i; j++)
		{
			H[iter*j+i] = 0.0;
			for (k=0; k<N*N; k++)	H[iter*j+i] += Q[N*N*j+k]*v[k];
		}

		// v = v - \sum h(j,i)*qj
		for (j=0; j<=i; j++)
		{
			for (k=0; k<N*N; k++)	v[k] -= H[iter*j+i]*Q[N*N*j+k];
		}

		// h(i+1,i) = ||v||
		norm(v, nrm, N*N);
		H[iter*(i+1)+i] = *nrm;
		// qi+1 = v/h(i+1,i)
		for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / *nrm;
	}
	t2 = clock();
	printf(" Arnoldi times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
}

