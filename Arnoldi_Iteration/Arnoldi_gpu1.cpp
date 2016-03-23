
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

	#pragma acc data present(x)
	{
		temp = 0.0;
		#pragma acc parallel loop reduction(+:temp)
		for (i=0; i<N; i++)	temp += x[i]*x[i];
		*nrm = sqrt(temp);
	}
}

//***********************************************************************************

void Arnoldi_Iteration(double *A, double *Q, double *H, double *b, int N, int iter)
{
	int i, j, k;
	double *v, *q;
	double *nrm, *dot, temp;

	v = (double *) malloc(N*N*sizeof(double));
	q = (double *) malloc(N*N*sizeof(double));

	nrm = (double *) malloc(1*sizeof(double));
	dot = (double *) malloc(1*sizeof(double));

	#pragma acc data copyin(b[0:N*N]) copyout(Q[0:N*N*(iter+1)])
	{
		norm(b, nrm, N*N);
		#pragma acc parallel loop independent
		for (k=0; k<N*N; k++)	Q[k] = b[k] / *nrm;
	}

	for (i=0; i<iter; i++)
	{
		// v= A*qi
		for (k=0; k<N*N; k++)	q[k] = Q[N*N*i+k];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, q, N, 0.0, v, N);

		// h(j,i) = qj*v
		#pragma acc data copy(Q[0:N*N*(iter+1)], H[0:(iter+1)*iter]) copyin(v[0:N*N])
		{ 
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
			// qi+1 = v/h(i+1,i)
			#pragma acc parallel loop independent
			for (k=0; k<N*N; k++)	Q[N*N*(i+1)+k] = v[k] / *nrm;
		} //end pragma acc
	}
}

