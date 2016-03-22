#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void normalization(double *A, double *Q, double *Nrm, double *nrm_temp, int N);
void initial(double *A, int N);

int main()
{
	int N;
	printf(" Input N = ");
	scanf("%d", &N);
	printf(" N = %d \n", N);

	double *A, *Q, *Nrm;
	double *nrm_temp, t1, t2;

	A = (double *) malloc(N*N*sizeof(double));
	Q = (double *) malloc(N*N*sizeof(double));
	Nrm = (double *) malloc(N*sizeof(double));

	nrm_temp = (double *) malloc(1*sizeof(double));

	initial(A, N);

	t1 = clock();
	normalization(A, Q, Nrm, nrm_temp, N);
	t2 = clock();

	printf(" times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}

void initial(double *A, int N)
{
	int i;

	for (i=0; i<N; i++)	A[i] = sin(i);
}

//****************************************************************************

void norm(double *x, double *nrm, int N)
{
	int i;

	#pragma acc data present(x)
	{
		*nrm = 0.0;
		#pragma acc parallel loop reduction(+:nrm)
		for (i=0; i<N; i++)	*nrm += x[i]*x[i];
		*nrm = sqrt(*nrm);
	}
}

//****************************************************************************

void normalization(double *A, double *Q, double *Nrm, double *nrm_temp, int N)
{
	int i, j;

	double *q;

	q = (double *) malloc(N*sizeof(double));

	#pragma acc data copyin(A[0:N*N]) copyout(Q[0:N*N], Nrm[0:N]) create(q[0:N])
	{
		#pragma acc kernels loop independent
		for (i=0; i<N; i++)
		{
			#pragma acc loop independent
			for (j=0; j<N; j++)	q[j] = A[N*j+i];
			norm(q, nrm_temp, N);
			Nrm[i] = *nrm_temp;
			#pragma acc loop independent
			for (j=0; j<N; j++)	Q[N*j+i] = q[j] / Nrm[i];
		}
	}
}
