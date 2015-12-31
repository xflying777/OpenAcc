#include "cublas_v2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 Note : cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n)
 		means matrix C = B * A
 		
 		In cublasDgemm ,we should set matrix as array.
		Here we do the discrete sine transform by setting the sine matrix S and then do S*b.
*/
float error(float *x, float *y, N);
void initial_sourse(float *b, int N);
void sine_matrix(float *S, int N);
void cublas3(float *a, float *b, float *c, int n);

int main()
{
	int N;
	float *S, *b, *c, *temp;

	printf("Input size N = ");
	scanf("%d",&N);

	S = (float *) malloc(N*N*sizeof(float));
	b = (float *) malloc(N*N*sizeof(float));
	c = (float *) malloc(N*N*sizeof(float));
	temp = (float *) malloc(N*N*sizeof(float));
	
	initial_sourse(b, N);
	sine_matrix(S, N);
	
	#pragma acc data copyin(b[0:N*N], S[0:N*N]), create(temp[0:N*N]), copyout(c[0:N*N])
	{
		cublas3(b, S, temp, N);
		cublas3(temp, S, c, N);
	}
	
	printf("\n Error = %f \n", error(b, c, N));


	return 0;
}

float error(float *x, float *y, N)
{
	int i, j;
	float error, temp;
	error = 0.0;
	
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp = fabs(x[N*i+j] - y[N*i+j]);
			if (temp > error) error = temp;
		}
	}
	
	return error;
}
void initial_sourse(float *b, int N)
{
	int i, j;
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)	b[N*i+j] = i + j;
	}
}

void sine_matrix(float *S, int N)
{
	int i, j;
	float x, y, s;
	s = sqrt(2.0/(N+1));
	
	for (i=0; i<N; i++)
	{
		x = (i+1)*h;
		for (j=0; j<N; j++)
		{
			y = (j+1)*h;
			S[N*i+j] = sin(x*y*M_PI/(N+1));
		}
	}
}

void cublas3(float *a, float *b, float *c, int n)
{
	cublasHandle_t handle;
	float alpha = 1.0;
	float beta = 0.0;
	cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n);
	cublasDestroy(handle);
}



