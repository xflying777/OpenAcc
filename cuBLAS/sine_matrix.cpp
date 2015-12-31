#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initial_vector(float *x, int N);
void sine_matrix(float *S, int N);
void matrix_vector(float *A, float *b, float *c, int N);
float error(float *a, float *b, int N);
void print_vector(float *a, int N);
void inverse_sine(float *S, float *data_in, float *data_out, int N);


int main()
{
	int N;
	float *b, *c, *S, *temp;
	
	printf(" Please input N = ");
	scanf("%d",&N);
	
	b = (float *) malloc(N*sizeof(float));
	c = (float *) malloc(N*sizeof(float));
	temp = (float *) malloc(N*sizeof(float));
	S = (float *) malloc(N*N*sizeof(float));
	
	
	initial_vector(b, N);	
	sine_matrix(S, N);
	
	matrix_vector(S, b, temp, N);
	matrix_vector(S, temp, c, N);
	
	printf(" Error = %e \n", error(b, c, N));
	
	return 0;
}

void print_vector(float *a, int N)
{
	int i;
	for (i=0; i<N; i++)	printf(" [%d] = %f \n", i, a[i]);
}

float error(float *a, float *b, int N)
{
	int i;
	float error, temp;
	error = 0.0;
	
	for (i=0; i<N; i++)
	{
		temp = fabs(a[i] - b[i]);
		if (temp > error)	error = temp;
	}
	return error;
}

void initial_vector(float *x, int N)
{
	int i;
	for (i=0; i<N; i++)
	{
		x[i] = 1.0*i;
	}
}

void sine_matrix(float *S, int N)
{
	int i, j;
	float x, y, s;
	
	s = sqrt(2.0/(N+1));
	for (i=0; i<N; i++)
	{
		x = i + 1;
		for (j=0; j<N; j++)
		{
			y = j + 1;
			S[N*i+j] = s*sin(x*y*M_PI/(N+1));
		}
	}
}

void matrix_vector(float *A, float *b, float *c, int N)
{
	int i, j;
	// Compute matrix multiple vector.
	// c = A*b
	for (i = 0; i<N; ++i) 
	{
		c[i] = 0.0;
		for (j = 0; j<N; ++j) 
		{
			c[i] += A[N*i+j] * b[j];
		}
	}
}

