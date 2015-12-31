#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void print_vector(float *a, int N);
void print_matrix(float *x, int N);
void sine_matrix(float *S, int N);


int main()
{
	int i, N;
	float *S;
	
	N = 20;
	S = (float *) malloc(N*N*sizeof(float));
	sine_matrix(S, N);
	
	printf(" S matrix: \n");
	print_matrix(S, N);
	
	return 0;
}

void print_vector(float *a, int N)
{
	int i;
	for (i=0; i<N; i++)	printf(" [%d] = %f \n", i, a[i]);
}

void sine_matrix(float *S, int N)
{
	int i, j;
	float x, y;

	for (i=0; i<N; i++)
	{
		x = i + 1;
		for (j=0; j<N; j++)
		{
			y = j + 1;
			S[N*i+j] = sin(x*y*M_PI/(N+1));
		}
	}
}

void print_matrix(float *x, int N)
{
	int i, j;
	for(i=0;i<N;i++)
	{
		for (j=0;j<N;j++) printf(" %f ", x[N*i+j]);
		printf("\n");
	}
}
