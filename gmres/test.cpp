#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_vector(float *x, int N);
void print_matrix(float *x, int N);

void backsolve(float *H, float *s, float *y, int iter)
{
	int j, k;
	float temp;
	
	for(j=iter; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=iter; k++)
		{
			temp -= y[k]*H[iter*j+k];
		}
		y[j] = temp/H[iter*j+j];
	}	
}

/*void backsolve(float *H, float *y, float *s, int iter)
{  
	int i, j;
	for (i=iter; i>= 0; i--) 
	{
		y[i] /= H[iter*i+iter];
		for (j= i-1; j>=0; j--)
		{
			y[j] -= H[iter*j+i]*y[i];
		}
	}
}
*/
void matrix_vector(float *A, float *data_in, float *data_out, int N)
{
	int i, j;
	for(i=0; i<N; i++)
	{
		data_out[i] = 0.0;
		for(j=0; j<N; j++)
		{
			data_out[i] += A[N*i+j]*data_in[j];
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
	printf("\n");
}

void print_vector(float *x, int N)
{
	int i;
	for(i=0; i<N; i++)	printf(" %f ", x[i]);
	printf("\n \n");
}

int main()
{
	int i, j, N;
	float *H, *s, *y, *u;
	
	N = 5;
	
	H = (float *) malloc((N+1)*N*sizeof(float));
	s = (float *) malloc(N*sizeof(float));
	y = (float *) malloc(N*sizeof(float));
	u = (float *) malloc(N*sizeof(float));
	
	for(i=0; i<N; i++)
	{
		u[i] = 1.0*i + 1;
		for(j=i; j<N; j++)
		{
			H[N*i+j] = i + j + 1;
		}
	}
	
	printf(" H matrix : \n");
	print_matrix(H, N);
	printf(" u vector: \n");
	print_vector(u, N);
	matrix_vector(H, u, s, N);
	printf(" s vector: \n");
	print_vector(s, N);
	backsolve(H, s, y, N);
	printf(" y vector: \n");
	print_vector(y, N);
	
	
	return 0;
	
}


