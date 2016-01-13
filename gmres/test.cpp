#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void backsolver(float *H, float *s, float *y, int iter)
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

void matrix_vector(float *A, float *data_in, float *data_out, int Nx)
{
	int i, j;
	for(i=0; i<Nx; i++)
	{
		data_out[i] = 0.0;
		for(j=0; j<Nx; j++)
		{
			data_out[i] += A[Nx*i+j]*data_in[j];
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
		if(i == 1) break;
	}
	
	printf("H matrix : \n");
	print_matrix(H, N);
/*	matrix_vector(H, u, s, N);
	backsolver(H, s, y, N);
	printf(" H matrix : \n");
	print_matrix(H, N);
	printf(" u vector: \n");
	print_vector(u, N);
	printf(" s vector: \n");
	print_vector(s, N);
	printf(" y vector: \n");
	print_vector(y, N);
*/	return 0;
	
}


