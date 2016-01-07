/*
	GMRES iteration sovler.
	
	Exact solution	: u(x) = x*sin(x).
	Problem			: -u''(x) = 2*cos(x)-x*sin(x);
					: u(0) = u(1) = 0.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_matrix(float *x, int N);
void Exact_Solution(float *u, int Nx);
void Exact_Source(float *b, int Nx);
void A_matrix(float *A, int Nx);
float norm(float *x, int Nx);

int main()
{
	int p, Nx; 
	float *u, *A, *x, *b;
	clock_t t1, t2;
	
	// Initialize the numbers of discrete points.
	// Here we consider Nx = Ny. 
	printf(" Please input (Nx = 2^p - 1) p = ");
	scanf("%d",&p);
	Nx = pow(2, p) - 1;
	printf(" Nx = %d \n", Nx);
	
	A = (float *) malloc(Nx*Nx*sizeof(float));
	b = (float *) malloc(Nx*sizeof(float));
	x = (float *) malloc(Nx*sizeof(float));
	u = (float *) malloc(Nx*sizeof(float));
	
	A_matrix(A, Nx);
	printf(" A = \n");
	print_matrix(A, Nx);
	printf(" norm(A) = %f \n", norm(A, Nx*Nx));
	
	return 0;
	
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

void Exact_Solution(float *u, int Nx)
{
	int i;
	float h, x;
	h = 1/(Nx+1);
	for(i=0; i<Nx; i++)
	{
		x = (1+i)*h;
		u[i] = x*sin(x);
	}
}

void Exact_Source(float *b, int Nx)
{
	int i;
	float h, x;
	h = 1/(Nx+1);
	for(i=0; i<Nx; i++)
	{
		x = (1+i)*h;
		b[i] = 2*cos(x) - x*sin(x);
	}
}

void A_matrix(float *A, int Nx)
{
	int i, j;
	float h;
	h = 1/(Nx+1);
	for(i=0; i<Nx; i++)
	{
		for(j=0; j<Nx; j++)
		{
			A[Nx*i+j] = 0.0;
		}
	}
	
	for(i=0; i<Nx; i++)
	{
		A[Nx*i+i] = 2.0;
	}
	
	for(i=0; i<Nx-1; i++)
	{
		A[Nx*(i+1)+i] = -1.0;
		A[Nx*i+(i+1)] = -1.0;
	}
}

float norm(float *x, int Nx)
{
	int i;
	float norm;
	norm = 0.0;
	for(i=0; i<Nx; i++)
	{
		norm += x[i]*x[i];
	}
	norm = sqrt(norm);
	
	return norm;
}

void matrix_vector(float *A, float *data_in, float data_out, int Nx)
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

void Q_subvector(float *Q, float *v, int Nx, int num_row)
{
	int i;
	for(i=0; i<Nx; i++)
	{
		Q[Nx*num_row + i] = v[i];
	}
}

void Q_subnormal(float *Q, float norm, int Nx, int num_row)
{
	int i;
	for(i=0; i<Nx; i++)
	{
		Q[Nx*num_row + i] = Q[Nx*num_row + i]/norm;
	}
}

void gmres(float *A, float *x, float *b, int Nx)
{
	int i, j, iter;
	float beta, h;
	float *Q, *H, *v;
	
	Q = (float *) malloc(Nx*(Nx+1)*sizeof(float));
	H = (float *) malloc((Nx+1)*Nx*sizeof(float));
	
	h = 1/(Nx+1);
	beta = norm(b, Nx);
	Q_subvector(Q, b, Nx, 0);
	Q_subnormal(Q, beta, Nx, 0);
	
	for(iter=0; iter<Nx; iter++)
	{
		
	} 
}
