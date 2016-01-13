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
float error(float *x, float *y, int Nx);
void gmres(float *A, float *x, float *b, int Nx, float tol);

int main()
{
	int p, Nx; 
	float *u, *A, *x, *b, tol;
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
	
	tol = 1.0e-6;
	Exact_Solution(u, Nx);
	Exact_Source(b, Nx);
	A_matrix(A, Nx);
	gmres(A, x, b, Nx, tol);
	
	printf(" error = %f \n", error(x, u, Nx));	
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
	printf("\n");
}

void Exact_Solution(float *u, int Nx)
{
	int i;
	float h, x;
	h = M_PI/(Nx+1);
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
	h = M_PI/(Nx+1);
	for(i=0; i<Nx; i++)
	{
		x = (1+i)*h;
		b[i] = 2*cos(x) - x*sin(x);
	}
}

float error(float *x, float *y, int Nx)
{
	int i;
	float error, temp;
	error = 0.0;
	for(i=0; i<Nx; i++)
	{
		temp = abs(x[i] - y[i]);
		if(temp > error) error = temp;
	}
	return error;
}

void A_matrix(float *A, int Nx)
{
	int i, j;
	float h, h2;
	h = M_PI/(Nx+1);
	h2 = h*h;
	for(i=0; i<Nx; i++)
	{
		for(j=0; j<Nx; j++)
		{
			A[Nx*i+j] = 0.0;
		}
	}
	
	for(i=0; i<Nx; i++)
	{
		A[Nx*i+i] = -2.0/h2;
	}
	
	for(i=0; i<Nx-1; i++)
	{
		A[Nx*(i+1)+i] = 1.0/h2;
		A[Nx*i+(i+1)] = 1.0/h2;
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

void vector_subQ(float *Q, float *v, int Nx, int num_row)
{
	int i;
	for(i=0; i<Nx; i++)
	{
		v[i] = Q[Nx*num_row + i];
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

float inner_product(float *data1, float *data2, int Nx)
{
	int i;
	float value;
	value = 0.0;
	for(i=0; i<Nx; i++)
	{
		value += data1[i]*data2[i];
	}
	return value;
}

void vector_shift(float *x, float a, float *y, int iter)
{
	int i;
	for(i=0; i<=iter; i++)
	{
		x[i] = x[i] - a*y[i];
	}
}

void GeneratePlaneRotation(float *dx, float *dy, float *cs, float *sn, int i, int Nx)
{
	// i = iter
	// dx : iter, dy : iter + 1
	float temp;
	if (dy[Nx*(i+1)+i] == 0.0) 
	{
		cs[i] = 1.0;
		sn[i] = 0.0;
	} 
	else if (abs(dy[Nx*(i+1)+i]) > abs(dx[Nx*i+i])) 
	{
		temp = dx[Nx*i+i] / dy[Nx*(i+1)+i];
		sn[i] = 1.0 / sqrt( 1.0 + temp*temp);
		cs[i] = temp * sn[i];
	} 
	else 
	{
		temp = dy[Nx*(i+1)+i] / dx[Nx*i+i];
		cs[i] = 1.0 / sqrt( 1.0 + temp*temp);
		sn[i] = temp * cs[i];
	}
}

void ApplyPlaneRotationH(float *dx, float *dy, float *cs, float *sn, int i, int Nx)
{
	// i = iter
	// dx : iter, dy : iter+1
	float temp;
	temp  =  cs[i]*dx[Nx*i+i] + sn[i]*dy[Nx*(i+1)+i];
	dy[Nx*(i+1)+i] = -sn[i]*dx[Nx*i+i] + cs[i]*dy[Nx*(i+1)+i];
	dx[Nx*i+i] = temp;
}

void ApplyPlaneRotationS(float *dx, float *dy, float *cs, float *sn, int i)
{
	// i = iter
	// dx : iter, dy : iter+1
	float temp;
	temp  =  cs[i]*dx[i] + sn[i]*dy[i+1];
	dy[i+1] = -sn[i]*dx[i] + cs[i]*dy[i+1];
	dx[i] = temp;
}

// y = H \ s.
void backsolver(float *H, float *s, float *y, int iter)
{
	int j, k;
	float temp;
	for(j=iter; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=iter; k++)
		{
			temp -= y[k]*H[iter*k+j];
		}
		y[j] = temp/H[iter*j+j];
	}	
}

void gmres(float *A, float *x, float *b, int Nx, float tol)
{
	int i, j, iter;
	float beta, error, a;
	float *Q, *H, *v, *cs, *sn, *s, *y, *temp;
	
	s = (float *) malloc(Nx*sizeof(float));
	cs = (float *) malloc(Nx*sizeof(float));
	sn = (float *) malloc(Nx*sizeof(float));
	Q = (float *) malloc(Nx*(Nx+1)*sizeof(float));
	H = (float *) malloc((Nx+1)*Nx*sizeof(float));
	v = (float *) malloc(Nx*sizeof(float));
	temp = (float *) malloc(Nx*sizeof(float));
	
	for(j=0; j<Nx; j++)
	{
		cs[j] = 0.0;
		sn[j] = 0.0;
		s[j] = 0.0;
	}

	beta = norm(b, Nx);
	Q_subvector(Q, b, Nx, 0);
	Q_subnormal(Q, beta, Nx, 0);
	
	s[0] = beta;
	for(iter=0; iter<Nx; iter++)
	{
		vector_subQ(Q, temp, Nx, iter);
		matrix_vector(A, temp, v, Nx);
		for(j=0; j<=iter; j++)
		{
			vector_subQ(Q, temp, Nx, j);
			a = inner_product(temp, v, Nx);
			H[Nx*j + iter] = a;
			vector_shift(v, a, temp, iter);
		}

		a = norm(v, Nx);
		H[Nx*(iter+1) + iter] = a;
		Q_subvector(Q, v, Nx, iter+1);
		Q_subnormal(Q, a, Nx, iter+1);

		for (j=0; j<iter; j++)
      	{
			ApplyPlaneRotationH(H, H, cs, sn, j, Nx);
		}
		
		GeneratePlaneRotation(H, H, cs, sn, iter, Nx);
		ApplyPlaneRotationH(H, H, cs, sn, iter, Nx);
		ApplyPlaneRotationS(s, s, cs, sn, iter);
	    
		error = abs(s[iter+1])/beta;
		if (error <= tol | iter == Nx-1)
		{
			backsolver(H, s, y, iter);
			for(i=0; i<iter; i++)
			{
				x[i] = 0.0;
				for(j=0; j<iter; j++)
				{
					x[i] += Q[Nx*j+i]*y[j];
				} 
			}
			break;
		}
	}
	printf(" H matrix : \n");
	print_matrix(H, iter);
}
