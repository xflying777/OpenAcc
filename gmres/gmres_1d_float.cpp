#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_vector(float *x, int N);
void matrix_vector(float *A, float *x, float *b, int N);
void initial(float *A, float *b, float *u, int N);
void gmres(float *A, float *x, float *b, int N, float tol);
float error(float *x, float *y, int N);

int main()
{
	int p, N;
	clock_t t1, t2;
	printf(" Please input (N = 2^p - 1) p = ");
	scanf("%d",&p);
	N = pow(2, p) - 1;
	printf(" N = %d \n", N);
	
	float *A, *x, *b, *u, tol;
	A = (float *) malloc(N*N*sizeof(float));
	x = (float *) malloc(N*sizeof(float));
	b = (float *) malloc(N*sizeof(float));
	u = (float *) malloc(N*sizeof(float));
	
	initial(A, b, u, N);	
	tol = 1.0e-2;
	t1 = clock();
	gmres(A, x, b, N, tol);
	t2 = clock();
	
	printf(" error = %e \n", error(x, u, N));
	printf(" times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
	return 0;
}

float error(float *x, float *y, int N)
{
	int i;
	float temp, error;
	error = 0.0;
	for(i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if(temp > error) error = temp;
	}
	return error;
}

void print_vector(float *x, int N)
{
	int i;
	for(i=0; i<N; i++)
	{
		printf(" %f ", x[i]);
	}
	printf("\n");
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

void initial(float *A, float *b, float *u, int N)
{
	int i, j;
	float h, h2, temp, x;
	
	h = M_PI/(N+1);
	h2 = h*h;
	
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			A[N*i+j] = 0.0;
		}
	}
	temp = -2.0/h2;
	for(i=0; i<N; i++)
	{
		x = (1+i)*h;
		u[i] = x*sin(x);
		
		x = (1+i)*h;
		b[i] = 2*cos(x) - x*sin(x);
		
		A[N*i+i] = temp;
	}
	temp = 1.0/h2;
	for(i=0; i<N-1; i++)
	{
		A[N*(i+1)+i] = temp;
		A[N*i+(i+1)] = temp;
	}
}

float norm(float *x, int N)
{
	int i;
	float norm;
	norm=0.0;
	for(i=0; i<N; i++)
	{
		norm += x[i]*x[i];
	}
	norm = sqrt(norm);
	return norm;
}

float inner_product(float *x, float *y, int N)
{
	int i;
	float temp;
	temp = 0.0;
	for(i=0; i<N; i++)
	{
		temp += x[i]*y[i];
	}
	return temp;
}

void matrix_vector(float *A, float *x, float *b, int N)
{
	int i, j;
	for(i=0; i<N; i++)
	{
		b[i] = 0.0;
		for(j=0; j<N; j++)
		{
			b[i] += A[N*i+j]*x[j];
		}
	}
}

void q_subQ(float *q, float *Q, int N, int iter)
{
	int i;
	for(i=0; i<N; i++)
	{
		q[i] = Q[(N+1)*i + iter];
	}
}

void subQ_v(float *Q, float *v, int N, int iter, float norm_v)
{
	int i;
	for(i=0; i<N; i++)
	{
		Q[(N+1)*i + iter] = v[i]/norm_v;
	}
}

void v_shift(float *v, float *q, float h, int N)
{
	int i;
	for(i=0; i<N; i++)
	{
		v[i] -= h*q[i];
	}
}

void backsolve(float *H, float *s, float *y, int N, int i)
{
	// i = iter
	int j, k;
	float temp;
	
	for(j=i; j>=0; j--)
	{
		temp = s[j];
		for(k=j+1; k<=i; k++)
		{
			temp -= y[k]*H[N*j+k];
		}
		y[j] = temp/H[N*j+j];
	}

}

void GeneratePlaneRotation(float dx, float dy, float *cs, float *sn, int i)
{
	float temp;
	if (dy == 0.0) 
	{
		cs[i] = 1.0;
		sn[i] = 0.0;
	} 
	else if (abs(dy) > abs(dx)) 
	{
		temp = dx / dy;
		sn[i] = 1.0 / sqrt( 1.0 + temp*temp );
		cs[i] = temp * sn[i];
	} 
	else 
	{
		temp = dy / dx;
		cs[i] = 1.0 / sqrt( 1.0 + temp*temp );
		sn[i] = temp * cs[i];
	}
}

/*void ApplyPlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
	Real temp  =  cs * dx + sn * dy;
	dy = -sn * dx + cs * dy;
	dx = temp;
}
*/

void gmres(float *A, float *x, float *b, int N, float tol)
{
	int i, j, k;
	float resid, beta, temp, *q, *v, *cs, *sn, *s, *y, *Q, *H;
	
	Q = (float *) malloc(N*(N+1)*sizeof(float));
	H = (float *) malloc((N+1)*N*sizeof(float));
	q = (float *) malloc(N*sizeof(float));
	v = (float *) malloc(N*sizeof(float));
	cs = (float *) malloc((N+1)*sizeof(float));
	sn = (float *) malloc((N+1)*sizeof(float));
	s = (float *) malloc((N+1)*sizeof(float));
	y = (float *) malloc((N+1)*sizeof(float));
	
	for(i=0; i<N+1; i++)
	{
		cs[i] = 0.0;
		sn[i] = 0.0;
		s[i] = 0.0;
		for(k=0; k<N; k++)
		{
			Q[N*k+i] = 0.0;
			H[N*i+k] = 0.0;
		}
	}
	
	beta = norm(b, N);
//	printf(" beta = %f \n", beta);
	for(i=0; i<N; i++)
	{
		Q[(N+1)*i+0] = b[i]/beta;
	}
	
	s[0] = beta;
	for (i = 0; i<N; i++) 
	{
  		q_subQ(q, Q, N, i);
//	  	printf(" norm(q) = %f \n", norm(q, N));
  		matrix_vector(A, q, v, N);
  		for (k=0; k<=i; k++) 
		{
			q_subQ(q, Q, N, k);
    		H[N*k+i] = inner_product(q, v, N);
    		v_shift(v, q, H[N*k+i], N);
  		}
  		
		H[N*(i+1)+i] = norm(v, N);
		subQ_v(Q, v, N, i+1, H[N*(i+1)+i]);
		
    	for (k = 0; k < i; k++)
      	{
      		//ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k))
      		temp = cs[k]*H[N*k+i] + sn[k]*H[N*(k+1)+i];
			H[N*(k+1)+i] = -1.0*sn[k]*H[N*k+i] + cs[k]*H[N*(k+1)+i];
			H[N*k+i] = temp;
		}
		
      	GeneratePlaneRotation(H[N*i+i], H[N*(i+1)+i], cs, sn, i);
      	//ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i))
		H[N*i+i] = cs[i]*H[N*i+i] + sn[i]*H[N*(i+1)+i];
		H[N*(i+1)+i] = 0.0;
      	//ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
      	temp = cs[i]*s[i];
      	s[i+1] = -1.0*sn[i]*s[i];
      	s[i] = temp;
      	resid = fabs(s[i+1]/beta);
     	
     	if (resid < tol | i == N-1) 
		{
			printf(" resid = %e \n", resid);
			printf(" converges at %d step \n", i);
			backsolve(H, s, y, N, i);
			for(j=0; j<N; j++)
			{
				temp = 0.0;
				for(k=0; k<=i; k++)
				{
					temp += Q[(N+1)*j+k]*y[k];
				}
				x[j] = temp;
			}
			break;
      	}
	}
}



