#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	
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

void subQ_v(float *Q, float *v, int N, int iter, float *norm_v)
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

int	GMRES(float *A, float *x, float *b, int N, float tol)
{
	int i, k, p;
	float resid, beta, *q, *v, *cs, *sn, *s, *Q, *H;
	
	Q = (float *) malloc(N*(N+1)*sizeof(float));
	H = (float *) malloc((N+1)*N*sizeof(float));
	q = (float *) malloc(N*sizeof(float));
	v = (float *) malloc(N*sizeof(float));
	s = (float *) malloc(N*sizeof(float));
	
	beta = norm(b, N);
	for(p=0; p<N+1; p++)
	{
		Q[N*p+0] = b[i]/beta;
	}
	
	s[0] = beta;
	for (i = 0; i<N; i++) 
	{
  		q_subQ(q, Q, N, i);
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
      		ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
		}
		
      	GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
      	ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
      	ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
      
      	if ((resid = abs(s(i+1)) / normb) < tol) 
		{
        	Update(x, i, H, s, v);
        	tol = resid;
        	max_iter = j;
        	delete [] v;
        	return 0;
      	}
	}
    Update(x, m - 1, H, s, v);
    r = M.solve(b - A * x);
    beta = norm(r);
    if ((resid = beta / normb) < tol) 
	{
    	tol = resid;
      	max_iter = j;
      	delete [] v;
      	return 0;
    }
}

tol = resid;
delete [] v;
return 1;
} 

template<class Real> 
void GeneratePlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
	if (dy == 0.0) 
	{
		cs = 1.0;
		sn = 0.0;
	} 
	else if (abs(dy) > abs(dx)) 
	{
		Real temp = dx / dy;
		sn = 1.0 / sqrt( 1.0 + temp*temp );
		cs = temp * sn;
	} 
	else 
	{
		Real temp = dy / dx;
		cs = 1.0 / sqrt( 1.0 + temp*temp );
		sn = temp * cs;
	}
}

template<class Real> 
void ApplyPlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
	Real temp  =  cs * dx + sn * dy;
	dy = -sn * dx + cs * dy;
	dx = temp;
}

