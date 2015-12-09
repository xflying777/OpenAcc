#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void FFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N, int L);
void Initial(double *x, double *y, int N);
int Generate_N(int p);

int main()
{
	int p, N, L;
	double *y_r, *y_i, *x_r, *x_i;
	clock_t t1, t2;
	
	printf(" Please input p ( N = 2^p - 1 ) = ");
	scanf("%d",&p);
	N = Generate_N(p);
	printf(" N=2^%d - 1 = %d\n",p,N);
	L = 2*N + 2;
	
	x_r = (double *) malloc(N*sizeof(double));
	x_i = (double *) malloc(N*sizeof(double));
	y_r = (double *) malloc(L*sizeof(double));
	y_i = (double *) malloc(L*sizeof(double));
	
	Initial(x_r, x_i, N);
	t1 = clock();
	FFTr2(x_r, x_i, y_r, y_i, N, L);
	t2 = clock();
	printf(" Fast FDSTR2: %f secs\n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
/*	for(k=0;k<N;k++)
	{
		printf(" %d = %f \n", k, y_r[k]);
	}
*/
	return 0;
} 

void Initial(double *x, double *y, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		x[n] = n;
		y[n] = 0.0;
	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	N = N - 1;
	return N;
}

void FFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N, int L)
{
	int i, j, k, n, M;
	double t_r, t_i;
	
	// expand y[n] to 2N+2-points from x[n]
	y_r[0] = y_i[0] = 0.0;
	y_r[N+1] = y_i[N+1] = 0.0;
	for(i=0;i<N;i++)
	{
		y_r[i+1] = x_r[i];
		y_i[i+1] = x_i[i];
		y_r[N+i+2] = -1.0*x_r[N-1-i];
		y_i[N+i+2] = -1.0*x_i[N-1-i];
	}
	
	
	i = j = 0;
	while(i < L)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
			t_i = y_i[i];
			y_r[i] = y_r[j];
			y_i[i] = y_i[j];
			y_r[j] = t_r;
			y_i[j] = t_i;
		}
		M = L/2;
		while(j >= M & M > 0)
		{
			j = j - M;
			M = M / 2;
		}
		j = j + M;		
		i = i + 1;
	}
	// Butterfly structure
	double theta, w_r, w_i;
	n = 2;
	while(n <= L)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<L;i+=n)
			{
				j = i + n/2;
				t_r = w_r * y_r[j] - w_i * y_i[j];
				t_i = w_r * y_i[j] + w_i * y_r[j];
				

				y_r[j] = y_r[i] - t_r;
				y_i[j] = y_i[i] - t_i;
				y_r[i] = y_r[i] + t_r;
				y_i[i] = y_i[i] + t_i;

			}
		}
		n = n * 2;
	}
	
	// After fft(y[k]), Y[k] = fft(y[k]), Sx[k] = i*Y[k+1]/2
	for(k=0;k<N;k++)
	{
		y_r[k] = -1.0*y_i[k+1]/2;
	}
	
}


