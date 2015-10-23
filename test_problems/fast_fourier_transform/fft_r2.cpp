#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void cpuFFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N);
void gpuFFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N, int p);
void error(double *y_r, double *y_i, double *z_r, double *z_i, int N);
void Initial(double *x, double *y, int N);
void Print_Complex_Vector(double *y_r, double *y_i, int N);
int Generate_N(int p);

int main()
{
	int i, p, N;
	double *y_r, *y_i, *z_r, *z_i, *x_r, *x_i, cpu_times, gpu_times;
	clock_t t1, t2;
	
	printf("Please input N = 2^p, p = ");
	scanf("%d",&p);
	N = Generate_N(p);
	printf("N = 2^%d = %d \n", p, N);
	
	x_r = (double *) malloc(N*sizeof(double));
	x_i = (double *) malloc(N*sizeof(double));
	y_r = (double *) malloc(N*sizeof(double));
	y_i = (double *) malloc(N*sizeof(double));
	z_r = (double *) malloc(N*sizeof(double));
	z_i = (double *) malloc(N*sizeof(double));
	
	Initial(x_r, x_i, N);
	
	t1 = clock();
	cpuFFTr2(x_r, x_i, y_r, y_i, N);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	gpuFFTr2(x_r, x_i, z_r, z_i, N, p);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;	
	
	printf("cpu times = %f \n gpu_times = %f \n cpu times / gpu times = %f \n", cpu_times, gpu_times, cpu_times/gpu_times);
	error(y_r, y_i, z_r, z_i, N);
//	Print_Complex_Vector(y_r, y_i, N);
	
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

void Print_Complex_Vector(double *y_r, double *y_i, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		if (y_i[n] >=0) printf("%d : %f +%f i\n", n, y_r[n], y_i[n]);
		else printf("%d : %f %f i\n", n, y_r[n], y_i[n]);
	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	return N;
}

void error(double *y_r, double *y_i, double *z_r, double *z_i, int N)
{
	int i;
	for(i=0;i<N;i++)
	{
		if(y_r[i] != z_r[i] | y_i[i] != z_i[i])
		{
			printf("error \n");
			exit(1);
		}
	}
	printf("OpenACC test was successful! \n");
		
}

void gpuFFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N, int p)
{
	// input : x = x_r + i * x_i
	// output: y = y_r + i * y_i
	int i, j, M, *temp, *size_n;
	double t;
	
	temp = (int *) malloc(N*sizeof(int));
	size_n = (int *) malloc((p+1)*sizeof(int));
	
	temp[0] = 0;
	for(i=0;i<p+1;i++) size_n[i] = pow(2, i);
	
	for(i=0;i<N;i++)
	{
		y_r[i] = x_r[i];
		y_i[i] = x_i[i];
	}

	#pragma acc data copyin(size_n[0:p+1]) copy(temp[0:N], y_r[0:N])
	#pragma acc kernels
	{
	#pragma acc loop independent
	for(M=N/2, j=0;M>0;M=M/2, j++)
	{
	#pragma acc loop independent
		for(i=size_n[j];i<size_n[j+1];i++)
		{
			temp[i] = temp[i-size_n[j]] + M;
		}
	}
	#pragma acc loop independent
	for(i=0;i<N;i++)
	{
		if(i < temp[i])
		{
			// swap y[i], y[j]
			t = y_r[i];
			y_r[i] = y_r[temp[i]];
			y_r[temp[i]] = t;
		}
	}
	}
	
	// Butterfly structure
	int n, k;
	double theta, w_r, w_i, t_r, t_i;
	n = 2;
	while(n <= N)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<N;i+=n)
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
	
}


void cpuFFTr2(double *x_r, double *x_i, double *y_r, double *y_i, int N)
{
	// input : x = x_r + i * x_i
	// output: y = y_r + i * y_i
	int k, n, i, j, M;
	double t_r, t_i;
	
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}

	i = j = 0;
	while(i < N)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
			y_r[i] = y_r[j];
			y_r[j] = t_r;
		}
		M = N/2;
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
	while(n <= N)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<N;i+=n)
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
}

